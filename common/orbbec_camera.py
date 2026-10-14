import argparse
import ctypes
import importlib.util
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import os
import signal
import sys
import time

import cv2
import numpy as np


def _preload_bundled_sdk():
    """优先加载 wheel 自带的 libOrbbecSDK, 必须在 import pyorbbecsdk 之前调用.

    ROS orbbec_camera 驱动将旧版 libOrbbecSDK(如 2.8.6)装入 /opt/ros/humble/lib,
    且该路径位于 LD_LIBRARY_PATH, 会抢先于 wheel 自带的新版(2.9.3)被动态链接,
    导致 undefined symbol 导入失败. 此处用 RTLD_GLOBAL 预加载配套版本, 链接器
    按 soname 去重后即绑定到本版本, 不影响 ROS 驱动进程继续使用旧版.
    """
    spec = importlib.util.find_spec('pyorbbecsdk')
    if spec is None or not spec.submodule_search_locations:
        return
    lib = os.path.join(spec.submodule_search_locations[0], 'libOrbbecSDK.so.2')
    if os.path.exists(lib):
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


_preload_bundled_sdk()
from pyorbbecsdk import (Config, Context, OBFormat, Pipeline,  # noqa: E402
                         OBSensorType, VideoStreamProfile)

# 深度显示范围(米): 线性映射到灰度(近=黑, 远=白), 超出按端点截断.
# 与策略 min_depth/max_depth(dex_config.yaml depth 块)保持一致, 便于对照观测
DISPLAY_DEPTH_MIN_M = 0.0
DISPLAY_DEPTH_MAX_M = 2.5


class OrbbecDepthCamera:
    """单线程阻塞式深度采集, 接口形态对齐 instinct_onboard 的 RealsenseCamera.

    __init__ 完成设备选择/流配置/启动/预热, 之后在调用线程内以
    read_depth() 阻塞取帧(pipeline.wait_for_frames), 无内部线程。
    """

    def __init__(self,
                 width: int = 640,
                 height: int = 360,
                 fps: int = 30,
                 serial: str | None = None):
        ctx = Context()
        dev_list = ctx.query_devices()
        if dev_list.get_count() == 0:
            raise RuntimeError("未检测到 Orbbec 设备 (检查 USB 连接与 udev 规则)")
        self.device = None
        for i in range(dev_list.get_count()):
            d = dev_list.get_device_by_index(i)
            if serial is None or d.get_device_info().get_serial_number() == serial:
                self.device = d
                break
        if self.device is None:
            raise RuntimeError(f"未找到序列号为 {serial} 的 Orbbec 设备")

        self.pipeline = Pipeline(self.device)
        profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR)

        # 依次尝试: 精确分辨率 -> 仅宽度匹配 -> 默认配置
        profile = None
        for w, h in ((width, height), (width, 0)):
            try:
                profile = profiles.get_video_stream_profile(
                    w, h, OBFormat.UNKNOWN_FORMAT, fps)
                break
            except Exception:
                continue
        if profile is None:
            profile = profiles.get_default_video_stream_profile()
            print(f"警告: 未找到 {width}x{height}@{fps} 深度流, 回退到默认配置")

        vp: VideoStreamProfile = profile.as_video_stream_profile()
        self.width = vp.get_width()
        self.height = vp.get_height()
        self.fps = vp.get_fps()

        self.config = Config()
        self.config.enable_stream(profile)
        self.pipeline.start(self.config)

        # 预热: 丢弃首帧(曝光/时戳未稳定)
        self.pipeline.wait_for_frames(1000)
        print(f"Orbbec 深度流已启动: {self.width}x{self.height}@{self.fps}")

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray | None:
        """阻塞读取一帧深度, 返回 float32 深度图(单位: 米), 失败返回 None"""
        if self.pipeline is None:  # stop() 之后
            return None
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return None
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
        # get_depth_scale(): 原始值 -> 毫米
        return depth_data.astype(np.float32) * depth_frame.get_depth_scale() / 1000.0

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


# ======================================================================
# 子进程 + 共享内存 (移植自 instinct_onboard/camera_base.py)
# ======================================================================

class MpSharedHeader(ctypes.Structure):
    """SHM 块头部 v2: 时间戳/写锁/终止信号/帧计数/原始与处理帧信息.

    布局: [header][原始深度 w*h float32][处理帧 proc_len float32(可选)]
    """
    _fields_ = [
        ("timestamp", ctypes.c_double),          # 8B: 写入时刻 time.time()
        ("proc_ms", ctypes.c_double),            # 8B: 最新一帧预处理耗时(毫秒)
        ("writer_status", ctypes.c_uint32),      # 4B: 0 空闲, 1 写入中
        ("writer_termination_signal", ctypes.c_uint32),  # 4B: 0 存活, 1 应退出
        ("frame_counter", ctypes.c_uint32),      # 4B: 单调递增帧计数
        ("width", ctypes.c_uint32),              # 4B: 原始流宽度
        ("height", ctypes.c_uint32),             # 4B: 原始流高度
        ("proc_valid", ctypes.c_uint32),         # 4B: 1=本帧处理帧有效
        ("proc_len", ctypes.c_uint32),           # 4B: 处理帧元素数(0=未启用)
        # 显式补齐: 字段共 44B, 加 4B 填充 = 48B(8B 对齐), 不依赖隐式尾部对齐
        ("_pad", ctypes.c_uint32),               # 4B
    ]


SIZE_OF_MP_SHARED_HEADER = ctypes.sizeof(MpSharedHeader)
assert SIZE_OF_MP_SHARED_HEADER == 48

WRITER_FREQ_LOG_FRAMES = 300  # 子进程每 N 帧打印一次采帧频率

try:
    _libgcc = ctypes.CDLL("libgcc_s.so.1")
    _sync_synchronize = _libgcc.__sync_synchronize
except (OSError, AttributeError):
    _sync_synchronize = None


def _release_barrier() -> None:
    if _sync_synchronize is not None:
        _sync_synchronize()


def _acquire_barrier() -> None:
    if _sync_synchronize is not None:
        _sync_synchronize()


def camera_subprocess_func(conn, width, height, fps, serial, shm_name, preprocess_func=None):
    """子进程: 持有相机, 阻塞取帧, 原始帧与预处理帧一并写入 SHM.

    握手: 先通过 pipe 回传 (实际流分辨率, 处理帧元素数)(父进程据此分配
    SHM 大小), 待父进程回执后再开始写帧. preprocess_func 为 None 时
    proc_len=0, SHM 只含原始帧, 处理帧槽位不存在.

    注意: preprocess_func 在写锁外执行(可能耗时), 父进程的读操作
    不会被预处理拖慢. 预处理异常时本帧 proc_valid=0(父进程侧表现为
    无处理帧, 策略回退中性填充), 原始帧照常输出.
    """
    stop = {"flag": False}

    def _on_sigterm(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    camera = OrbbecDepthCamera(width, height, fps, serial)

    # 用一帧(丢弃)标定预处理输出长度, 供父进程分配 SHM
    proc_func = preprocess_func
    proc_len = 0
    if proc_func is not None:
        try:
            cal_frame = camera.read_depth()
            if cal_frame is not None:
                cal_out = proc_func(cal_frame)
                proc_len = int(np.ascontiguousarray(
                    cal_out, dtype=np.float32).size)
        except Exception as e:
            print(f"[camera subprocess] 预处理标定失败, 仅输出原始帧: {e}")
            proc_func = None
            proc_len = 0

    conn.send((camera.width, camera.height, proc_len))
    conn.recv()  # 等待父进程创建 SHM 完成
    conn.close()

    # 不用 mp_shm.SharedMemory attach: 3.10.12 中 attach 也会向 resource_tracker
    # 注册, 而 tracker 不做引用计数 —— 若父/子进程共享同一 tracker, 子进程任何
    # 注销都会抢先删掉父进程的注册记录, 令父进程 unlink 内部的注销报 KeyError;
    # 若 tracker 各自独立, 子进程退出时其 tracker 又会重复 unlink 报 ENOENT.
    # 改用原始 POSIX shm_open: 生命周期完全由父进程管理, 子进程与 tracker 无关.
    import mmap as _mmap
    import _posixshmem  # 标准库 shared_memory.py 自身使用的私有模块
    fd = _posixshmem.shm_open("/" + shm_name, os.O_RDWR)
    try:
        size = os.fstat(fd).st_size
        mm = _mmap.mmap(fd, size)
    finally:
        os.close(fd)
    shm_buf = memoryview(mm)
    header = MpSharedHeader.from_buffer(shm_buf)
    buffer = np.ndarray((camera.height, camera.width), dtype=np.float32,
                        buffer=shm_buf, offset=SIZE_OF_MP_SHARED_HEADER)
    raw_bytes = camera.width * camera.height * np.dtype(np.float32).itemsize
    proc_buffer = None
    if proc_len > 0:
        proc_buffer = np.ndarray((proc_len,), dtype=np.float32,
                                 buffer=shm_buf,
                                 offset=SIZE_OF_MP_SHARED_HEADER + raw_bytes)

    n_frames, proc_errors, t0 = 0, 0, time.time()
    try:
        while not stop["flag"] and header.writer_termination_signal == 0:
            depth = camera.read_depth()
            if depth is None:
                continue

            # 预处理在写锁外执行, 并计时(写入头部 proc_ms)
            proc_arr, proc_valid = None, 0
            _t0 = time.perf_counter()
            if proc_func is not None:
                try:
                    proc_arr = np.ascontiguousarray(proc_func(depth),
                                                    dtype=np.float32)
                    proc_valid = 1
                except Exception as e:
                    proc_errors += 1
                    if proc_errors % 100 == 1:
                        print(f"[camera subprocess] 预处理异常(第{proc_errors}次): {e}")
                if proc_arr is not None and proc_arr.size != proc_len:
                    proc_arr, proc_valid = None, 0  # 长度突变视为无效, 防越界
            proc_ms = (time.perf_counter() - _t0) * 1000.0

            header.writer_status = 1
            buffer[:] = depth
            if proc_arr is not None and proc_buffer is not None:
                proc_buffer[:] = proc_arr
            header.proc_valid = proc_valid
            header.proc_ms = proc_ms
            header.frame_counter += 1
            header.timestamp = time.time()
            _release_barrier()
            header.writer_status = 0

            n_frames += 1
            if n_frames % WRITER_FREQ_LOG_FRAMES == 0:
                now = time.time()
                print(f"[camera subprocess] {n_frames / (now - t0):.1f} FPS, "
                      f"proc {header.proc_ms:.2f} ms")
                n_frames, t0 = 0, now
    finally:
        camera.stop()  # 必须显式 stop, 否则相机占用残留(见 CLAUDE.md 坑 4)
        # 先删除 numpy/memoryview 导出指针, 否则 mmap 无法关闭
        # (BufferError: cannot close exported pointers exist)
        del buffer, header, shm_buf, proc_buffer
        mm.close()


class OrbbecDepthSpawner:
    """主进程侧: 启动相机子进程, 非阻塞读取最新帧(原始 + 预处理).

    refresh() 在控制循环中调用(有新帧返回 True); get_depth_image() 返回
    最近一帧原始缓存(float32 米); get_processed_image() 返回最近一帧
    预处理结果(启用 preprocess_func 且本帧有效时), 否则 None.
    last_latency_ms 为该帧 子进程写入 -> 本次读取 的延迟,
    last_proc_ms 为子进程侧预处理耗时.
    """

    def __init__(self,
                 width: int = 480,
                 height: int = 270,
                 fps: int = 60,
                 serial: str | None = None,
                 preprocess_func=None):
        self._shm_name = f"orbbec_depth_{os.getpid()}"
        parent_conn, child_conn = mp.Pipe()
        self._proc = mp.Process(target=camera_subprocess_func,
                                args=(child_conn, width, height, fps, serial,
                                      self._shm_name, preprocess_func),
                                daemon=True)
        self._proc.start()
        # 父进程必须立刻关闭自己持有的 child_conn 副本: 否则相机子进程早夭
        # (如 uvc busy)时管道写端仍被父进程持有, recv() 永远等不到 EOF,
        # 握手死锁(fail-fast 失效)
        child_conn.close()
        try:
            # (实际分辨率, 处理帧元素数[0=未启用预处理])
            self.width, self.height, self.proc_len = parent_conn.recv()
        except EOFError:
            raise RuntimeError(
                "相机子进程初始化失败(未检测到设备? 尝试重插 USB 并 "
                "rm -f /dev/shm/orbbec_device_lock)") from None

        raw_bytes = self.width * self.height * np.dtype(np.float32).itemsize
        size = (SIZE_OF_MP_SHARED_HEADER + raw_bytes +
                self.proc_len * np.dtype(np.float32).itemsize)
        self._shm = mp_shm.SharedMemory(name=self._shm_name, create=True,
                                        size=size)
        shm_buf = self._shm.buf
        assert shm_buf is not None
        self._header = MpSharedHeader.from_buffer(shm_buf)
        self._buffer = np.ndarray((self.height, self.width), dtype=np.float32,
                                  buffer=shm_buf,
                                  offset=SIZE_OF_MP_SHARED_HEADER)
        self._proc_buffer = None
        if self.proc_len > 0:
            self._proc_buffer = np.ndarray((self.proc_len,), dtype=np.float32,
                                           buffer=shm_buf,
                                           offset=SIZE_OF_MP_SHARED_HEADER +
                                           raw_bytes)
        self._depth = np.zeros((self.height, self.width), dtype=np.float32)
        self._processed = (np.zeros(self.proc_len, dtype=np.float32)
                           if self.proc_len > 0 else None)
        self._processed_valid = False
        self._last_frame_counter = 0
        self.last_latency_ms: float | None = None
        self.last_proc_ms: float = 0.0
        parent_conn.send(True)
        parent_conn.close()
        print(f"相机子进程已启动 (pid {self._proc.pid}), "
              f"SHM {self._shm_name} {self.width}x{self.height}, "
              f"proc_len {self.proc_len}")

    def refresh(self) -> bool:
        """非阻塞: 有新帧则拷入缓存并返回 True"""
        header = self._header
        if header is None:  # stop() 之后
            return False
        if header.writer_status == 0:
            _acquire_barrier()
            fc = header.frame_counter
            if fc != self._last_frame_counter:
                self._depth[:] = self._buffer
                self._processed_valid = bool(header.proc_valid)
                if (self._processed_valid and self._proc_buffer is not None
                        and self._processed is not None):
                    self._processed[:] = self._proc_buffer
                self.last_proc_ms = header.proc_ms
                self._last_frame_counter = fc
                self.last_latency_ms = (time.time() -
                                        header.timestamp) * 1000.0
                return True
        return False

    def get_depth_image(self) -> np.ndarray:
        return self._depth

    def get_processed_image(self) -> np.ndarray | None:
        """最新预处理帧(扁平 float32); 未启用或本帧无效时返回 None"""
        if self._processed is not None and self._processed_valid:
            return self._processed
        return None

    def stop(self):
        if self._header is not None:
            self._header.writer_termination_signal = 1
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                print("警告: 相机子进程未正常退出, 强制终止")
                self._proc.terminate()
                self._proc.join()
        self._buffer = None
        self._proc_buffer = None  # 同 _buffer: 释放 mmap 导出指针
        self._header = None
        if self._shm is not None:
            self._shm.close()
            # 3.10.12 的 unlink() 内部已调用 resource_tracker.unregister,
            # 不要再手动注销(第二条 UNREGISTER 会让 tracker 崩溃 KeyError).
            self._shm.unlink()
            self._shm = None


def _visualize(depth_m: np.ndarray, fps: float, delay_ms: float | None,
               center_m: float) -> np.ndarray:
    vis = np.clip(depth_m, DISPLAY_DEPTH_MIN_M, DISPLAY_DEPTH_MAX_M)
    vis = ((vis - DISPLAY_DEPTH_MIN_M) /
           (DISPLAY_DEPTH_MAX_M - DISPLAY_DEPTH_MIN_M) *
           255.0).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  # 3通道以便叠加白色文字
    delay_txt = f"  delay {delay_ms:.1f} ms" if delay_ms is not None else ""
    cv2.putText(vis, f"{fps:.1f} FPS{delay_txt}  center {center_m:.2f} m",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis
