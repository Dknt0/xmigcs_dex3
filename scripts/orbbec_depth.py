"""
Orbbec (Gemini 336L) 深度相机 SDK 直采测试

背景: 480x270 深度图经 ROS 话题传输开销过大, 部署端绕过 ROS, 由 SDK 在
同进程内直接取流(与 instinct_onboard 的 RealsenseCamera 相同的单线程
阻塞式 polling 模式)。仿真端仍走 ROS depth 话题, 两者互不影响。

两种测试模式:
  默认      子进程 + 共享内存(移植自 instinct_onboard 的 CameraProcessSpawner),
            主线程非阻塞取帧, 实测帧率与 帧写入->主线程读取 的延迟;
  --single  单进程直采(原模式), 用于对比。

依赖: pyorbbecsdk2 (V2 SDK, import 名仍为 pyorbbecsdk), 详见
https://orbbec.github.io/pyorbbecsdk/index.html

用法:
    python3 orbbec_depth.py                       # 默认 480x270@60, 子进程+SHM
    python3 orbbec_depth.py --single              # 单进程直采
    python3 orbbec_depth.py --serial CPCG853000F4 # 多相机时按序列号选择
    (显示窗口内按 q / ESC 退出)
"""
import argparse
import ctypes
import importlib.util
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
from multiprocessing import resource_tracker
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

# 深度显示范围(米): 线性映射到灰度(近=黑, 远=白), 超出按端点截断
DISPLAY_DEPTH_MIN_M = 0.0
DISPLAY_DEPTH_MAX_M = 5.0


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
    """SHM 块头部: 帧时间戳/写锁/终止信号/帧计数/实际分辨率"""
    _fields_ = [
        ("timestamp", ctypes.c_double),          # 8B: 写入时刻 time.time()
        ("writer_status", ctypes.c_uint32),      # 4B: 0 空闲, 1 写入中
        ("writer_termination_signal", ctypes.c_uint32),  # 4B: 0 存活, 1 应退出
        ("frame_counter", ctypes.c_uint32),      # 4B: 单调递增帧计数
        ("width", ctypes.c_uint32),              # 4B: 实际流宽度
        ("height", ctypes.c_uint32),             # 4B: 实际流高度
        # 显式补齐到 40B: 字段共 28B, 加 12B 填充. 不依赖编译器隐式尾部
        # 对齐(double 使结构体按 8B 对齐, 28B 会被隐式补到 32B, 不受控)
        ("_pad", ctypes.c_uint32 * 3),           # 12B: 28 + 12 = 40B 整
    ]


SIZE_OF_MP_SHARED_HEADER = ctypes.sizeof(MpSharedHeader)
assert SIZE_OF_MP_SHARED_HEADER == 40

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


def camera_subprocess_func(conn, width, height, fps, serial, shm_name):
    """子进程: 持有相机, 阻塞取帧写入 SHM, 直到收到终止信号.

    握手: 先通过 pipe 回传实际流分辨率(父进程据此分配 SHM 大小),
    待父进程回执后再开始写帧。
    """
    stop = {"flag": False}

    def _on_sigterm(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    camera = OrbbecDepthCamera(width, height, fps, serial)
    conn.send((camera.width, camera.height))  # 实际档位可能不同于请求
    conn.recv()  # 等待父进程创建 SHM 完成

    shm = mp_shm.SharedMemory(name=shm_name)
    shm_buf = shm.buf
    assert shm_buf is not None
    header = MpSharedHeader.from_buffer(shm_buf)
    buffer = np.ndarray((camera.height, camera.width), dtype=np.float32,
                        buffer=shm_buf, offset=SIZE_OF_MP_SHARED_HEADER)
    conn.close()

    n_frames, t0 = 0, time.time()
    try:
        while not stop["flag"] and header.writer_termination_signal == 0:
            depth = camera.read_depth()
            if depth is None:
                continue
            header.writer_status = 1
            buffer[:] = depth
            header.frame_counter += 1
            header.timestamp = time.time()
            _release_barrier()
            header.writer_status = 0

            n_frames += 1
            if n_frames % WRITER_FREQ_LOG_FRAMES == 0:
                now = time.time()
                print(f"[camera subprocess] {n_frames / (now - t0):.1f} FPS")
                n_frames, t0 = 0, now
    finally:
        camera.stop()  # 必须显式 stop, 否则相机占用残留(见 CLAUDE.md 坑 4)
        del buffer, header, shm_buf
        shm.close()


class OrbbecDepthSpawner:
    """主进程侧: 启动相机子进程, 非阻塞读取最新帧.

    refresh() 在控制循环中调用(有新帧返回 True), get_depth_image()
    返回最近一帧缓存(float32 米), last_latency_ms 为该帧
    子进程写入 -> 本次读取 的延迟。
    """

    def __init__(self,
                 width: int = 480,
                 height: int = 270,
                 fps: int = 60,
                 serial: str | None = None):
        self._shm_name = f"orbbec_depth_{os.getpid()}"
        parent_conn, child_conn = mp.Pipe()
        self._proc = mp.Process(target=camera_subprocess_func,
                                args=(child_conn, width, height, fps, serial,
                                      self._shm_name),
                                daemon=True)
        self._proc.start()
        try:
            self.width, self.height = parent_conn.recv()  # 实际分辨率
        except EOFError:
            raise RuntimeError(
                "相机子进程初始化失败(未检测到设备? 尝试重插 USB 并 "
                "rm -f /dev/shm/orbbec_device_lock)") from None

        size = SIZE_OF_MP_SHARED_HEADER + self.width * self.height * np.dtype(
            np.float32).itemsize
        self._shm = mp_shm.SharedMemory(name=self._shm_name, create=True,
                                        size=size)
        shm_buf = self._shm.buf
        assert shm_buf is not None
        self._header = MpSharedHeader.from_buffer(shm_buf)
        self._buffer = np.ndarray((self.height, self.width), dtype=np.float32,
                                  buffer=shm_buf,
                                  offset=SIZE_OF_MP_SHARED_HEADER)
        self._depth = np.zeros((self.height, self.width), dtype=np.float32)
        self._last_frame_counter = 0
        self.last_latency_ms: float | None = None
        parent_conn.send(True)
        parent_conn.close()
        print(f"相机子进程已启动 (pid {self._proc.pid}), "
              f"SHM {self._shm_name} {self.width}x{self.height}")

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
                self._last_frame_counter = fc
                self.last_latency_ms = (time.time() -
                                        header.timestamp) * 1000.0
                return True
        return False

    def get_depth_image(self) -> np.ndarray:
        return self._depth

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
        self._header = None
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            resource_tracker.unregister(self._shm._name, "shared_memory")
            self._shm = None


# ======================================================================
# 测试主循环
# ======================================================================

def _visualize(depth_m: np.ndarray, fps: float, delay_ms: float | None,
               center_m: float) -> np.ndarray:
    vis = np.clip(depth_m, DISPLAY_DEPTH_MIN_M, DISPLAY_DEPTH_MAX_M)
    vis = ((vis - DISPLAY_DEPTH_MIN_M) /
           (DISPLAY_DEPTH_MAX_M - DISPLAY_DEPTH_MIN_M) *
           255.0).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)  # 3通道以便叠加白色文字
    delay_txt = f"  delay {delay_ms:.1f} ms" if delay_ms is not None else ""
    cv2.putText(vis, f"{fps:.1f} FPS{delay_txt}  center {center_m:.2f} m",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def run_mp_test(args):
    """子进程+SHM 模式: 主线程测帧率与延迟"""
    spawner = OrbbecDepthSpawner(args.width, args.height, args.fps, args.serial)
    win = "orbbec depth (subprocess + shm)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fps_cnt, fps_t0, fps_val, delay_ms = 0, time.perf_counter(), 0.0, None

    try:
        while True:
            if spawner.refresh():
                depth_m = spawner.get_depth_image()
                delay_ms = spawner.last_latency_ms

                fps_cnt += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    fps_val = fps_cnt / (now - fps_t0)
                    fps_cnt, fps_t0 = 0, now
                    center_m = float(depth_m[spawner.height // 2,
                                             spawner.width // 2])
                    print(f"depth {fps_val:.1f} FPS, "
                          f"delay {delay_ms:.2f} ms, center {center_m:.3f} m")

                center_m = float(depth_m[spawner.height // 2,
                                         spawner.width // 2])
                cv2.imshow(win,
                           _visualize(depth_m, fps_val, delay_ms, center_m))
            if cv2.waitKey(1) in (ord('q'), 27):
                break
            time.sleep(0.001)  # 避免空转烧 CPU
    except KeyboardInterrupt:
        pass
    finally:
        spawner.stop()
        cv2.destroyAllWindows()


def run_single_test(args):
    """单进程直采模式(对照)"""
    camera = OrbbecDepthCamera(args.width, args.height, args.fps, args.serial)
    win = "orbbec depth (single process)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fps_cnt, fps_t0, fps_val = 0, time.perf_counter(), 0.0

    try:
        while True:
            depth_m = camera.read_depth()
            if depth_m is None:
                continue
            fps_cnt += 1
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                fps_val = fps_cnt / (now - fps_t0)
                fps_cnt, fps_t0 = 0, now
                print(f"depth {fps_val:.1f} FPS")
            center_m = float(depth_m[camera.height // 2, camera.width // 2])
            cv2.imshow(win, _visualize(depth_m, fps_val, None, center_m))
            if cv2.waitKey(1) in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def main():
    # SIGTERM(如 timeout 命令)默认直接终止进程, 不执行 finally 中的
    # pipeline.stop(), 相机会保持占用(uvc busy)直到重新上电. 转成
    # SystemExit 让清理逻辑生效.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    parser = argparse.ArgumentParser(description="Orbbec 深度流测试")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--single", action="store_true", default=False,
                        help="单进程直采模式(默认为子进程+SHM)")
    args = parser.parse_args()

    if args.single:
        run_single_test(args)
    else:
        run_mp_test(args)


if __name__ == "__main__":
    main()
