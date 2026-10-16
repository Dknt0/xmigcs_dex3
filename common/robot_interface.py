"""
Robot Interface
Python equivalent of the C++ RobotInterface class
"""
from __future__ import annotations
import copy
import queue
from common.peekqueue import PeekableQueue
import yaml
import os
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
# ROS messages
from bodyctrl_msgs.msg import MotorStatusMsg, CmdMotorCtrl, MotorCtrl, Imu, MotorStatus
import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import transforms3d as t3d
from .body_id_map import BodyServoIdMap
from .robot_data import RobotData
import functools
import time
import math
from sptlib_python import funcSPTrans as FuncSPTrans

from common.joystick import ControlFlag
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from FSM.fsm_base import FSMStateName
# from common import orbbec_camera  # 相机驱动由外部 ROS 节点负责
import cv2 as cv

def timing_decorator(func):
    """
    装饰器：记录函数执行时间
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"[TIMING] {func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper


class RobotInterface(ABC):
    """机器人接口抽象基类"""

    # 由 load_config 从 dex_config.yaml 读取; 仿真节点经 XMIGCSControlNode
    # 构造参数在控制线程启动前覆盖(见 rl_control_node.py)
    sim: bool

    def __init__(self, robot_data: RobotData):
        self.robot_data_ = robot_data

    @abstractmethod
    def init(self, node: Node):
        """初始化接口"""
        pass

    @abstractmethod
    def update_robot_data(self, flag:ControlFlag, time_passed: float):
        """更新机器人状态"""
        pass


    @abstractmethod
    def send_motor_commands(self, flag: ControlFlag):
        """发布电机控制命令"""
        pass

    # 深度采集生命周期钩子(默认空操作, 由 RobotInterfaceImpl 按需覆盖).
    # 声明在基类使控制节点可统一调用, 且 Pyright 不报未知属性.
    def refresh_depth(self):
        """控制线程每拍调用: 深度刷新兼容钩子(ROS 由回调更新)"""

    def finish_control(self):
        """控制线程退出前调用: 清理该线程内创建的资源"""

    def pump_debug_vis(self):
        """主线程(spin)定时器回调: 显示调试帧(cv 窗口须在主线程)"""

    def close_debug_window(self):
        """主线程销毁调试窗口"""

    def shutdown(self):
        """节点销毁时调用: 释放接口资源"""



class RobotInterfaceImpl(RobotInterface):
    """机器人接口具体实现"""

    def __init__(self, robot_data: RobotData, config_path: str = ''):
        super().__init__(robot_data)
        self.initialized = False
        self.node = None
        self.config_path = config_path

        # ID映射
        self.id_map = BodyServoIdMap()
        self.id_map.body_can_id_map_init()

        # 消息队列
        self.queue_leg_motor_state = PeekableQueue(maxsize=1)
        self.queue_arm_motor_state = PeekableQueue(maxsize=1)
        self.queue_waist_motor_state = PeekableQueue(maxsize=1)
        self.queue_imu_xsens = PeekableQueue(maxsize=1)
        self.queue_walk_cmd = PeekableQueue(maxsize=1)


        # 关节维度
        self.floating_base_dof = 6
        self.whole_joint_nums = self.id_map.whole_motor_nums + self.floating_base_dof

        # 临时变量用于优化计算
        self.temp_q = np.empty(self.id_map.whole_motor_nums)
        # 预分配另一个用于存储中间计算的临时数组
        self._temp_zero_cnt = np.empty(self.id_map.whole_motor_nums)

        # 电机控制参数
        self.motor_dir = np.ones(self.id_map.whole_motor_nums)
        self.zero_cnt = np.zeros(self.id_map.whole_motor_nums)
        self.zero_offset = np.zeros(self.id_map.whole_motor_nums)

        # 添加标志位，用于跟踪是否是首次接收数据
        self.first_leg_data_received = False
        self.first_arm_data_received = False
        self.first_waist_data_received = False

        # 关节限位
        self.joint_limits = {}
        self.joint_pos_threshold = math.pi

        # 串并联转换器
        self.fun_s2p = FuncSPTrans()

        # 串并联转换相关变量
        self.left_ankle_indices = np.array([4, 5]) + self.floating_base_dof
        self.right_ankle_indices = np.array([10, 11]) + self.floating_base_dof
        self.ankle_kp_p = np.zeros(4)  # 并联关节刚度
        self.ankle_kd_p = np.zeros(4)  # 并联关节阻尼

        # TF相关属性
        self.tf_buffer = None
        self.tf_listener = None
        
        # ROS publishers and subscribers
        self.pub_leg_motor_cmd = None
        self.pub_arm_motor_cmd = None
        self.pub_waist_motor_cmd = None
        self.sub_leg_state = None
        self.sub_arm_state = None
        self.sub_waist_state = None

        # 深度直采(orbbec)状态: 其余模式下保持 None
        # self.orbbec_spawner = None
        # self._depth_fps_t0 = None      # 首帧时刻起算 FPS, 避免 0.0 FPS 假值
        self._depth_vis_frame = None   # 最新调试帧(控制线程构建, 主线程显示)
        self._depth_win_open = False   # cv 窗口须在主线程创建/销毁
        self._policy_win_open = False  # WALKDEPTH 处理帧窗口, 同上

        # 当前机器人所处状态
        self.current_state: FSMStateName = FSMStateName.STOP
        # Velocity commands are only meaningful in walking policy states.  All
        # prep/stop/zero states force RobotData.walk_cmd_ back to zero so stale
        # controller input cannot leak into a later policy tick.
        self.walk_command_status = {
            FSMStateName.WALKAMP,
            FSMStateName.WALKLION,
            FSMStateName.WALKLIONDWAQ,
            FSMStateName.WALKDEPTH,
            FSMStateName.WALKDEPTH100HZ,
            FSMStateName.MULTIAMP,
        }

    def update_param(self, current_state: FSMStateName = None):
        """更新机器人接口"""
        if current_state is not None:
            self.current_state = current_state

    def load_config(self):
        """从配置文件加载关键参数"""
        config_path = self.config_path
        if not os.path.exists(config_path):
            self.node.get_logger().error(
                f"Joint limits config file not found: {config_path}")
            raise FileNotFoundError(
                f"Joint limits config file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.node.get_logger().error(
                f"Failed to load joint limits config from {config_path}: {e}")
            raise RuntimeError(
                f"Failed to load joint limits config from {config_path}: {e}")

        # 读取运行模式
        self.sim = config.get('sim')
        self.debug = config.get('debug')
        # 机器人接口配置
        interface_config = config.get('robot_interface')
        # 是否限位
        self.clip_actions = interface_config.get('clip_actions')
        # 加载关节限位值
        self._load_joint_limits(interface_config)
        # 加载控制状态
        self._load_control_status(interface_config)
        # 零位
        self.zero_pos = np.array(interface_config.get('zero_pos'))
        # 电流转换比例
        self.ct_scale = np.array(interface_config.get('ct_scale'))
        # IMU 数据偏移
        self.xsense_roll_offset = interface_config.get(
            'xsense_data_roll_offset')
        # 禁用电机
        self.disable_joints_ = interface_config.get('disable_joints', False)
        # 脚踝Kp,Kd
        self.ankle_kp_p = np.array(interface_config.get('ankle_kp_p'))
        self.ankle_kd_p = np.array(interface_config.get('ankle_kd_p'))

        self.depth_cfg = interface_config.get('depth', {})
        self.depth_type = self.depth_cfg.get('type', 'disable')
        self.camera_debug_vis = self.depth_cfg.get('debug_vis', False)

        self.depth_w_ = int(self.depth_cfg['width'])
        self.depth_h_ = int(self.depth_cfg['height'])
        self.depth_min_ = float(self.depth_cfg.get('min_depth', 0.0))
        self.depth_max_ = float(self.depth_cfg.get('max_depth', 2.5))
        self.depth_noise_ = float(self.depth_cfg.get('noise_stddev', 0.0))
        self.depth_normalize_ = bool(self.depth_cfg.get('normalize', True))
        self.depth_crop_l_ = int(self.depth_cfg.get('crop_left', 0))
        self.depth_crop_r_ = int(self.depth_cfg.get('crop_right', 0))
        self.depth_crop_t_ = int(self.depth_cfg.get('crop_top', 0))
        self.depth_crop_b_ = int(self.depth_cfg.get('crop_bottom', 0))
        self.depth_blur_ksize_ = int(self.depth_cfg.get('blur_kernel', 3))
        self.depth_blur_sigma_ = float(self.depth_cfg.get('blur_sigma', 1.0))
        # 无效像素修复阈值(米): Orbbec 336L 的无效点输出 0.0m, 若不处理会被
        # 归一化成"最近障碍". 低于该阈值的像素按 instinct_onboard 方案 inpaint
        # (邻域填充). 仿真中 MuJoCo 空像素为远距离(zfar), 掩码恒为空 → 本步骤
        # 零开销, 不影响训练分布. 设为 0 禁用.
        self.depth_inpaint_near_ = float(self.depth_cfg.get('inpaint_near_threshold', 0.2))
        self.depth_inpaint_far_ = float(self.depth_cfg.get('inpaint_far_threshold', 40.0))
        # 送入策略的帧尺寸 = 相机分辨率裁剪后 (ovinf obs_width_/obs_height_)
        self.depth_crop_w_ = self.depth_w_ - self.depth_crop_l_ - self.depth_crop_r_
        self.depth_crop_h_ = self.depth_h_ - self.depth_crop_t_ - self.depth_crop_b_

    def _load_control_status(self, config: Dict[str, Any]):
        # 字符串命令到枚举值的映射
        state_to_FSMState = {
            "STOP": FSMStateName.STOP,
            "ZERO": FSMStateName.ZERO,
            "WALKAMP": FSMStateName.WALKAMP,
            "BEYONDMIMIC": FSMStateName.BEYONDMIMIC,
            "BEYONDZERO": FSMStateName.BEYONDZERO,
            "WALKLION": FSMStateName.WALKLION,
            "WALKLIONDWAQ": FSMStateName.WALKLIONDWAQ,
            "LIONZERO": FSMStateName.LIONZERO,
            "POLICYZERO": FSMStateName.POLICYZERO,
            "WALKDEPTH": FSMStateName.WALKDEPTH,
            "STAND": FSMStateName.STAND,
            "WALKDEPTH100HZ": FSMStateName.WALKDEPTH100HZ,
            "MULTIAMP": FSMStateName.MULTIAMP,
        }
        self.waist_control_status = [state_to_FSMState[state] for state in config.get('waist_control_status')]
        self.legs_control_status = [state_to_FSMState[state] for state in config.get('legs_control_status')]
        self.arms_control_status = [state_to_FSMState[state] for state in config.get('arms_control_status')]
        self.left_arm_only_status = [state_to_FSMState[state] for state in config.get('left_arm_only_status')]
        self.right_arm_only_status = [state_to_FSMState[state] for state in config.get('right_arm_only_status')]

    def _load_joint_limits(self, config: Dict[str, Any]):
        """从配置文件加载关节限位值"""
        # 从配置中获取关节限位信息
        joint_limits_config = config.get('joint_limits', None)

        if joint_limits_config is None:
            error_msg = "No joint_limits section in config"
            self.node.get_logger().error(error_msg)
            raise ValueError(error_msg)
        else:
            # 从配置中加载限位值
            for joint_name, limits in joint_limits_config.items():
                if 'min' in limits and 'max' in limits:
                    self.joint_limits[joint_name] = {
                        'min': float(limits['min']),
                        'max': float(limits['max'])
                    }
                else:
                    error_msg = f"Invalid limits for joint {joint_name}"
                    self.node.get_logger().error(error_msg)
                    raise ValueError(error_msg)

            # self.node.get_logger().info(f"Loaded joint limits from {config}")
        # 预计算ID到限位的映射
        self.id_to_limits = {}
        for joint_name, limits in self.joint_limits.items():
            index = self.id_map.get_index_by_name(joint_name)
            if index >= 0:
                motor_id = self.id_map.get_id_by_index(index)
                self.id_to_limits[motor_id] = limits

        # 记录加载的限位值
        for joint_name, limits in self.joint_limits.items():
            self.node.get_logger().debug(
                f"Joint {joint_name}: [{limits['min']}, {limits['max']}]")
        # print("-" * 30 + '关节限位值' + '-' * 30)
        # print(self.joint_limits)

    def init(self, node: Node):
        """初始化接口"""
        self.node = node
        self.initialized = True

        # 初始化ROS接口
        self._init_ros_interfaces()
        # 加载配置文件
        self.load_config()
        self._init_depth_camera()
        
        node.get_logger().info("Robot interface initialized")

    def _init_ros_interfaces(self):
        """初始化ROS接口"""
        qos_profile = QoSProfile(
            # reliability=ReliabilityPolicy.RELIABLE,
            # history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # 发布者
        self.pub_leg_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/leg/cmd_ctrl', qos_profile)
        self.pub_arm_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/arm/cmd_ctrl', qos_profile)
        self.pub_waist_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/waist/cmd_ctrl', qos_profile)

        # 订阅者
        self.sub_leg_state = self.node.create_subscription(
            MotorStatusMsg, '/leg/status', self._leg_motor_status_callback,
            qos_profile)
        self.sub_arm_state = self.node.create_subscription(
            MotorStatusMsg, '/arm/status', self._arm_motor_status_callback,
            qos_profile)
        self.sub_waist_state = self.node.create_subscription(
            MotorStatusMsg, '/waist/status', self._waist_motor_status_callback,
            qos_profile)

        #（非电机相关）
        self.sub_imu_xsens = self.node.create_subscription(
            Imu, '/imu/status', self._imu_status_callback, qos_profile)

    def _init_depth_camera(self):
        # 深度相机（传感器数据用 BEST_EFFORT, 只保留最新一帧）
        if self.depth_type == "disable":
            self.node.get_logger().warn("Depth camera is disabled.")
            # TODO: Add nominal depth image in RobotData
        elif self.depth_type == "ros2":
            self.node.get_logger().info("Subscribing depth camera ros topic.")
            sensor_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                    reliability=ReliabilityPolicy.BEST_EFFORT)
            depth_topic = self.depth_cfg.get("ros2", {}).get("topic", "/camera/depth/image_raw")
            self.sub_depth_image = self.node.create_subscription(
                Image, depth_topic, self._depth_image_callback,
                sensor_qos)
        # elif self.depth_type == "orbbec":
        #     self.node.get_logger().info("Using Orbbec depth camera.")
        #     orbbec_cfg = self.depth_cfg.get("orbbec", {})
        #     # yaml 中 serial_number 为空/None 字符串时表示使用第一台设备
        #     serial = orbbec_cfg.get("serial_number", None)
        #     serial = None if serial in (None, "", "None") else str(serial)
        #     self.orbbec_spawner = orbbec_camera.OrbbecDepthSpawner(
        #         orbbec_cfg.get("cam_width", 480),
        #         orbbec_cfg.get("cam_height", 270),
        #         orbbec_cfg.get("fps", 60),
        #         serial,
        #         # 预处理在相机子进程内执行(写锁外), 产出随 SHM 传输;
        #         # 绑定方法依赖 fork 启动方式(spawn 下不可 pickle, 见 CLAUDE.md)
        #         self._preprocess_depth,
        #     )

        else:
            raise ValueError(f"Unknown depth_type: {self.depth_type}")

    def _preprocess_depth(self, img) -> np.ndarray:
        """训练侧 Instinct510 dex23dof_depth_cfg.py noise_pipeline 的移植。

        顺序: 缩放到相机分辨率 -> 加性高斯噪声(米制) -> crop ->
        inpaint无效像素(< threshold, 仿真中为空操作) -> 高斯模糊(reflect)
        -> 裁剪[min,max] -> 归一化, 归一化在最后。

        自带耗时统计: EMA 写入 robot_data_.depth_processing_time_(秒)。
        """
        _t0 = time.perf_counter()
        processed = np.asarray(img, dtype=np.float32)

        # Resize
        if processed.shape != (self.depth_h_, self.depth_w_):
            processed = cv.resize(processed, (processed.shape[1] // 2, processed.shape[0] // 2),
                             interpolation=cv.INTER_NEAREST)

        # Inpaint
        # NaN/Inf、336L 无效像素(0.0m)与超出阈值的深度统一做邻域填充。
        # NaN 不满足远近阈值比较，必须显式标记，避免后续模糊扩散 NaN。
        invalid_mask = (
            ~np.isfinite(processed)
            | (processed < self.depth_inpaint_near_)
            | (processed > self.depth_inpaint_far_)
        ).astype(np.uint8)
        if invalid_mask.any():
            processed = cv.inpaint(processed, invalid_mask, 3,
                                       cv.INPAINT_NS)
        # Resize
        if processed.shape != (self.depth_h_, self.depth_w_):
            processed = cv.resize(processed, (self.depth_w_, self.depth_h_),
                             interpolation=cv.INTER_NEAREST)

        # Noise
        if self.depth_noise_ > 0.0:
            processed += np.random.normal(0.0, self.depth_noise_, processed.shape).astype(np.float32)

        # Crop
        h = self.depth_crop_h_
        w = self.depth_crop_w_
        processed = processed[self.depth_crop_t_:self.depth_crop_t_ + h,
                        self.depth_crop_l_:self.depth_crop_l_ + w]


        # Blur
        if self.depth_blur_ksize_ > 0:
            # torchvision GaussianBlur 默认 'reflect' 填充 == REFLECT_101
            processed = cv.GaussianBlur(
                processed, (self.depth_blur_ksize_, self.depth_blur_ksize_),
                self.depth_blur_sigma_, borderType=cv.BORDER_REFLECT_101)

        # Clip
        processed = np.clip(processed, self.depth_min_, self.depth_max_)

        # Normalize
        if self.depth_normalize_:
            processed = (processed - self.depth_min_) / (self.depth_max_ - self.depth_min_)

        # Processtime
        _proc_s = time.perf_counter() - _t0
        _prev = self.robot_data_.depth_processing_time_
        self.robot_data_.depth_processing_time_ = (
            _prev + 0.1 * (_proc_s - _prev) if _prev > 0 else _proc_s)

        print(f"Depth preprocessing time: {_proc_s:.6f} seconds")

        return processed.reshape(-1)

    def _preprocess_depth_old(self, img) -> np.ndarray:
        """训练侧 Instinct510 dex23dof_depth_cfg.py noise_pipeline 的移植。

        顺序: 缩放到相机分辨率 -> 加性高斯噪声(米制) -> crop ->
        inpaint无效像素(< threshold, 仿真中为空操作) -> 高斯模糊(reflect)
        -> 裁剪[min,max] -> 归一化, 归一化在最后。

        自带耗时统计: EMA 写入 robot_data_.depth_processing_time_(秒)。
        """
        _t0 = time.perf_counter()
        img = np.asarray(img, dtype=np.float32)
        if img.shape != (self.depth_h_, self.depth_w_):
            # img = cv.resize(img, (self.depth_w_, self.depth_h_),
            #                  interpolation=cv.INTER_AREA)
            img = cv.resize(img, (self.depth_w_, self.depth_h_),
                             interpolation=cv.INTER_NEAREST)
        else:
            img = img.copy()
        if self.depth_noise_ > 0.0:
            img += np.random.normal(0.0, self.depth_noise_, img.shape).astype(np.float32)
        h = self.depth_crop_h_
        w = self.depth_crop_w_
        processed = img[self.depth_crop_t_:self.depth_crop_t_ + h,
                        self.depth_crop_l_:self.depth_crop_l_ + w]
        if self.depth_inpaint_ > 0.0:
            # 336L 无效像素(0.0m)与近场噪声 → 邻域填充, 避免被读成最近障碍
            invalid_mask = (processed < self.depth_inpaint_).astype(np.uint8)
            if invalid_mask.any():
                processed = cv.inpaint(processed, invalid_mask, 3,
                                        cv.INPAINT_NS)
        if self.depth_blur_ksize_ > 0:
            # torchvision GaussianBlur 默认 'reflect' 填充 == REFLECT_101
            processed = cv.GaussianBlur(
                processed, (self.depth_blur_ksize_, self.depth_blur_ksize_),
                self.depth_blur_sigma_, borderType=cv.BORDER_REFLECT_101)
        processed = np.clip(processed, self.depth_min_, self.depth_max_)
        if self.depth_normalize_:
            processed = (processed - self.depth_min_) / (self.depth_max_ - self.depth_min_)
        _proc_s = time.perf_counter() - _t0
        _prev = self.robot_data_.depth_processing_time_
        self.robot_data_.depth_processing_time_ = (
            _prev + 0.1 * (_proc_s - _prev) if _prev > 0 else _proc_s)
        return processed.reshape(-1)

    def _depth_image_callback(self, msg: Image):
        """缓存最新深度帧(32FC1, 米), 供 WALKDEPTH 策略读取"""
        try:
            if msg.encoding == "16UC1":
                img = (
                    np.frombuffer(msg.data, dtype=np.uint16)
                    .reshape(msg.height, msg.width)
                    .astype(np.float32) * 0.001
                )
            else:  # 32FC1，仿真或已转换过的源
                img = np.frombuffer(msg.data, dtype=np.float32).reshape(
                    msg.height, msg.width)
            # img = np.frombuffer(msg.data, dtype=np.float32).reshape(
                # msg.height, msg.width)
            self.robot_data_.depth_image_ = img
            self.robot_data_.depth_image_stamp_ = msg.header.stamp.sec + \
                msg.header.stamp.nanosec * 1e-9

            self.robot_data_.processed_depth_image_ = self._preprocess_depth(img)

        except Exception as e:
            print(f"depth image callback error: {e}")


    # @timing_decorator
    def get_imu_data(self):
        """处理传感器数据（非电机）"""
        # 处理IMU数据
        while True:
            try:
                msg = self.queue_imu_xsens.peek()
                self.robot_data_.imu_data_[0] = msg.euler.yaw
                self.robot_data_.imu_data_[1] = msg.euler.pitch
                self.robot_data_.imu_data_[2] = msg.euler.roll
                self.robot_data_.imu_data_[3] = msg.angular_velocity.x
                self.robot_data_.imu_data_[4] = msg.angular_velocity.y
                self.robot_data_.imu_data_[5] = msg.angular_velocity.z
                self.robot_data_.imu_data_[6] = msg.linear_acceleration.x
                self.robot_data_.imu_data_[7] = msg.linear_acceleration.y
                self.robot_data_.imu_data_[8] = msg.linear_acceleration.z
                break
            except queue.Empty:
                time.sleep(0.0001)

    def update_robot_state(self):
        """读取电机状态数据更新为机器人状态数据"""
        # 处理腿部电机状态
        while True:
            try:
                msg = self.queue_leg_motor_state.peek()
                if self.debug:
                    current_time = self.node.get_clock().now().to_msg()
                    msg_time = msg.header.stamp
                    time_diff = (current_time.sec -
                                 msg_time.sec) * 1000000000 + (
                                     current_time.nanosec - msg_time.nanosec)
                    time_diff_ms = time_diff / 1000000.0
                    print(f"Time difference: {time_diff_ms} ms")
                for status in msg.status:
                    self.motor_state_to_robot_state(
                        status, self.first_leg_data_received)
                break
            except queue.Empty:
                time.sleep(0.0001)

        # 处理手臂电机状态
        while True:
            try:
                msg = self.queue_arm_motor_state.peek()
                if self.debug:
                    current_time = self.node.get_clock().now().to_msg()
                    msg_time = msg.header.stamp
                    time_diff = (current_time.sec -
                                 msg_time.sec) * 1000000000 + (
                                     current_time.nanosec - msg_time.nanosec)
                    time_diff_ms = time_diff / 1000000.0
                    print(f"Time difference: {time_diff_ms} ms")
                for status in msg.status:
                    self.motor_state_to_robot_state(
                        status, self.first_arm_data_received)

                break
            except queue.Empty:
                time.sleep(0.0001)

        # 处理腰部电机状态
        while True:
            try:
                msg = self.queue_waist_motor_state.peek()
                if self.debug:
                    current_time = self.node.get_clock().now().to_msg()
                    msg_time = msg.header.stamp
                    time_diff = (current_time.sec -
                                 msg_time.sec) * 1000000000 + (
                                     current_time.nanosec - msg_time.nanosec)
                    time_diff_ms = time_diff / 1000000.0
                    print(f"Time difference: {time_diff_ms} ms")
                for status in msg.status:
                    self.motor_state_to_robot_state(
                        status, self.first_waist_data_received)
                break
            except queue.Empty:
                time.sleep(0.0001)

    def motor_state_to_robot_state(self, status, received_flag: bool):
        index = self.id_map.get_index_by_id(status.name)
        if index >= 0:
            robotdata_index = index + self.floating_base_dof  # 偏移到完整关节数组中
            # 直接赋值
            self.robot_data_.q_a_p_[robotdata_index] = status.pos
            self.robot_data_.q_dot_a_p_[robotdata_index] = status.speed
            self.robot_data_.tau_a_p_[
                robotdata_index] = status.current * self.ct_scale[min(
                    index,
                    len(self.ct_scale) - 1)]
            self.robot_data_.temperature_a[
                robotdata_index - self.floating_base_dof] = status.temperature

            self.robot_data_.q_a_p_[robotdata_index] = (
                status.pos - self.zero_pos[index]
            ) * self.motor_dir[index] + self.zero_offset[index]
            # if self.robot_data_.q_a_p_[robotdata_index] > math.pi:
            #     self.zero_cnt[index] = -1.0
            # elif self.robot_data_.q_a_p_[robotdata_index] < -math.pi:
            #     self.zero_cnt[index] = 1.0

            self.robot_data_.q_a_p_[
                robotdata_index] += self.zero_cnt[index] * 2.0 * math.pi
            self.robot_data_.q_dot_a_p_[robotdata_index] *= self.motor_dir[index]
            self.robot_data_.tau_a_p_[robotdata_index] *= self.motor_dir[index]

            # The received_flag is always be False, this code is not used
            if not received_flag or abs(
                    self.robot_data_.q_a_p_[robotdata_index] -
                    self.robot_data_.q_a_last[robotdata_index]
            ) > self.joint_pos_threshold:
                if received_flag:
                    self.node.get_logger().warn(
                        f"Joint {index} error detected")
                    self.robot_data_.q_a_p_[
                        robotdata_index] = self.robot_data_.q_a_last[
                            robotdata_index]
                    self.robot_data_.q_dot_a_p_[
                        robotdata_index] = self.robot_data_.qdot_a_last[
                            robotdata_index]
                    self.robot_data_.tau_a_p_[
                        robotdata_index] = self.robot_data_.tor_a_last[
                            robotdata_index]
                else:
                    # 首次接收数据，更新标志位
                    received_flag = True
            self.robot_data_.q_a_last[robotdata_index] = self.robot_data_.q_a_p_[
                robotdata_index]
            self.robot_data_.qdot_a_last[
                robotdata_index] = self.robot_data_.q_dot_a_p_[robotdata_index]
            self.robot_data_.tor_a_last[
                robotdata_index] = self.robot_data_.tau_a_p_[robotdata_index]

    def refresh_depth(self):
        """保留控制循环接口；ROS 深度帧由订阅回调更新。"""
        return

        # if self.depth_type != "orbbec" or self.orbbec_spawner is None:
        #     return
        # if not self.orbbec_spawner.refresh():
        #     return
        # depth_m = self.orbbec_spawner.get_depth_image()
        # # 独立数组(不别名 spawner 内部缓冲), 语义与 ROS 回调路径一致
        # self.robot_data_.depth_image_ = depth_m.copy()
        # self.robot_data_.depth_image_stamp_ = time.time()
        # # 预处理帧由相机子进程产出; 未启用/本帧无效时置 None(策略回退中性填充)
        # processed = self.orbbec_spawner.get_processed_image()
        # self.robot_data_.processed_depth_image_ = (
        #     processed.copy() if processed is not None else None)
        # if self.orbbec_spawner.last_proc_ms > 0:
        #     self.robot_data_.depth_processing_time_ = \
        #         self.orbbec_spawner.last_proc_ms / 1000.0

        # # 帧率/延迟统计(1Hz); t0 从首帧起算, 避免初始 0.0 FPS 假值
        # self.robot_data_.depth_counter_ += 1
        # now = time.perf_counter()
        # if self._depth_fps_t0 is None:
        #     self._depth_fps_t0 = now
        #     self.robot_data_.depth_counter_ = 0
        #     return
        # delay_ms = self.orbbec_spawner.last_latency_ms
        # center_m = float(depth_m[self.orbbec_spawner.height // 2,
        #                  self.orbbec_spawner.width // 2])
        # fps = self.robot_data_.depth_counter_ / (now - self._depth_fps_t0)
        # if now - self._depth_fps_t0 >= 1.0:
        #     self.robot_data_.fps_val = fps
        #     print(f"depth {fps:.1f} FPS, delay {delay_ms:.2f} ms, "
        #           f"proc {self.robot_data_.depth_processing_time_ * 1000:.2f} ms, "
        #           f"center {center_m:.3f} m")
        #     self.robot_data_.depth_counter_, self._depth_fps_t0 = 0, now

        # # 调试可视化(1Hz): 控制线程只构建图像(纯绘制, 无 GUI 调用);
        # # imshow/waitKey 由主线程(spin)的 1Hz 定时器执行, 保证 cv 窗口的
        # # 创建/刷新/销毁全部在同一线程 —— 否则 Qt 报 QObject::killTimer
        # # 跨线程告警, 且 GUI 卡顿会直接拖慢控制循环
        # if self.camera_debug_vis:
        #     vis = orbbec_camera._visualize(depth_m, fps, delay_ms, center_m)
        #     # 红色细框标出送入策略的裁剪区域: 策略先把相机帧缩放到
        #     # depth.width x depth.height(64x36), 再裁 crop_top/bottom/left/right
        #     # (见 fsm_walkdepth._preprocess_depth); 此处把该窗口等比映射回
        #     # 相机分辨率画框, 直观显示策略实际"看到"的范围
        #     dw = int(self.depth_cfg.get('width', 64))
        #     dh = int(self.depth_cfg.get('height', 36))
        #     cw = self.orbbec_spawner.width
        #     ch = self.orbbec_spawner.height
        #     x1 = int(round(self.depth_cfg.get('crop_left', 0) * cw / dw))
        #     x2 = int(round((dw - self.depth_cfg.get('crop_right', 0)) * cw / dw))
        #     y1 = int(round(self.depth_cfg.get('crop_top', 0) * ch / dh))
        #     y2 = int(round((dh - self.depth_cfg.get('crop_bottom', 0)) * ch / dh))
        #     # clamp 到帧内: crop_bottom/crop_right 为 0 时 y2/x2 恰好等于
        #     # 帧高/宽, 该边线画在图像外永远不可见
        #     x2 = min(x2, cw - 1)
        #     y2 = min(y2, ch - 1)
        #     cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #     self._depth_vis_frame = vis

    def pump_debug_vis(self):
        """主线程(spin)定时器回调: 显示最新调试帧. cv/Qt 窗口必须在
        主线程创建与刷新(QApplication 归属线程). 相机帧与策略处理帧
        (如 WALKDEPTH)分属两个窗口, 均由本方法驱动. 两窗口均为
        WINDOW_NORMAL: 可鼠标拖拽缩放, 并保持纵横比."""
        frame = self._depth_vis_frame
        if frame is not None:
            if not self._depth_win_open:
                # WINDOW_NORMAL: 可鼠标缩放; WINDOW_KEEPRATIO: 保持纵横比
                cv.namedWindow("Depth Image",
                               cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
                self._depth_win_open = True
            cv.imshow("Depth Image", frame)
        policy_frame = self.robot_data_.policy_vis_frame_
        if policy_frame is not None:
            if not self._policy_win_open:
                cv.namedWindow("WALKDEPTH obs",
                               cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
                self._policy_win_open = True
            cv.imshow("WALKDEPTH obs", policy_frame)
        if self._depth_win_open or self._policy_win_open:
            cv.waitKey(1)  # 一次 waitKey 即刷新全部 HighGUI 窗口

    def close_debug_window(self):
        """主线程(destroy_node)销毁调试窗口"""
        if self._depth_win_open or self._policy_win_open:
            cv.destroyAllWindows()
            self._depth_win_open = False
            self._policy_win_open = False

    def finish_control(self):
        """控制线程退出前调用(本实现无控制线程内资源需清理)"""
        return

    def shutdown(self):
        """ROS 图像模式下无本地相机资源需要释放。"""
        # if self.orbbec_spawner is not None:
        #     self.orbbec_spawner.stop()
        #     self.orbbec_spawner = None
        return

    def update_sensor_states(self):
        # 获取Imu数据
        self.get_imu_data()
        # 添加IMU偏移
        self.robot_data_.imu_data_[2] += self.xsense_roll_offset

    def update_robot_data(self, flag: ControlFlag, time_passed: float):
        # 更新传感器数据
        self.update_sensor_states()
        # 更新电机状态数据
        self.update_robot_state()
        # 更新机器人控制命令
        self.update_robot_cmd(flag)
        # Snapshot normalized actuator feedback into independent serial buffers.
        np.copyto(self.robot_data_.q_a_s_, self.robot_data_.q_a_p_)
        np.copyto(self.robot_data_.q_dot_a_s_, self.robot_data_.q_dot_a_p_)
        np.copyto(self.robot_data_.tau_a_s_, self.robot_data_.tau_a_p_)
        # Only the ankle entries require parallel -> serial kinematics.
        if not self.sim:
            #真机
            self.ankle_parallel_to_serial()

        # 更新robot_data 时间戳
        self.robot_data_.time_now_ = time_passed

    def _check_and_clip_joint_limits_fast(
            self, cmd_name: int, position: float) -> tuple[bool, float]:
        """
        快速检查并修正关节位置限位（避免重复查询）

        Disabled by default.
        """
        if not self.sim and self.clip_actions:
            limit = self.id_to_limits[cmd_name]
            clipped_pos = np.clip(position, limit["min"], limit["max"])
            return clipped_pos == position, clipped_pos
        else:
            return True, position

    # @timing_decorator
    def convert_to_motor_commands(self):
        """Map actuator targets to wire coordinates without modifying s or p."""
        rd = self.robot_data_
        np.copyto(rd.q_d_motor_, rd.q_d_p_)
        np.copyto(rd.q_dot_d_motor_, rd.q_dot_d_p_)
        np.copyto(rd.tau_d_motor_, rd.tau_d_p_)
        tail = slice(self.floating_base_dof, None)
        np.subtract(rd.q_d_p_[tail], self.zero_offset, out=self.temp_q)
        np.multiply(self.zero_cnt, 2.0 * math.pi, out=self._temp_zero_cnt)
        np.subtract(self.temp_q, self._temp_zero_cnt, out=self.temp_q)
        np.multiply(self.temp_q, self.motor_dir, out=self.temp_q)
        np.add(self.temp_q, self.zero_pos, out=rd.q_d_motor_[tail])
        np.multiply(rd.q_dot_d_p_[tail], self.motor_dir, out=rd.q_dot_d_motor_[tail])
        np.multiply(rd.tau_d_p_[tail], self.motor_dir, out=rd.tau_d_motor_[tail])

        if self.disable_joints_:
            rd.joint_kp_p_.fill(0.0)
            rd.joint_kd_p_.fill(0.0)
            rd.tau_d_motor_.fill(0.0)
            self.node.get_logger().warn("Joints disabled!")

    def publish_motor_commands(self, flag: ControlFlag):
        # flag_fsm_command = flag.fsm_state_command
        current_state = self.current_state
        # 发布腿部控制命令
        if self.legs_control_status == [] or current_state in self.legs_control_status:
            leg_msg = CmdMotorCtrl()
            leg_msg.header.stamp = self.node.get_clock().now().to_msg()
            for i in range(self.id_map.leg_motor_nums):
                cmd = MotorCtrl()
                cmd.name = self.id_map.get_id_by_index(i)
                cmd.kp = float(self.robot_data_.joint_kp_p_[i])
                cmd.kd = float(self.robot_data_.joint_kd_p_[i])
                cmd.pos = float(self.robot_data_.q_d_motor_[i +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_motor_[i + self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_motor_[i + self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    print(
                        f"Joint (id: {cmd.name}) position {cmd.pos} is out of limits"
                    )
                leg_msg.cmds.append(cmd)
            self.pub_leg_motor_cmd.publish(leg_msg)

        # 只在特定模式下控制腰部
        if current_state in self.waist_control_status:
            # 腰部控制命令
            waist_msg = CmdMotorCtrl()
            waist_msg.header.stamp = self.node.get_clock().now().to_msg()
            for i in range(self.id_map.waist_motor_nums):
                cmd = MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums
                cmd.name = self.id_map.get_id_by_index(
                    motor_idx)  # 12 -> 33(yaw)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_motor_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_motor_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_motor_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    print(
                        f"Joint (id: {cmd.name}) position {cmd.pos} is out of limits"
                    )
                waist_msg.cmds.append(cmd)
            # print(f'waist_msg {waist_msg}')
            self.pub_waist_motor_cmd.publish(waist_msg)

        # 只在特定模式下控制手臂
        if current_state in self.arms_control_status:
            # 手臂控制命令
            arm_msg = CmdMotorCtrl()
            arm_msg.header.stamp = self.node.get_clock().now().to_msg()
            if current_state in self.left_arm_only_status:
                control_index = np.arange(0, 7)
            elif current_state in self.right_arm_only_status:
                control_index = np.arange(self.id_map.arm_motor_nums - 7, self.id_map.arm_motor_nums)
            else:
                control_index = np.arange(0, self.id_map.arm_motor_nums)
            for i in control_index:
                cmd = MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_motor_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_motor_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_motor_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    print(
                        f"Joint (id: {cmd.name}) position {cmd.pos} is out of limits"
                    )
                arm_msg.cmds.append(cmd)
            # print(f'arm_msg {arm_msg}')
            self.pub_arm_motor_cmd.publish(arm_msg)

    # @timing_decorator
    def send_motor_commands(self, flag: ControlFlag):
        """发布电机控制命令"""
        if not self.initialized:
            return

        # Start EVERY send from serial targets, including decimated/hold ticks.
        # The solver may only overwrite independent actuator output buffers.
        rd = self.robot_data_
        np.copyto(rd.q_d_p_, rd.q_d_s_)
        np.copyto(rd.q_dot_d_p_, rd.q_dot_d_s_)
        np.copyto(rd.tau_d_p_, rd.tau_d_s_)
        np.copyto(rd.joint_kp_p_, rd.joint_kp_s_)
        np.copyto(rd.joint_kd_p_, rd.joint_kd_s_)

        if not self.sim:
            #真机
            self.ankle_serial_to_parallel()
        self.convert_to_motor_commands()
        self.publish_motor_commands(flag)

    # ROS回调函数
    def _leg_motor_status_callback(self, msg):
        """腿部电机状态回调"""
        try:
            self.queue_leg_motor_state.put_nowait(msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_leg_motor_state.get_nowait()  # 移除旧数据
                self.queue_leg_motor_state.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _arm_motor_status_callback(self, msg):
        """手臂电机状态回调"""
        try:
            self.queue_arm_motor_state.put_nowait(msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_arm_motor_state.get_nowait()  # 移除旧数据
                self.queue_arm_motor_state.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _waist_motor_status_callback(self, msg):
        """腰部电机状态回调"""
        try:
            self.queue_waist_motor_state.put_nowait(msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_waist_motor_state.get_nowait()  # 移除旧数据
                self.queue_waist_motor_state.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _imu_status_callback(self, msg):
        """IMU状态回调"""
        try:
            self.queue_imu_xsens.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_imu_xsens.get_nowait()  # 移除旧数据
                self.queue_imu_xsens.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略


    def ankle_parallel_to_serial(self):
        # 串并联转换：并转串 (类似C++版本中的处理)
        # 提取左右脚两个踝关节（并联关节）
        q_a_p = np.zeros(4)  # 并联关节角度（实际）
        qdot_a_p = np.zeros(4)  # 并联关节速度（实际）
        tor_a_p = np.zeros(4)  # 并联关节力矩（实际）
        q_a_s = np.zeros(4)  # 串联关节角度（实际）
        qdot_a_s = np.zeros(4)  # 串联关节速度（实际）
        tor_a_s = np.zeros(4)  # 串联关节力矩（实际）

        q_a_p[:2] = self.robot_data_.q_a_p_[
            self.left_ankle_indices]  # 左脚踝关节 (pitch, roll)
        q_a_p[2:] = self.robot_data_.q_a_p_[
            self.right_ankle_indices]  # 右脚踝关节 (pitch, roll)

        qdot_a_p[:2] = self.robot_data_.q_dot_a_p_[self.left_ankle_indices]
        qdot_a_p[2:] = self.robot_data_.q_dot_a_p_[self.right_ankle_indices]

        tor_a_p[:2] = self.robot_data_.tau_a_p_[self.left_ankle_indices]
        tor_a_p[2:] = self.robot_data_.tau_a_p_[self.right_ankle_indices]

        if self.debug:
            print("-" * 20 + "并转串联前" + "-" * 20)
            print("q_a_p:", q_a_p)
            print("qdot_a_p:", qdot_a_p)
            print("tor_a_p:", tor_a_p)

        # 计算并转串（正运动学）
        self.fun_s2p.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        self.fun_s2p.calcFK()
        self.fun_s2p.calcIK()

        # 3.0 sptlib 使用返回值；不再向预分配数组原地写入。
        success, q_a_s, qdot_a_s, tor_a_s = self.fun_s2p.get_s_state()

        if self.debug:
            print("-" * 20 + "并转串联后" + "-" * 20)
            print("q_a_s:", q_a_s)
            print("qdot_a_s:", qdot_a_s)
            print("tor_a_s:", tor_a_s)

        # Write serial feedback; preserve the parallel measurement buffers
        self.robot_data_.q_a_s_[self.left_ankle_indices] = q_a_s[:2]  # 左脚踝关节串联值
        self.robot_data_.q_a_s_[self.right_ankle_indices] = q_a_s[2:]  # 右脚踝关节串联值

        self.robot_data_.q_dot_a_s_[self.left_ankle_indices] = qdot_a_s[:2]
        self.robot_data_.q_dot_a_s_[self.right_ankle_indices] = qdot_a_s[2:]

        self.robot_data_.tau_a_s_[self.left_ankle_indices] = tor_a_s[:2]
        self.robot_data_.tau_a_s_[self.right_ankle_indices] = tor_a_s[2:]

    def ankle_serial_to_parallel(self):
        # 串转并：将串联关节命令转换为并联关节命令（类似C++版本）
        # 提取踝关节两关节的串联命令
        q_d_p = np.zeros(4)  # 并联关节角度（期望）
        qdot_d_p = np.zeros(4)  # 并联关节速度（期望）
        tor_d_p = np.zeros(4)  # 并联关节力矩（期望）
        q_d_s = np.zeros(4)  # 串联关节角度（期望）
        qdot_d_s = np.zeros(4)  # 串联关节速度（期望）
        tor_d_s = np.zeros(4)  # 串联关节力矩（期望）

        q_d_s[:2] = self.robot_data_.q_d_s_[self.left_ankle_indices]  # 左脚踝关节串联命令
        q_d_s[2:] = self.robot_data_.q_d_s_[
            self.right_ankle_indices]  # 右脚踝关节串联命令

        qdot_d_s[:2] = self.robot_data_.q_dot_d_s_[self.left_ankle_indices]
        qdot_d_s[2:] = self.robot_data_.q_dot_d_s_[self.right_ankle_indices]

        tor_d_s[:2] = self.robot_data_.tau_d_s_[self.left_ankle_indices]
        tor_d_s[2:] = self.robot_data_.tau_d_s_[self.right_ankle_indices]

        q_a_s = np.zeros(4)  # 串联关节角度（实际）
        qdot_a_s = np.zeros(4)  # 串联关节速度（实际）
        q_a_s[:2] = self.robot_data_.q_a_s_[self.left_ankle_indices]  # 左脚踝关节串联值
        q_a_s[2:] = self.robot_data_.q_a_s_[self.right_ankle_indices]  # 右脚踝关节串联值
        qdot_a_s[:2] = self.robot_data_.q_dot_a_s_[
            self.left_ankle_indices]  # 左脚踝关节串联速度
        qdot_a_s[2:] = self.robot_data_.q_dot_a_s_[
            self.right_ankle_indices]  # 右脚踝关节串联速度

        kp = np.zeros(4)  # 串联关节刚度
        kd = np.zeros(4)  # 串联关节阻尼
        kp[:2] = self.robot_data_.joint_kp_s_[self.left_ankle_indices -
                                              self.floating_base_dof]
        kp[2:] = self.robot_data_.joint_kp_s_[self.right_ankle_indices -
                                              self.floating_base_dof]
        kd[:2] = self.robot_data_.joint_kd_s_[self.left_ankle_indices -
                                              self.floating_base_dof]
        kd[2:] = self.robot_data_.joint_kd_s_[self.right_ankle_indices -
                                              self.floating_base_dof]

        tor_d_s = kp * (q_d_s - q_a_s) + kd * (qdot_d_s - qdot_a_s)

        if self.debug:
            print("-" * 20 + "串转并联前" + "-" * 20)
            print("q_d_s:", q_d_s)
            print("qdot_d_s:", qdot_d_s)
            print("tor_d_s:", tor_d_s)

        # 串转并计算
        self.fun_s2p.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        self.fun_s2p.calc_joint_pos_ref()
        self.fun_s2p.calc_joint_tor_des()
        success, q_d_p, qdot_d_p, tor_d_p = self.fun_s2p.get_p_des()

        ankle_indices = np.concatenate((self.left_ankle_indices, self.right_ankle_indices))
        q_a_p = self.robot_data_.q_a_p_[ankle_indices]
        qdot_a_p = self.robot_data_.q_dot_a_p_[ankle_indices]
        q_d_p = (tor_d_p - self.ankle_kd_p *
                 (qdot_d_p - qdot_a_p)) / self.ankle_kp_p + q_a_p

        if self.debug:
            print("-" * 20 + "串转并联后" + "-" * 20)
            print("q_d_p:", q_d_p)
            print("qdot_d_p:", qdot_d_p)
            print("tor_d_p:", tor_d_p)

        # Write only actuator targets/gains; serial policy targets stay intact
        self.robot_data_.q_d_p_[self.left_ankle_indices] = q_d_p[:2]  # 左脚踝关节并联命令
        self.robot_data_.q_d_p_[self.right_ankle_indices] = q_d_p[2:]  # 右脚踝关节并联命令

        self.robot_data_.q_dot_d_p_[self.left_ankle_indices] = qdot_d_p[:2]
        self.robot_data_.q_dot_d_p_[self.right_ankle_indices] = qdot_d_p[2:]

        # self.robot_data_.tau_d_p_[self.left_ankle_indices] = tor_d_p[:2]
        # self.robot_data_.tau_d_p_[self.right_ankle_indices] = tor_d_p[2:]
        self.robot_data_.tau_d_p_[self.left_ankle_indices] = 0.0
        self.robot_data_.tau_d_p_[self.right_ankle_indices] = 0.0
        # 更新脚踝Kp,Kd
        self.robot_data_.joint_kp_p_[
            self.left_ankle_indices -
            self.floating_base_dof] = self.ankle_kp_p[:2]
        self.robot_data_.joint_kp_p_[
            self.right_ankle_indices -
            self.floating_base_dof] = self.ankle_kp_p[2:]
        self.robot_data_.joint_kd_p_[
            self.left_ankle_indices -
            self.floating_base_dof] = self.ankle_kd_p[:2]
        self.robot_data_.joint_kd_p_[
            self.right_ankle_indices -
            self.floating_base_dof] = self.ankle_kd_p[2:]

    def _accept_walk_command(self, flag: ControlFlag) -> bool:
        """Return True when velocity command may enter RobotData.walk_cmd_."""
        return bool(flag.enable) and self.current_state in self.walk_command_status

    def update_robot_cmd(self, flag: ControlFlag):
        """更新机器人控制命令.

        Controller adapters continuously publish desired walk velocities, but
        this bridge is the final safety gate.  Disabled input or non-walking
        FSM states write explicit zero speed instead of preserving old values.
        """
        if not self._accept_walk_command(flag):
            self.robot_data_.walk_cmd_[:] = 0.0
        else:
            x_command = float(getattr(flag, 'x_speed_command', 0.0))
            y_command = float(getattr(flag, 'y_speed_command', 0.0))
            yaw_command = float(getattr(flag, 'yaw_speed_command', 0.0))

            self.robot_data_.walk_cmd_[:] = [x_command, y_command, yaw_command]

            if not hasattr(self, "command_counter"):
                self.command_counter = 0
            self.command_counter += 1
            if self.command_counter % 50 == 0:
                print(f"Updated walk_cmd_: {self.robot_data_.walk_cmd_}")
                self.command_counter = 0
        self.robot_data_.control_flag = copy.deepcopy(flag)

def get_robot_interface(robot_data: RobotData, config_path: str) -> RobotInterface:
    """工厂函数，返回机器人接口实例"""
    return RobotInterfaceImpl(robot_data, config_path)
