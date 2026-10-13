"""
深度感知行走策略状态 (WALKDEPTH)

移植自 bitbot_gz 的 perceptive locomotion (ovinf HikePolicy):
  - 观测: 8帧本体感知历史(每帧78维) + 8帧深度历史(每帧32x18, 间隔10个tick)
  - 深度预处理同 Instinct510 训练 noise_pipeline: 加噪 -> crop -> 模糊
    -> 裁剪[0,max] -> 归一化 (归一化在最后)
  - 单输入 [1, 624+4608], 单输出 [1, 23]
关节顺序: 训练(Isaaclab)顺序与 walk_amp.yaml 的 joint_names 完全一致,
经 lab2mj 名称映射 scatter 到 29 关节 MuJoCo 顺序, 与 ovinf 的自动解析等价。
相机参数(width/crop/blur/min-max 等)来自 config/dex_config.yaml 的 depth 段。
"""

from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import yaml

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData


class FSMStateWALKDEPTH(FSMState):
    """WALKDEPTH 深度感知行走策略状态"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "walk_depth.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        # 深度相机参数与 robot_interface 共用 dex_config.yaml 的 depth 段
        dex_config_path = os.path.join(current_dir, "..", "..", "config",
                                       "dex_config.yaml")
        with open(dex_config_path, 'r') as f:
            depth_cfg = yaml.safe_load(f)['robot_interface']['depth']

        self.action_num_ = policy_config.get('actions_size')
        self.motor_num_ = policy_config.get('motor_num')
        self.dt_ = policy_config.get('dt')

        size_config = policy_config.get('size', {})
        self.num_hist_ = size_config.get('num_hist')
        self.obs_size_ = size_config.get('observations_size')

        control_config = policy_config.get('control', {})
        self.action_scale_ = control_config.get('action_scale')
        self.decimation_ = control_config.get('decimation')
        self.warm_start_time_ = control_config.get('warm_start_time', 0.3)

        norm_config = policy_config.get('normalization', {})
        clip_config = norm_config.get('clip_scales', {})
        obs_config = norm_config.get('obs_scales', {})
        self.clip_obs_ = clip_config.get('clip_observations', 100.0)
        self.clip_act_ = clip_config.get('clip_actions', 18.0)
        self.lin_vel_scale_ = obs_config.get('lin_vel', 1.0)
        self.ang_vel_scale_ = obs_config.get('ang_vel', 0.25)
        self.command_scale_ = obs_config.get('command', 1.0)
        self.dof_pos_scale_ = obs_config.get('dof_pos', 1.0)
        self.dof_vel_scale_ = obs_config.get('dof_vel', 0.05)
        self.proj_gravity_scale_ = obs_config.get('proj_gravity', 1.0)

        depth_config = policy_config.get('depth', {})
        self.depth_num_images_ = depth_config.get('num_images', 8)
        self.depth_image_skip_ = depth_config.get('image_skip', 10)
        self.depth_debug_vis_ = bool(depth_config.get('debug_vis', False))

        # 相机/预处理参数: dex_config.yaml robot_interface.depth
        # (对应 bitbot-ovinf dex_mj.xml 的 MujocoDepthCamera 设备)
        self.depth_w_ = int(depth_cfg['width'])
        self.depth_h_ = int(depth_cfg['height'])
        self.depth_min_ = float(depth_cfg.get('min_depth', 0.0))
        self.depth_max_ = float(depth_cfg.get('max_depth', 2.5))
        self.depth_noise_ = float(depth_cfg.get('noise_stddev', 0.0))
        self.depth_normalize_ = bool(depth_cfg.get('normalize', True))
        self.depth_crop_l_ = int(depth_cfg.get('crop_left', 0))
        self.depth_crop_r_ = int(depth_cfg.get('crop_right', 0))
        self.depth_crop_t_ = int(depth_cfg.get('crop_top', 0))
        self.depth_crop_b_ = int(depth_cfg.get('crop_bottom', 0))
        self.depth_blur_ksize_ = int(depth_cfg.get('blur_kernel', 3))
        self.depth_blur_sigma_ = float(depth_cfg.get('blur_sigma', 1.0))
        # 送入策略的帧尺寸 = 相机分辨率裁剪后 (ovinf obs_width_/obs_height_)
        self.depth_crop_w_ = self.depth_w_ - self.depth_crop_l_ - self.depth_crop_r_
        self.depth_crop_h_ = self.depth_h_ - self.depth_crop_t_ - self.depth_crop_b_

        # 23 关节(训练顺序) -> 29 关节(mujoco xml 顺序)
        joint_names = policy_config.get('joint_names')
        if joint_names is None:
            raise ValueError("[FSMStateWALKDEPTH] Missing 'joint_names' in walk_depth.yaml")
        self.joint_seq = list(joint_names)

        self.action_scale = (np.full(len(self.joint_seq), float(self.action_scale_))
                             if np.isscalar(self.action_scale_)
                             else np.array(self.action_scale_, dtype=np.float32))
        if len(self.action_scale) != len(self.joint_seq):
            raise ValueError("[FSMStateWALKDEPTH] control.action_scale length mismatch")

        init_state_config = policy_config.get('init_state', {})
        default_joint_angles = init_state_config.get('default_joint_angles')
        if default_joint_angles is None:
            raise ValueError("[FSMStateWALKDEPTH] Missing 'init_state.default_joint_angles'")
        self.joint_pos_array_seq = np.array(default_joint_angles, dtype=np.float32)
        if len(self.joint_pos_array_seq) != len(self.joint_seq):
            raise ValueError("[FSMStateWALKDEPTH] default_joint_angles length mismatch")

        # gains_config = policy_config.get('gains', {})
        # self.stiffness_array_seq = np.array(gains_config.get('kp'), dtype=np.float32)
        # self.damping_array_seq = np.array(gains_config.get('kd'), dtype=np.float32)
        # if (len(self.stiffness_array_seq) != len(self.joint_seq)
        #         or len(self.damping_array_seq) != len(self.joint_seq)):
        #     raise ValueError("[FSMStateWALKDEPTH] gains.kp/kd length mismatch")

        self.joint_xml = [
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ]

        self.lab2mj = np.array(
            [self.joint_xml.index(name) for name in self.joint_seq], dtype=int)

        # 23 维(lab) scatter 到 29 维(mj), 其余 6 个腕关节保持 0
        n_mj = len(self.joint_xml)
        # self.joint_pos_array = np.zeros(n_mj, dtype=np.float32)
        # self.stiffness_array = np.zeros(n_mj, dtype=np.float32)
        # self.damping_array = np.zeros(n_mj, dtype=np.float32)
        # for lab_idx, mj_idx in enumerate(self.lab2mj):
        #     self.joint_pos_array[mj_idx] = self.joint_pos_array_seq[lab_idx]
        #     self.stiffness_array[mj_idx] = self.stiffness_array_seq[lab_idx]
        #     self.damping_array[mj_idx] = self.damping_array_seq[lab_idx]

        gains_config = policy_config.get('gains', {})
        self.stiffness_array = np.array(gains_config.get('kp'), dtype=np.float32)
        self.damping_array = np.array(gains_config.get('kd'), dtype=np.float32)
        if len(self.stiffness_array) != len(self.joint_xml):
            raise ValueError("[FSMStateSTAND] gains.kp must be 29-length (xml order)")
        self.hold_pos_ = np.array(policy_config.get('hold_joint_angles'),
                                  dtype=np.float32)
        if len(self.hold_pos_) != len(self.joint_xml):
            raise ValueError("[FSMStateSTAND] hold_joint_angles must be 29-length (xml order)")

        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale

        # 缓冲区: 本体历史 + 深度 skipped ring + 动作
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.depth_ring_ = deque(
            maxlen=(self.depth_num_images_ - 1) * self.depth_image_skip_ + 1)
        self._depth_neutral_frame_ = np.full(
            self.depth_crop_w_ * self.depth_crop_h_, 0.5, dtype=np.float32)

        step = (self.decimation_ if self.decimation_ else 1) * self.dt_
        if self.warm_start_time_ > 0 and step > 0:
            self._warm_start_steps = max(1, int(self.warm_start_time_ / step))
        else:
            self._warm_start_steps = 0
        self._warmup_inference_counter = 0
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)

        self.is_first_obs_ = True
        self._depth_warned = False
        self._depth_received = False
        self._policy_tick_count = 0
        self._policy_t0 = 0.0

        self.model_path = os.path.join(current_dir, "model", policy_config["model_path"])
        self._init_onnx_session()

    def _init_onnx_session(self):
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.enable_mem_pattern = False
            options.enable_mem_reuse = True
            self.ort_session_ = ort.InferenceSession(
                self.model_path, options, providers=['CPUExecutionProvider'])
            print(f"[FSMStateWALKDEPTH-ONNX] ONNX model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"[FSMStateWALKDEPTH] Failed to load ONNX model: {e}")
            self.ort_session_ = None

    def on_enter(self):
        """进入WALKDEPTH: 热身填充(同 ovinf WarmUp: 本体历史置零, 深度历史填0.5)"""
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)
        self.depth_ring_.clear()
        for _ in range(self.depth_ring_.maxlen):
            self.depth_ring_.append(self._depth_neutral_frame_)
        self.is_first_obs_ = True
        self._warmup_inference_counter = 0
        if self.robot_data_ is not None:
            try:
                self._warm_start_pose = self.robot_data_.get_joint_pos().copy()
            except Exception:
                self._warm_start_pose.fill(0.0)
        else:
            self._warm_start_pose.fill(0.0)
        print("[FSMStateWALKDEPTH] enter")

    def _preprocess_depth(self, img) -> np.ndarray:
        """训练侧 Instinct510 dex23dof_depth_cfg.py noise_pipeline 的移植。

        顺序: 缩放到相机分辨率 -> 加性高斯噪声(米制) -> crop ->
        高斯模糊(reflect) -> 裁剪[min,max] -> 归一化, 归一化在最后。
        """
        img = np.asarray(img, dtype=np.float32)
        if img.shape != (self.depth_h_, self.depth_w_):
            # img = cv2.resize(img, (self.depth_w_, self.depth_h_),
            #                  interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (self.depth_w_, self.depth_h_),
                             interpolation=cv2.INTER_NEAREST)
        else:
            img = img.copy()
        if self.depth_noise_ > 0.0:
            img += np.random.normal(0.0, self.depth_noise_, img.shape).astype(np.float32)
        h = self.depth_crop_h_
        w = self.depth_crop_w_
        processed = img[self.depth_crop_t_:self.depth_crop_t_ + h,
                        self.depth_crop_l_:self.depth_crop_l_ + w]
        if self.depth_blur_ksize_ > 0:
            # torchvision GaussianBlur 默认 'reflect' 填充 == REFLECT_101
            processed = cv2.GaussianBlur(
                processed, (self.depth_blur_ksize_, self.depth_blur_ksize_),
                self.depth_blur_sigma_, borderType=cv2.BORDER_REFLECT_101)
        processed = np.clip(processed, self.depth_min_, self.depth_max_)
        if self.depth_normalize_:
            processed = (processed - self.depth_min_) / (self.depth_max_ - self.depth_min_)
        return processed.reshape(-1)

    def _show_depth_frame(self, frame):
        """弹窗显示送入策略的处理帧 (0=近, 255=远)"""
        try:
            vis = (frame.reshape(self.depth_crop_h_, self.depth_crop_w_)
                   * 255.0).astype(np.uint8)
            vis = cv2.resize(vis, (self.depth_crop_w_ * 8, self.depth_crop_h_ * 8),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imshow('walkdepth_depth_obs', vis)
            cv2.waitKey(1)
        except cv2.error:
            print("[FSMStateWALKDEPTH] debug_vis 需要 X 显示, 已关闭")
            self.depth_debug_vis_ = False

    def _get_depth_history(self) -> np.ndarray:
        """按 image_skip 间隔取 num_images 帧, 旧->新排列"""
        frames = []
        for k in range(self.depth_num_images_ - 1, -1, -1):
            frames.append(self.depth_ring_[-1 - k * self.depth_image_skip_])
        return np.concatenate(frames)

    def compute_observation(self, flag: ControlFlag) -> np.ndarray:
        """单帧78维本体观测, 排布同 ovinf_hike.cc:131-146"""
        x_cmd, y_cmd, yaw_cmd = self.robot_data_.get_walk_cmd()
        command = np.array([
            x_cmd * self.lin_vel_scale_,
            y_cmd * self.lin_vel_scale_,
            yaw_cmd * self.ang_vel_scale_,
        ], dtype=np.float32) * self.command_scale_

        q_mj = self.robot_data_.get_joint_pos()
        dq_mj = self.robot_data_.get_joint_vel()
        qj = (q_mj[self.lab2mj] - self.default_angles_lab) * self.dof_pos_scale_
        dqj = dq_mj[self.lab2mj] * self.dof_vel_scale_

        ang_vel = self.robot_data_.get_angular_velocity() * self.ang_vel_scale_
        proj_gravity = self.robot_data_.get_project_gravity() * self.proj_gravity_scale_

        proprio = np.concatenate([
            command,
            qj.astype(np.float32),
            dqj.astype(np.float32),
            self.last_actions_,
            ang_vel.astype(np.float32),
            proj_gravity.astype(np.float32),
        ])

        # 深度: 每个策略tick把最新一帧预处理后入ring
        img = self.robot_data_.depth_image_
        if img is not None:
            self._depth_received = True
            frame = self._preprocess_depth(img)
            self.depth_ring_.append(frame)
            if self.depth_debug_vis_:
                self._show_depth_frame(frame)
        else:
            if not self._depth_warned:
                print("[FSMStateWALKDEPTH] no depth image yet, using neutral 0.5 frames")
                self._depth_warned = True
            self.depth_ring_.append(self._depth_neutral_frame_)

        # 本体历史: 首8帧用当前帧填满(walk_amp做法), 之后滚动
        if self.is_first_obs_:
            for i in range(self.num_hist_):
                s = i * self.obs_size_
                self.proprio_hist_buf_[s:s + self.obs_size_] = proprio
            self.is_first_obs_ = False
        else:
            shift = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift] = self.proprio_hist_buf_[self.obs_size_:]
            self.proprio_hist_buf_[shift:] = proprio

        return np.clip(self.proprio_hist_buf_, -self.clip_obs_, self.clip_obs_)

    def compute_actions(self, policy_input: np.ndarray):
        if self.ort_session_ is None:
            return
        try:
            input_name = self.ort_session_.get_inputs()[0].name
            outputs = self.ort_session_.run(
                None, {input_name: policy_input.reshape(1, -1).astype(np.float32)})
            self.actions_ = np.clip(outputs[0][0], -self.clip_act_, self.clip_act_)
        except Exception as e:
            print(f"[FSMStateWALKDEPTH] ONNX Runtime inference error: {e}")

    def run(self, flag: ControlFlag):
        """运行WALKDEPTH状态"""
        if int(self.robot_data_.time_now_ / self.dt_) % self.decimation_ == 0:
            # self._policy_tick_count += 1
            # now = time.perf_counter()
            # if self._policy_tick_count == 1:
            #     self._policy_t0 = now
            # elif self._policy_tick_count % 250 == 0:
            #     print(f"[FSMStateWALKDEPTH] policy rate: "
            #           f"{250.0 / (now - self._policy_t0):.1f} Hz")
            #     self._policy_t0 = now

            obs_hist = self.compute_observation(flag)
            depth_hist = self._get_depth_history()
            policy_input = np.concatenate([obs_hist, depth_hist])
            self.compute_actions(policy_input)

            target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab
            # target_dof_pos_mj = self.robot_data_.get_joint_pos().copy()
            target_dof_pos_mj = self.hold_pos_.copy()
            target_dof_pos_mj[self.lab2mj] = target_dof_pos_lab

            commanded_pos = target_dof_pos_mj
            if self._warm_start_steps > 0 and self._warmup_inference_counter < self._warm_start_steps:
                self._warmup_inference_counter += 1
                blend = self._warmup_inference_counter / float(self._warm_start_steps)
                commanded_pos = (1.0 - blend) * self._warm_start_pose + blend * target_dof_pos_mj

            base = self.robot_data_.q_d_.shape[0] - self.motor_num_
            self.robot_data_.q_d_[base:base + len(self.joint_xml)] = commanded_pos
            self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
            self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0
            self.last_actions_[:] = self.actions_

        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array

    def on_exit(self):
        print("[FSMStateWALKDEPTH] exit")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        elif flag.fsm_state_command == "gotoSTAND":
            return FSMStateName.STAND
        else:
            return None
