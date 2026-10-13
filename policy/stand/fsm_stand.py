"""
站立/迈步策略状态 (STAND)

移植自 bitbot_gz 的 policy_standing (ovinf HumanoidStandPolicy, policy_msl):
  - 15 关节(双腿+腰)策略, 手臂保持在 hold_joint_angles
  - 观测: 15帧历史(每帧56维): [sin,cos 步态时钟 | cmd | q-def | dq | last_action | ang_vel | gravity]
  - 步态时钟(ovinf_humanoid_stand.cc:106-130): 命令范数 > stand_threshold 时启动,
    命令归零后在周期边界(|sin|<0.05 且 cos>0)停止; 静止时时钟为 [0, 1]
  - 单输入 [1, 840], 单输出 [1, 15], 目标 = action*0.25 + 默认角
关节顺序: joint_names 为训练(Isaaclab)顺序, 经 lab2mj 名称映射 scatter 到
29 关节 MuJoCo 顺序。
"""

import numpy as np
import onnxruntime as ort
import os
import yaml

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData


class FSMStateSTAND(FSMState):
    """STAND 站立/迈步策略状态"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "stand.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)

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
        self.cycle_time_ = control_config.get('cycle_time', 0.9)
        self.stand_threshold_ = control_config.get('stand_threshold', 0.045)

        norm_config = policy_config.get('normalization', {})
        clip_config = norm_config.get('clip_scales', {})
        obs_config = norm_config.get('obs_scales', {})
        self.clip_obs_ = clip_config.get('clip_observations', 100.0)
        self.clip_act_ = clip_config.get('clip_actions', 18.0)
        self.lin_vel_scale_ = obs_config.get('lin_vel', 2.0)
        self.ang_vel_scale_ = obs_config.get('ang_vel', 1.0)
        self.command_scale_ = obs_config.get('command', 1.0)
        self.dof_pos_scale_ = obs_config.get('dof_pos', 1.0)
        self.dof_vel_scale_ = obs_config.get('dof_vel', 0.05)
        self.proj_gravity_scale_ = obs_config.get('proj_gravity', 1.0)

        joint_names = policy_config.get('joint_names')
        if joint_names is None:
            raise ValueError("[FSMStateSTAND] Missing 'joint_names' in stand.yaml")
        self.joint_seq = list(joint_names)

        self.action_scale = (np.full(len(self.joint_seq), float(self.action_scale_))
                             if np.isscalar(self.action_scale_)
                             else np.array(self.action_scale_, dtype=np.float32))
        if len(self.action_scale) != len(self.joint_seq):
            raise ValueError("[FSMStateSTAND] control.action_scale length mismatch")

        init_state_config = policy_config.get('init_state', {})
        default_joint_angles = init_state_config.get('default_joint_angles')
        if default_joint_angles is None:
            raise ValueError("[FSMStateSTAND] Missing 'init_state.default_joint_angles'")
        self.joint_pos_array_seq = np.array(default_joint_angles, dtype=np.float32)
        if len(self.joint_pos_array_seq) != len(self.joint_seq):
            raise ValueError("[FSMStateSTAND] default_joint_angles length mismatch")

        # 15 关节(训练顺序) -> 29 关节(mujoco xml 顺序) 名称映射
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

        # 增益与保持角直接按 xml 顺序给出(29维)
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

        # 缓冲区
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)

        step = (self.decimation_ if self.decimation_ else 1) * self.dt_
        if self.warm_start_time_ > 0 and step > 0:
            self._warm_start_steps = max(1, int(self.warm_start_time_ / step))
        else:
            self._warm_start_steps = 0
        self._warmup_inference_counter = 0
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)

        self.is_first_obs_ = True
        self.gait_started_ = False
        self.gait_time_ = 0.0

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
            print(f"[FSMStateSTAND-ONNX] ONNX model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"[FSMStateSTAND] Failed to load ONNX model: {e}")
            self.ort_session_ = None

    def on_enter(self):
        """进入STAND: 历史清零, 时钟复位(同 ovinf WarmUp)"""
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)
        self.is_first_obs_ = True
        self.gait_started_ = False
        self.gait_time_ = 0.0
        self._warmup_inference_counter = 0
        if self.robot_data_ is not None:
            try:
                self._warm_start_pose = self.robot_data_.get_joint_pos().copy()
            except Exception:
                self._warm_start_pose.fill(0.0)
        else:
            self._warm_start_pose.fill(0.0)
        print("[FSMStateSTAND] enter")

    def compute_observation(self, flag: ControlFlag) -> np.ndarray:
        """单帧56维观测 + 步态时钟, 排布同 ovinf_humanoid_stand.cc:105-148"""
        cmd = np.array(self.robot_data_.get_walk_cmd(), dtype=np.float32)
        cmd_norm = float(np.linalg.norm(cmd))

        # 步态时钟: 启动/停止逻辑与 ovinf 一致
        if (not self.gait_started_) and cmd_norm > self.stand_threshold_:
            self.gait_started_ = True
            self.gait_time_ = 0.0
        elif (self.gait_started_ and cmd_norm < self.stand_threshold_
              and abs(np.sin(2 * np.pi * self.gait_time_ / self.cycle_time_)) < 0.05
              and np.cos(2 * np.pi * self.gait_time_ / self.cycle_time_) > 0.0):
            self.gait_started_ = False
        if self.gait_started_:
            self.gait_time_ += self.dt_ * (self.decimation_ or 1)
            phase = 2 * np.pi * self.gait_time_ / self.cycle_time_
        else:
            phase = 0.0

        command = np.array([
            cmd[0] * self.lin_vel_scale_,
            cmd[1] * self.lin_vel_scale_,
            cmd[2] * self.ang_vel_scale_,
        ], dtype=np.float32) * self.command_scale_

        q_mj = self.robot_data_.get_joint_pos()
        dq_mj = self.robot_data_.get_joint_vel()
        qj = (q_mj[self.lab2mj] - self.default_angles_lab) * self.dof_pos_scale_
        dqj = dq_mj[self.lab2mj] * self.dof_vel_scale_
        ang_vel = self.robot_data_.get_angular_velocity() * self.ang_vel_scale_
        proj_gravity = self.robot_data_.get_project_gravity() * self.proj_gravity_scale_

        proprio = np.concatenate([
            np.array([np.sin(phase), np.cos(phase)], dtype=np.float32),
            command,
            qj.astype(np.float32),
            dqj.astype(np.float32),
            self.last_actions_,
            ang_vel.astype(np.float32),
            proj_gravity.astype(np.float32),
        ])

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
            print(f"[FSMStateSTAND] ONNX Runtime inference error: {e}")

    def run(self, flag: ControlFlag):
        """运行STAND状态"""
        if int(self.robot_data_.time_now_ / self.dt_) % self.decimation_ == 0:
            obs_hist = self.compute_observation(flag)
            self.compute_actions(obs_hist)

            target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab
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
        print("[FSMStateSTAND] exit")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        elif flag.fsm_state_command == "gotoWALKDEPTH":
            return FSMStateName.WALKDEPTH
        else:
            return None
