"""
FSM State Implementations
Concrete implementations of different FSM states
"""

import numpy as np
import onnxruntime as ort

from common.control_timing import TickDecimator
from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData
import os
import yaml
from scipy.spatial.transform import Rotation

class FSMStateWALKAMP(FSMState):
    """WALKAMP策略状态实现"""

    def _reset_internal_state(self):
        """把所有随时间变化的内部状态重置成初始值"""

        # 1) 清空 obs / hist / actions
        self.observations_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)

        # 2) 标志位重置
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self.is_first_step_ = True

        # 3) 期望关节 / 期望速度 / 力矩重置为“初始姿态”
        # 你已经有 self.joint_pos_array（mj 顺序，长度 len(self.joint_xml)）
        base = self.robot_data_.q_d_s_.shape[0] - self.motor_num_
        # 期望角 = 初始角
        self.robot_data_.q_d_s_[base:base + len(self.joint_xml)] = self.joint_pos_array
        # 期望速度 = 0
        self.robot_data_.q_dot_d_s_[base:base + len(self.joint_xml)] = 0.0
        # 期望力矩 = 0（位置控制）
        self.robot_data_.tau_d_s_[base:base + len(self.joint_xml)] = 0.0
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "walk_amp.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        # Load configuration exactly like C++
        self.action_num_ = policy_config.get('actions_size')
        self.motor_num_ = policy_config.get('motor_num')
        self.dt_ = policy_config.get('dt')

        # Size configuration
        size_config = policy_config.get('size', {})
        self.num_hist_ = size_config.get('num_hist')
        self.obs_size_ = size_config.get('observations_size')

        # Control configuration
        control_config = policy_config.get('control', {})
        self.action_scale_ = control_config.get('action_scale')
        # self.gait_cycle_period_ = control_config.get('gait_cycle_period', 1.0)
        self.decimation_ = control_config.get('decimation')
        self._inference_clock = TickDecimator(
            self.decimation_, self.dt_, self.robot_data_.control_dt_)
        self.warm_start_time_ = control_config.get('warm_start_time', 0.3)

        # Normalization configuration
        norm_config = policy_config.get('normalization', {})
        clip_config = norm_config.get('clip_scales', {})
        obs_config = norm_config.get('obs_scales', {})

        self.clip_obs_ = clip_config.get('clip_observations', 100.0)
        self.clip_act_ = clip_config.get('clip_actions', 100.0)
        self.lin_vel_scale_ = obs_config.get('lin_vel')
        self.ang_vel_scale_ = obs_config.get('ang_vel')
        self.dof_pos_scale_ = obs_config.get('dof_pos')
        self.dof_vel_scale_ = obs_config.get('dof_vel')


        # Initialize buffers and actions
        self.observations_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)


        # Flags matching C++
        self.is_first_obs_ = True
        self.is_first_action_ = True
        # self.phase_locked = False
        self.is_first_step_ = True
        step = (self.decimation_ if self.decimation_ else 1) * self.dt_
        if self.warm_start_time_ > 0 and step > 0:
            self._warm_start_steps = max(1, int(self.warm_start_time_ / step))
        else:
            self._warm_start_steps = 0
        self._warmup_inference_counter = 0


        # Initialize ONNX session
        self.model_path = os.path.join(current_dir, "model", policy_config["model_path"]) 
        self._init_onnx_session()

        # ====== 从 ONNX metadata 里读 lab 侧关节信息 ======
        # self.onnx_model = onnx.load(self.model_path)
        # self.onnx_model = onnxruntime.InferenceSession(self.model_path)
        meta = self.ort_session_.get_modelmeta().custom_metadata_map

        # 从ONNX模型中读取参数
        self.joint_seq = None
        self.joint_pos_array_seq = None
        self.action_scale = None
        self.stiffness_array_seq = None
        self.damping_array_seq = None
        
        self.joint_seq = meta["joint_names"].split(",")
        self.joint_pos_array_seq = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
        self.stiffness_array_seq = np.array([float(x) for x in meta["joint_stiffness"].split(",")])
        self.damping_array_seq = np.array([float(x) for x in meta["joint_damping"].split(",")])
        self.action_scale = np.array([float(x) for x in meta["action_scale"].split(",")])
        # for prop in self.onnx_model.metadata_props:
        #     if prop.key == "joint_names":
        #         self.joint_seq = prop.value.split(",")
        #     if prop.key == "default_joint_pos":   
        #         self.joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
        #     if prop.key == "joint_stiffness":
        #         self.stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
        #     if prop.key == "joint_damping":
        #         self.damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
        #     if prop.key == "action_scale":
        #         self.action_scale = np.array([float(x) for x in prop.value.split(",")])
        
        # # 设置从序列到实验室顺序的映射
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

        # 从MjXUML顺序映射到实验室顺序
        # self.mj2lab = np.array([self.joint_xml.index(joint) for joint in self.joint_seq])
        self.lab2mj = []
        for name in self.joint_seq:
            if name not in self.joint_xml:
                raise ValueError(f"[FSMStateWALKAMP] joint '{name}' from ONNX not found in joint_xml!")
            self.lab2mj.append(self.joint_xml.index(name))
        self.lab2mj = np.array(self.lab2mj, dtype=int)

        # 从实验室顺序映射到MjXUML顺序
        # self.joint_pos_array = np.array([self.joint_pos_array_seq[self.joint_seq.index(joint)] for joint in self.joint_xml])
        # self.stiffness_array = np.array([self.stiffness_array_seq[self.joint_seq.index(joint)] for joint in self.joint_xml])
        # self.damping_array = np.array([self.damping_array_seq[self.joint_seq.index(joint)] for joint in self.joint_xml])
                # ====== 把 23 个 lab 关节 scatter 到 29 个 xml 里，多的 6 个保持默认 ======
        n_mj = len(self.joint_xml)

        # 29 长度，mujoco XML 顺序，先全 0 或者你想要的默认值
        self.joint_pos_array = np.zeros(n_mj, dtype=np.float32)
        self.stiffness_array = np.zeros(n_mj, dtype=np.float32)
        self.damping_array = np.zeros(n_mj, dtype=np.float32)

        # joint_pos_array_seq / stiffness_array_seq / damping_array_seq 是 23 长度，lab 顺序
        for lab_idx, mj_idx in enumerate(self.lab2mj):
            self.joint_pos_array[mj_idx] = self.joint_pos_array_seq[lab_idx]
            self.stiffness_array[mj_idx] = self.stiffness_array_seq[lab_idx]
            self.damping_array[mj_idx] = self.damping_array_seq[lab_idx]


        # 设置其他参数
        self.kps_lab = self.stiffness_array_seq
        self.kds_lab = self.damping_array_seq
        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale


        self.filtered_x_speed = 0
        # n_mj = len(self.joint_xml)
        # self.kp_array = np.zeros(n_mj, dtype=np.float32)
        # self.kd_array = np.zeros(n_mj, dtype=np.float32)

        # for lab_idx, mj_idx in enumerate(self.lab2mj):
        #     self.kp_array[mj_idx] = self.kps_lab[lab_idx]
        #     self.kd_array[mj_idx] = self.kds_lab[lab_idx]

    def _init_onnx_session(self):
        """初始化ONNX推理会话"""
        try:
            # 配置SessionOptions
            options = ort.SessionOptions()

            # 启用图优化，使用所有可用的优化（包括算子融合等）
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # 设置执行模式（可选，默认执行模式是顺序执行，但图优化会改变计算图）
            # 设置线程数（根据CPU核心数调整）
            # 建议设置为CPU物理核心数（非超线程数），因为超线程可能不会带来线性提升
            options.intra_op_num_threads = 1  # 设置计算图中的运算符内部并行线程数
            options.inter_op_num_threads = 1  # 设置多个运算符之间的并行线程数（如果模型有多个分支）

            # 启用内存优化（避免重复分配内存）
            options.enable_mem_pattern = False  # 对于固定输入大小，可以设为False以避免内存规划的开销
            options.enable_mem_reuse = True # 启用内存重用机制

            self.ort_session_ = ort.InferenceSession(self.model_path, options, providers=['CPUExecutionProvider'])
            
            print(f"[FSMStateWALKAMP-ONNX] ONNX model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"[FSMStateWALKAMP] Failed to load ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}")
            self.ort_session_ = None

    def on_enter(self):
        """进入WALKAMP状态"""
        self._inference_clock.reset()
        self._reset_internal_state()
        print("[FSMStateWALKAMP] enter")
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self._warmup_inference_counter = 0
        if self.robot_data_ is not None:
            try:
                self._warm_start_pose = self.robot_data_.get_joint_pos().copy()
            except Exception:
                self._warm_start_pose.fill(0.0)
        else:
            self._warm_start_pose.fill(0.0)

    def run(self, flag: ControlFlag):
        """运行WALKAMP状态 - 与C++版本完全一致"""
        # print("[FSMStateWALKAMP] run")
        # Only run policy inference every decimation_ steps
        if self._inference_clock.step():
            self.compute_observation(flag)
            self.compute_actions()


                    # ====== 临时测试：强制动作为 0 ======
            # self.actions_.fill(0.0)
            # q_des = self.robot_data_.get_joint_pos()
            # q_des = q_des[self.mj2lab]
            # q_des = (q_des - self.default_angles_lab)

            # lab 顺序目标角 23 维
            target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab

            # 拿一份当前 mj 顺序的关节角（或你原来用的 default 也行）
            target_dof_pos_mj = self.robot_data_.get_joint_pos().copy()

            # 只更新 23 个受控 DOF
            target_dof_pos_mj[self.lab2mj] = target_dof_pos_lab
            # target_dof_pos_mj = np.zeros(29)
            # target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab
            # target_dof_pos_mj[self.mj2lab] = target_dof_pos_lab
            commanded_pos = target_dof_pos_mj
            if self._warm_start_steps > 0 and self._warmup_inference_counter < self._warm_start_steps:
                self._warmup_inference_counter += 1
                blend = self._warmup_inference_counter / float(self._warm_start_steps)
                commanded_pos = (1.0 - blend) * self._warm_start_pose + blend * target_dof_pos_mj

            base = self.robot_data_.q_d_s_.shape[0] - self.motor_num_
            self.robot_data_.q_d_s_[base:base + len(self.joint_xml)] = commanded_pos

            self.robot_data_.q_dot_d_s_[base:base + len(self.joint_xml)] = 0.0
            self.robot_data_.tau_d_s_[base:base + len(self.joint_xml)] = 0.0
            # q_des_lab = self.default_angles_lab + self.actions_ * self.action_scale_lab
            # qdot_des_lab = np.zeros_like(q_des_lab, dtype=np.float32)

            self.last_actions_[:] = self.actions_

            # q_des_mj = np.zeros_like(q_des_lab, dtype=np.float32)
            # qdot_des_mj = np.zeros_like(qdot_des_lab, dtype=np.float32)

            # 把 lab 顺序的期望角，按 lab2mj 映射到 mj 顺序里面
            # q_des_mj[self.lab2mj] = q_des_lab
            # qdot_des_mj[self.lab2mj] = qdot_des_lab
            # q_des_mj[self.lab2mj] = q_des_lab[self.lab2mj]
            # qdot_des_mj[self.lab2mj] = qdot_des_lab[self.lab2mj]

            # robot_data.q_d_s_.tail(motor_num_) = q_des_mj
            # base = self.robot_data_.q_a_s_.shape[0] - self.motor_num_  # whole_joint_num - motor_num
            # self.robot_data_.q_d_s_[base:] = q_des_mj
            # self.robot_data_.q_dot_d_s_[base:] = qdot_des_mj
            # self.robot_data_.tau_d_s_[base:] = 0.0  # 位置控制，力矩留给 PD 算
            

        # ====== 3) 写 PD 增益（mj 顺序） ======
        # base = self.robot_data_.joint_kp_s_.shape[0] - self.motor_num_
        # self.robot_data_.joint_kp_s_[self.mj2lab][:self.action_num_] = self.kps_lab
        # self.robot_data_.joint_kd_s_[self.mj2lab][:self.action_num_] = self.kds_lab
        # ====== 3) 写 PD 增益（mj 顺序，29DOF） ======
        # base = self.robot_data_.joint_kp_s_.shape[0] - self.motor_num_
        self.robot_data_.joint_kp_s_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_s_[:len(self.joint_xml)] = self.damping_array
        self.robot_data_.joint_kp_s_[27] = 80
        self.robot_data_.joint_kd_s_[27] = 5
        self.robot_data_.joint_kp_s_[26] = 80
        self.robot_data_.joint_kd_s_[26] = 5
        self.robot_data_.joint_kp_s_[28] = 80
        self.robot_data_.joint_kd_s_[28] = 5
        self.robot_data_.q_d_s_[33] = 0
        self.robot_data_.q_d_s_[34] = 0
        self.robot_data_.q_d_s_[32] = 0

        self.robot_data_.joint_kp_s_[21] = 80
        self.robot_data_.joint_kd_s_[21] = 5
        self.robot_data_.joint_kp_s_[20] = 80
        self.robot_data_.joint_kd_s_[20] = 5
        self.robot_data_.joint_kp_s_[19] = 80
        self.robot_data_.joint_kd_s_[19] = 5
        self.robot_data_.q_d_s_[27] = 0
        self.robot_data_.q_d_s_[26] = 0
        self.robot_data_.q_d_s_[25] = 0








    def compute_observation(self, flag: ControlFlag):
        #         # ====== Save q_a_s_ / q_dot_a_s_ raw joint order to txt (run only once) ======
        # if not hasattr(self, "_saved_joint_order"):
        #     self._saved_joint_order = True

        #     save_path = "/home/eric/q_a_joint_order.txt"

        #     with open(save_path, "w") as f:
        #         f.write("q_a_s_ joint order (raw mj index order)\n")
        #         f.write("--------------------------------------\n")

        #         # q_a_s_ 里存的是 base(7) + joints(...)
        #         total = len(self.robot_data_.q_a_s_)
        #         f.write(f"Total q_a_s_ length = {total}\n\n")

        #         for i in range(total):
        #             # 找 XML 名字
        #             name = self.joint_xml[i] if i < len(self.joint_xml) else "BASE_or_unused"
        #             f.write(f"Idx {i:02d}: {name:25s}  q={self.robot_data_.q_a_s_[i]: .6f}   dq={self.robot_data_.q_dot_a_s_[i]: .6f}\n")

        #         f.write("\nControlled 23 joints mapping (for reference):\n")
        #         f.write("--------------------------------------\n")
        #         for j, mj_idx in enumerate(self.controlled_joint_indices_):
        #             name = self.joint_xml[mj_idx]
        #             f.write(f"WALKAMP {j:02d} -> MjIdx {mj_idx:02d} ({name})\n")

        """计算观测量 - 与C++版本完全一致"""

        # t_now = float(self.robot_data_.time_now_)

        # Phase calculation exactly like C++
        # phase = math.fmod(t_now, self.gait_cycle_period_)

        # cmd_norm = math.sqrt(
        #     flag.x_speed_command * flag.x_speed_command +
        #     flag.y_speed_command * flag.y_speed_command +
        #     flag.yaw_speed_command * flag.yaw_speed_command
        # )

        # if cmd_norm >= 0.05:
        #     self.phase_locked = False

        # tolerance = 0.1
        # if cmd_norm < 0.05 and abs(phase - self.gait_cycle_period_) < tolerance:
        #     self.phase_locked = True

        # if self.phase_locked:
        #     phase = 0

        roll, pitch, yaw = (
                        float(self.robot_data_.imu_data_[2]),
                        float(self.robot_data_.imu_data_[1]),
                        float(self.robot_data_.imu_data_[0]),
                    )
        quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
        q_xyzw    = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        gravity_init   = self.quat_rotate_inverse_numpy(q_xyzw, np.array([0.,0.,-1.], dtype=np.float32))
        

        x_speed_command, y_speed_command, yaw_speed_command = self.robot_data_.get_walk_cmd()
        new_filtered_x_speed = 1 * x_speed_command + (1 - 1) * self.filtered_x_speed
        change = new_filtered_x_speed - self.filtered_x_speed
        change = np.clip(change, -0.005, 0.005)
        self.filtered_x_speed = self.filtered_x_speed + change

        # Command vector exactly like C++
        command = np.array([
            # flag.x_speed_command,
            x_speed_command,
            # self.filtered_x_speed,
            y_speed_command,
            yaw_speed_command
        ], dtype=np.float32)
        # print(f'Input command: {command}')

        # IMU data exactly like C++
        # rpy = np.array([
        #     self.robot_data_.imu_data_[2],  # roll
        #     self.robot_data_.imu_data_[1],  # pitch
        #     self.robot_data_.imu_data_[0]   # yaw
        # ], dtype=np.float32) * 1.0

        gyro = np.array([
            self.robot_data_.imu_data_[3],
            self.robot_data_.imu_data_[4],
            self.robot_data_.imu_data_[5]
        ], dtype=np.float32) * self.ang_vel_scale_

        q_mj = self.robot_data_.get_joint_pos()
        qdot_mj = self.robot_data_.get_joint_vel()



        ang_vel = self.robot_data_.get_angular_velocity()
        q_mj = self.robot_data_.get_joint_pos()   # mj 顺序，长度 29
        dq_mj = self.robot_data_.get_joint_vel()

        # 只取 23 个受控关节，变成 lab 顺序
        qj = q_mj[self.lab2mj]
        dqj = dq_mj[self.lab2mj]

        qj = qj - self.default_angles_lab


        # Concatenate exactly like C++
        proprio = np.concatenate([
            ang_vel ,              # 3 elements
            gravity_init,
            command ,           # 5 elements
            qj,         # 12 elements
            dqj,         # 12 elements
            self.last_actions_, # 12 elements
        ])  # Total: 47 elements

        # History buffer management exactly like C++
        if self.is_first_obs_:
            for i in range(self.num_hist_):
                start_idx = i * self.obs_size_
                end_idx = start_idx + self.obs_size_
                self.proprio_hist_buf_[start_idx:end_idx] = proprio
            self.is_first_obs_ = False
        else:
            # Shift history: head((num_hist-1)*obs_size) = tail((num_hist-1)*obs_size)
            shift_size = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift_size] = self.proprio_hist_buf_[self.obs_size_:]
            self.proprio_hist_buf_[shift_size:] = proprio

        # Clip observations exactly like C++
        self.observations_ = np.clip(self.proprio_hist_buf_, -self.clip_obs_, self.clip_obs_)


    @staticmethod
    def euler_to_quaternion_scipy(roll, pitch, yaw, degrees=False):
        r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
        q_xyzw = r.as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    @staticmethod
    def quat_rotate_inverse_numpy(q_xyzw, v):
        q_w = q_xyzw[3]
        q_v = q_xyzw[:3]
        a = v * (2.0 * q_w * q_w - 1.0)
        b = np.cross(q_v, v) * (2.0 * q_w)
        c = q_v * (2.0 * np.dot(q_v, v))
        return a - b + c
    def compute_actions(self):
        """使用ONNX模型计算动作 - 与C++版本完全一致"""
        if self.ort_session_ is None:
            return

        try:
            # Prepare input tensor
            input_data = self.observations_.reshape(1, -1).astype(np.float32)

            # ONNX inference
            input_name = self.ort_session_.get_inputs()[0].name
            outputs = self.ort_session_.run(None, {input_name: input_data})

            # Extract and clip actions exactly like C++
            output_data = outputs[0][0]
            for i in range(self.action_num_):
                self.actions_[i] = np.clip(output_data[i], -self.clip_act_, self.clip_act_)

            if self.is_first_action_:
                # print("[FSMStateWALKAMP-ONNX] First Observation:")
                # for i in range(self.obs_size_):
                #     print(f"{self.observations_[i]:.6f} ", end="")
                # print()
                self.is_first_action_ = False

        except Exception as e:
            print(f"[FSMStateWALKAMP] ONNX Runtime inference error: {e}")

    def on_exit(self):
        """退出WALKAMP状态"""
        print("[FSMStateWALKAMP] exit")
        # 关掉 obs 日志文件
        if getattr(self, "obs_log_file", None) is not None:
            try:
                self.obs_log_file.flush()
                self.obs_log_file.close()
                print(f"[FSMStateWALKAMP] obs log saved to {self.obs_log_path}")
            except Exception as e:
                print(f"[FSMStateWALKAMP] failed to close obs log: {e}")
            self.obs_log_file = None


        # if getattr(self, "log_file_", None) is not None:
        #     self.log_file_.flush()
        #     self.log_file_.close()
        #     self.log_file_ = None
        #     print(f"[FSMStateWALKAMP] log saved to {self.log_path_}")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoLIONZERO":
            return FSMStateName.LIONZERO
        elif flag.fsm_state_command == "gotoWALKDEPTH":
            return FSMStateName.WALKDEPTH
        elif flag.fsm_state_command == "gotoSTAND":
            return FSMStateName.STAND
        elif flag.fsm_state_command == "gotoWALKDEPTH100HZ":
            return FSMStateName.WALKDEPTH100HZ
        else:
            return None  # 无状态转换
