"""Shared runtime module for WalkLion policy-series FSM states.

WalkLion and WalkLionDWAQ share the same deployment shell:

1. Load policy config and ONNX model.
2. Validate policy-order and motor-order contracts.
3. Build 29-motor default targets and PD gains.
4. Run decimated inference inside the FSM control loop.
5. Scatter compact 15-DOF policy actions back into RobotData motor order.

Concrete policy modules keep only the parts that intentionally differ: gait
terms, observation layout, ONNX input shape, and action post-processing.  This
keeps policy contracts local while avoiding duplicate motor-command plumbing.
"""

import os

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation

from common.control_timing import TickDecimator
from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData


GRAVITY_VECTOR = np.array([0.0, 0.0, -1.0], dtype=np.float32)


class WalkLionPolicyBase(FSMState):
    """Common runtime flow for WalkLion-series ONNX policy states.

    The public FSM interface stays the same as every other state:
    `on_enter`, `run`, `on_exit`, and `check_transition`.  Subclasses provide a
    narrow policy-specific interface through `_init_policy_specific_config`,
    `_init_policy_buffers`, `_validate_observation_width`, `compute_observation`,
    and `compute_actions`.

    Three coordinate spaces are kept explicit:
    - policy/action order: compact 15-DOF ONNX order;
    - motor/XML order: 29 motors used by BodyServoIdMap and RobotData tails;
    - RobotData vectors: 35-wide arrays with 6 floating-base slots first.
    """

    def __init__(
        self,
        robot_data: RobotData,
        *,
        state_label: str,
        current_dir: str,
        config_filename: str,
    ):
        super().__init__(robot_data)
        self.state_label = state_label

        # Load one policy-local YAML file.  Shared runtime reads common keys,
        # then delegates gait/buffer setup to the concrete policy module.
        config_path = os.path.join(current_dir, "config", config_filename)
        with open(config_path, "r") as f:
            self.policy_config = yaml.safe_load(f)

        # Basic policy contract.  Any drift here means the ONNX graph and
        # deployment config are no longer describing the same actor.
        self.action_num_ = int(self.policy_config["actions_size"])
        self.motor_num_ = int(self.policy_config["motor_num"])
        self.dt_ = float(self.policy_config["dt"])
        self.debug_ = bool(self.policy_config.get("debug", False))

        # History width is policy-specific, but both policies express it as
        # `obs_size_ * num_hist_` before feeding onnxruntime.
        size_config = self.policy_config["size"]
        self.num_hist_ = int(size_config["num_hist"])
        self.obs_size_ = int(size_config["observations_size"])

        # Decimation lets the controller keep a 100 Hz command loop while the
        # learned policy may infer at a lower cadence.
        control_config = self.policy_config["control"]
        self.decimation_ = control_config.get("decimation", 1)
        self._inference_clock = TickDecimator(
            self.decimation_, self.dt_, self.robot_data_.control_dt_)
        self.warm_start_time_ = float(
            self.policy_config.get("warm_start_time", control_config.get("warm_start_time", 0.0))
        )
        self.action_scale_lab = self._as_vector(
            control_config["action_scale"], self.action_num_, "control.action_scale"
        )

        norm_config = self.policy_config.get("normalization", {})
        clip_config = norm_config.get("clip_scales", {})
        obs_config = norm_config.get("obs_scales", {})
        self.clip_obs_ = float(clip_config.get("clip_observations", 100.0))
        self.clip_act_ = float(clip_config.get("clip_actions", 100.0))
        self.ang_vel_scale_ = float(obs_config.get("ang_vel", 1.0))
        self.command_scale_ = float(obs_config.get("commands", 1.0))
        self.dof_pos_scale_ = float(obs_config.get("dof_pos", 1.0))
        self.dof_vel_scale_ = float(obs_config.get("dof_vel", 1.0))
        self.action_obs_scale_ = float(obs_config.get("actions", 1.0))

        # `joint_seq` is the compact ONNX order. `joint_xml` is the 29-motor
        # publishing order.  The scatter map `lab2mj` is derived, not hand-coded.
        self.joint_seq = list(self.policy_config["joint_names"])
        self.joint_xml = list(self.policy_config["joint_xml"])
        if len(self.joint_seq) != self.action_num_:
            raise ValueError(
                f"{self._prefix()} actions_size={self.action_num_} "
                f"does not match joint_names length={len(self.joint_seq)}"
            )
        if len(self.joint_xml) != self.motor_num_:
            raise ValueError(
                f"{self._prefix()} motor_num={self.motor_num_} "
                f"does not match joint_xml length={len(self.joint_xml)}"
            )

        self.lab2mj = self._resolve_policy_to_motor_indices()

        # Policy defaults and gains are configured in policy/action order, then
        # written into 29-wide motor-order arrays below.
        self.default_angles_lab = self._as_vector(
            self.policy_config["init_state"]["default_joint_angles"],
            self.action_num_,
            "init_state.default_joint_angles",
        )
        gains_config = self.policy_config["gains"]
        self.kps_lab = self._as_vector(gains_config["kp"], self.action_num_, "gains.kp")
        self.kds_lab = self._as_vector(gains_config["kd"], self.action_num_, "gains.kd")

        # Passive targets keep non-policy motors, especially arms and wrists,
        # actively held even when hidden-arm actors do not observe those joints.
        passive_cfg = self.policy_config.get("passive_joint_targets", {})
        self.default_motor_pos = self._build_motor_vector(passive_cfg, "angle", default=0.0)
        self.stiffness_array = self._build_motor_vector(passive_cfg, "kp", default=0.0)
        self.damping_array = self._build_motor_vector(passive_cfg, "kd", default=0.0)
        self.default_motor_pos[self.lab2mj] = self.default_angles_lab
        self.stiffness_array[self.lab2mj] = self.kps_lab
        self.damping_array[self.lab2mj] = self.kds_lab

        # Policy-local setup comes after shared arrays so subclasses can size
        # their buffers against validated config dimensions.
        self._init_policy_specific_config(self.policy_config.get("gait", {}))
        self._init_policy_buffers()

        # Warm start blends from measured entry pose to first policy target,
        # reducing discontinuity at the moment the policy state takes ownership.
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)
        self._warm_start_steps = self._compute_warm_start_steps()
        self._warmup_inference_counter = 0
        self.timer_gait_ = 0.0
        self.is_first_obs_ = True
        self.is_first_action_ = True

        self.model_path = os.path.join(current_dir, "model", self.policy_config["model_path"])
        self._init_onnx_session()
        self._validate_observation_width()

    def _prefix(self) -> str:
        return f"[FSMState{self.state_label}]"

    def _onnx_prefix(self) -> str:
        return f"[FSMState{self.state_label}-ONNX]"

    def _as_vector(self, value, expected_len: int, label: str) -> np.ndarray:
        """Return a float32 vector and fail fast on scalar/list length drift.

        YAML values such as `action_scale` may be a scalar or a per-joint list.
        Normalizing them here keeps all downstream action math vector-based.
        """
        if np.isscalar(value):
            vector = np.full(expected_len, float(value), dtype=np.float32)
        else:
            vector = np.array(value, dtype=np.float32)
        if vector.shape != (expected_len,):
            raise ValueError(f"{self._prefix()} {label} length {len(vector)} != {expected_len}")
        return vector

    def _build_motor_vector(self, passive_cfg: dict, field: str, default: float) -> np.ndarray:
        """Build one 29-wide motor-order vector from passive joint config.

        Missing entries intentionally stay at `default`; policy-controlled
        entries are overwritten from compact policy-order arrays after this.
        """
        values = np.full(self.motor_num_, default, dtype=np.float32)
        for joint_name, joint_cfg in passive_cfg.items():
            if joint_name not in self.joint_xml:
                raise ValueError(f"{self._prefix()} passive joint not in joint_xml: {joint_name}")
            values[self.joint_xml.index(joint_name)] = float(joint_cfg.get(field, default))
        return values

    def _resolve_policy_to_motor_indices(self) -> np.ndarray:
        """Resolve compact ONNX joint order into RobotData motor order."""
        missing = [name for name in self.joint_seq if name not in self.joint_xml]
        if missing:
            raise ValueError(f"{self._prefix()} policy joints missing from joint_xml: {missing}")
        return np.array([self.joint_xml.index(name) for name in self.joint_seq], dtype=np.int64)

    def _compute_warm_start_steps(self) -> int:
        """Convert configured warm-start seconds into inference steps."""
        step = self.decimation_ * self.dt_
        if self.warm_start_time_ <= 0.0 or step <= 0.0:
            return 0
        return max(1, int(self.warm_start_time_ / step))

    def _init_onnx_session(self) -> None:
        """Create a CPU onnxruntime session with deterministic settings.

        Inference runs inside a real-time-ish control loop, so thread counts are
        fixed to avoid onnxruntime spawning variable worker pools between runs.
        """
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.enable_mem_pattern = False
            options.enable_mem_reuse = True
            self.ort_session_ = ort.InferenceSession(
                self.model_path, options, providers=["CPUExecutionProvider"]
            )
            print(f"{self._onnx_prefix()} loaded: {self.model_path}")
        except Exception as exc:
            print(f"{self._prefix()} failed to load ONNX model: {exc}")
            self.ort_session_ = None

    def _reset_internal_state(self) -> None:
        """Reset inference state and write safe targets to RobotData.

        On state entry, every motor receives a deterministic target before the
        first inference tick.  This prevents passive motors from going limp and
        gives the command publisher valid PD gains immediately.
        """
        self._reset_policy_buffers()
        self.is_first_obs_ = True
        self.is_first_action_ = True

        base = self.robot_data_.q_d_s_.shape[0] - self.motor_num_
        self.robot_data_.q_d_s_[base : base + self.motor_num_] = self.default_motor_pos
        self.robot_data_.q_dot_d_s_[base : base + self.motor_num_] = 0.0
        self.robot_data_.tau_d_s_[base : base + self.motor_num_] = 0.0
        self._write_pd_gains()

    def _write_pd_gains(self) -> None:
        """Write the current 29-wide PD vectors in motor order."""
        self.robot_data_.joint_kp_s_[: self.motor_num_] = self.stiffness_array
        self.robot_data_.joint_kd_s_[: self.motor_num_] = self.damping_array

    def on_enter(self):
        """Enter policy state and cache current pose for optional warm start."""
        self._inference_clock.reset()
        self._reset_internal_state()
        self._warmup_inference_counter = 0
        self.timer_gait_ = 0.0
        try:
            self._warm_start_pose = self.robot_data_.get_joint_pos().copy()
        except Exception:
            self._warm_start_pose = self.default_motor_pos.copy()
        print(f"{self._prefix()} enter")

    def run(self, flag: ControlFlag):
        """Run one control-loop tick and write desired motor state.

        Subclasses only compute observation/action.  Applying actions to the
        29-motor command tail stays centralized so WalkLion variants cannot
        silently diverge in scatter, warm-start, or passive-hold behavior.
        """
        if self._inference_clock.step():
            self.compute_observation()
            self.compute_actions()

            # Start from passive holds, then overwrite policy-controlled motors
            # with default pose plus scaled ONNX action in motor/XML order.
            target_motor_pos = self.default_motor_pos.copy()
            target_motor_pos[self.lab2mj] = self.default_angles_lab + self.actions_ * self.action_scale_lab
            commanded_pos = self._blend_warm_start(target_motor_pos)

            # RobotData command arrays are 35-wide; the last 29 entries are
            # motors.  Velocity/torque targets stay zero for PD position mode.
            base = self.robot_data_.q_d_s_.shape[0] - self.motor_num_
            self.robot_data_.q_d_s_[base : base + self.motor_num_] = commanded_pos
            self.robot_data_.q_dot_d_s_[base : base + self.motor_num_] = 0.0
            self.robot_data_.tau_d_s_[base : base + self.motor_num_] = 0.0
            self.last_actions_[:] = self.actions_

        self.timer_gait_ += self.dt_
        self._write_pd_gains()

    def _blend_warm_start(self, target_motor_pos: np.ndarray) -> np.ndarray:
        """Interpolate from measured entry pose to first policy target if enabled."""
        if self._warm_start_steps <= 0 or self._warmup_inference_counter >= self._warm_start_steps:
            return target_motor_pos
        self._warmup_inference_counter += 1
        blend = self._warmup_inference_counter / float(self._warm_start_steps)
        return (1.0 - blend) * self._warm_start_pose + blend * target_motor_pos

    @staticmethod
    def euler_to_quaternion_scipy(roll, pitch, yaw, degrees=False):
        """Convert IMU RPY to wxyz quaternion."""
        quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    @staticmethod
    def quat_rotate_inverse_numpy(q_xyzw, v):
        """Rotate vector by inverse quaternion, using xyzw quaternion order."""
        q_w = q_xyzw[3]
        q_v = q_xyzw[:3]
        return (
            v * (2.0 * q_w * q_w - 1.0)
            - np.cross(q_v, v) * (2.0 * q_w)
            + q_v * (2.0 * np.dot(q_v, v))
        )

    def _validate_modelmetadata(self) -> None:
        """Validate ONNX model metadata for expected input/output names and shapes."""
        if self.ort_session_ is None:
            return
        try:
            metadata = self.ort_session_.get_modelmeta().custom_metadata_map
            joint_seq = metadata["joint_names"].split(",")
            joint_pos_array_seq = np.array([float(x) for x in metadata["default_joint_pos"].split(",")])
            stiffness_array_seq = np.array([float(x) for x in metadata["joint_stiffness"].split(",")])
            damping_array_seq = np.array([float(x) for x in metadata["joint_damping"].split(",")])
            action_scale = np.array([float(x) for x in metadata["action_scale"].split(",")])
            print(f"{self._onnx_prefix()} model metadata:")
            print(f"  joint_names: {joint_seq}")
            print(f"  default_joint_pos: {joint_pos_array_seq}")
            print(f"  joint_stiffness: {stiffness_array_seq}")
            print(f"  joint_damping: {damping_array_seq}")
            print(f"  action_scale: {action_scale}")
            if joint_seq != self.joint_seq:
                raise ValueError(f"ONNX joint_names {joint_seq} != config {self.joint_seq}")
            if joint_pos_array_seq.shape != (self.action_num_,):
                raise ValueError(
                    f"ONNX default_joint_pos shape {joint_pos_array_seq.shape} != ({self.action_num_},)"
                )
            if stiffness_array_seq.shape != (self.action_num_,):
                raise ValueError(
                    f"ONNX joint_stiffness shape {stiffness_array_seq.shape} != ({self.action_num_},)"
                )
            if damping_array_seq.shape != (self.action_num_,):
                raise ValueError(
                    f"ONNX joint_damping shape {damping_array_seq.shape} != ({self.action_num_},)"
                )
            if action_scale.shape != (self.action_num_,):
                raise ValueError(f"ONNX action_scale shape {action_scale.shape} != ({self.action_num_},)")
        except Exception as exc:
            print(f"{self._prefix()} ONNX model metadata validation error: {exc}")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """Map control flag command strings to WalkLion-series FSM states.

        Entry into WALKLION/WALKLIONDWAQ is still gated by surrounding FSM state
        transition rules.  This common mapping only defines what this policy
        state accepts once the FSM has already reached it.
        """
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        if flag.fsm_state_command == "gotoWALKLION":
            return FSMStateName.WALKLION
        if flag.fsm_state_command == "gotoLIONZERO":
            return FSMStateName.LIONZERO
        if flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        if flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        if flag.fsm_state_command == "gotoPOLICYZERO":
            return FSMStateName.POLICYZERO
        if flag.fsm_state_command == "gotoBEYONDMIMIC":
            return FSMStateName.BEYONDMIMIC
        if flag.fsm_state_command == "gotoWALKLIONDWAQ":
            return FSMStateName.WALKLIONDWAQ
        return None

    def _init_policy_specific_config(self, gait_config: dict) -> None:
        raise NotImplementedError

    def _init_policy_buffers(self) -> None:
        raise NotImplementedError

    def _reset_policy_buffers(self) -> None:
        raise NotImplementedError

    def _validate_observation_width(self) -> None:
        raise NotImplementedError

    def compute_observation(self):
        raise NotImplementedError

    def compute_actions(self):
        raise NotImplementedError
