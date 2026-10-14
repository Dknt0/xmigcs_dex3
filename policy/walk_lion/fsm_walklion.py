"""WalkLion FSM state for Dex3 lower+waist hidden-arm deployment.

The shared WalkLion-series runtime owns config loading, ONNX setup, action
application, warm start, and FSM transition mapping.  This module keeps the
TienKung-Lab WalkLion observation layout and gait terms local.
"""

import os

import numpy as np

from common.robot_data import RobotData
from policy.walk_lion_series import GRAVITY_VECTOR, WalkLionPolicyBase


class FSMStateWalkLion(WalkLionPolicyBase):
    """Run the exported TienKung-Lab Dex3 lower+waist ONNX policy."""

    def __init__(self, robot_data: RobotData):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(
            robot_data,
            state_label="WALKLION",
            current_dir=current_dir,
            config_filename="walk_lion.yaml",
        )

    def _init_policy_specific_config(self, gait_config: dict) -> None:
        """Load TienKung-Lab two-foot phase parameters from YAML."""
        self.gait_cycle = float(gait_config.get("cycle", 0.85))
        self.phase_offsets = np.array(
            [
                gait_config.get("left_phase_offset", 0.38),
                gait_config.get("right_phase_offset", 0.88),
            ],
            dtype=np.float32,
        )
        self.phase_ratio = np.array(
            [
                gait_config.get("left_air_ratio", 0.38),
                gait_config.get("right_air_ratio", 0.38),
            ],
            dtype=np.float32,
        )

    def _init_policy_buffers(self) -> None:
        """Allocate the single-input ONNX history buffer.

        WalkLion consumes one flattened `[1, 600]` input: ten 60-wide frames.
        The running buffer is stored flat because onnxruntime expects that shape.
        """
        self.observations_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros_like(self.observations_)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)

    def _reset_policy_buffers(self) -> None:
        """Clear WalkLion observation history and previous action state."""
        self.observations_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)

    def _validate_observation_width(self) -> None:
        """Validate config dimensions against the ONNX graph before runtime use."""
        # Per-frame layout:
        #   ang_vel(3), projected_gravity(3), command(3),
        #   q_error(15), qd(15), previous_action(15), gait(6)
        gait_width = 6
        frame_width = 3 + 3 + 3 + self.action_num_ + self.action_num_ + self.action_num_ + gait_width
        if frame_width != self.obs_size_:
            raise ValueError(f"{self._prefix()} computed obs frame width {frame_width} != config {self.obs_size_}")
        if self.ort_session_ is None:
            return
        model_inputs = self.ort_session_.get_inputs()
        model_outputs = self.ort_session_.get_outputs()
        model_obs = model_inputs[0].shape[-1]
        model_actions = model_outputs[0].shape[-1]
        expected_obs = self.obs_size_ * self.num_hist_
        if model_obs != expected_obs:
            raise ValueError(f"{self._prefix()} ONNX input width {model_obs} != {expected_obs}")
        if model_actions != self.action_num_:
            raise ValueError(f"{self._prefix()} ONNX output width {model_actions} != {self.action_num_}")

    def _gait_terms(self) -> np.ndarray:
        """Return the six gait observation channels expected by the actor.

        Layout is `[sin(left/right), cos(left/right), phase_ratio(left/right)]`.
        The two feet have different phase offsets but share the same gait clock.
        """
        phase = ((self.timer_gait_ / self.gait_cycle) + self.phase_offsets) % 1.0
        return np.concatenate(
            [
                np.sin(2.0 * np.pi * phase),
                np.cos(2.0 * np.pi * phase),
                self.phase_ratio,
            ]
        ).astype(np.float32)

    def compute_observation(self):
        """Build and roll the 600-wide actor observation buffer.

        Per-frame layout, in ONNX training order:
        `ang_vel(3), projected_gravity(3), command(3), q_error(15),
        qd(15), previous_action(15), gait(6)`.
        """
        roll, pitch, yaw = (
            float(self.robot_data_.imu_data_[2]),
            float(self.robot_data_.imu_data_[1]),
            float(self.robot_data_.imu_data_[0]),
        )
        quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        projected_gravity = self.quat_rotate_inverse_numpy(quat_xyzw, GRAVITY_VECTOR)

        command = np.array(self.robot_data_.get_walk_cmd(), dtype=np.float32) * self.command_scale_
        motor_pos = self.robot_data_.get_joint_pos()
        motor_vel = self.robot_data_.get_joint_vel()
        policy_pos = (motor_pos[self.lab2mj] - self.default_angles_lab) * self.dof_pos_scale_
        policy_vel = motor_vel[self.lab2mj] * self.dof_vel_scale_
        ang_vel = self.robot_data_.get_angular_velocity() * self.ang_vel_scale_

        # Keep this layout in lockstep with the TienKung-Lab export contract.
        # Joint terms use policy/action order; motor data is gathered via lab2mj.
        proprio = np.concatenate(
            [
                ang_vel,
                projected_gravity,
                command,
                policy_pos,
                policy_vel,
                self.last_actions_ * self.action_obs_scale_,
                self._gait_terms(),
            ]
        ).astype(np.float32)
        if proprio.shape != (self.obs_size_,):
            raise ValueError(f"{self._prefix()} observation frame shape {proprio.shape} != ({self.obs_size_},)")

        if self.is_first_obs_:
            # Match RSL-RL export behavior: fill every history slot with the
            # first real frame instead of starting with zero history.
            for i in range(self.num_hist_):
                start = i * self.obs_size_
                self.proprio_hist_buf_[start : start + self.obs_size_] = proprio
            self.is_first_obs_ = False
        else:
            # FIFO one frame left, newest frame appended at the tail.
            shift_size = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift_size] = self.proprio_hist_buf_[self.obs_size_ :]
            self.proprio_hist_buf_[shift_size:] = proprio

        self.observations_ = np.clip(self.proprio_hist_buf_, -self.clip_obs_, self.clip_obs_)
        if self.debug_:
            print(f"{self._prefix()} command={command}")

    def compute_actions(self):
        """Run ONNX inference and clip output actions in policy order."""
        if self.ort_session_ is None:
            return
        try:
            input_data = self.observations_.reshape(1, -1).astype(np.float32)
            input_name = self.ort_session_.get_inputs()[0].name
            outputs = self.ort_session_.run(None, {input_name: input_data})
            output_data = np.asarray(outputs[0][0], dtype=np.float32)
            if output_data.shape != (self.action_num_,):
                raise ValueError(f"action shape {output_data.shape} != ({self.action_num_},)")
            self.actions_[:] = np.clip(output_data, -self.clip_act_, self.clip_act_)
            if self.is_first_action_:
                print(f"{self._onnx_prefix()} first inference ok")
                self.is_first_action_ = False
        except Exception as exc:
            print(f"{self._prefix()} ONNX Runtime inference error: {exc}")

    def on_exit(self):
        """Leave WALKLION state."""
        print(f"{self._prefix()} exit")
