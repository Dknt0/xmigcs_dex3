"""WalkLionDWAQ FSM state for Dex3 lower+waist hidden-arm deployment.

The shared WalkLion-series runtime owns config loading, ONNX setup, action
application, warm start, and FSM transition mapping.  This module keeps the
LeggedLabUltra DWAQ observation layout, two-input history model, action filter,
and standing/walking gait terms local.
"""

import os

import numpy as np

from common.robot_data import RobotData
from policy.walk_lion_series import GRAVITY_VECTOR, WalkLionPolicyBase


class FSMStateWalkLionDwaq(WalkLionPolicyBase):
    """Run the exported LeggedLabUltra Dex3 lower+waist DWAQ ONNX policy."""

    def __init__(self, robot_data: RobotData):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(
            robot_data,
            state_label="WALKLIONDWAQ",
            current_dir=current_dir,
            config_filename="walk_lion_dwaq.yaml",
        )

    def _init_policy_specific_config(self, gait_config: dict) -> None:
        """Load DWAQ gait classifier and action-filter parameters."""
        self.gait_cycle = float(gait_config.get("cycle", 0.75))
        self.standing_phase = float(gait_config.get("standing_phase", 0.125))
        self.linear_speed_threshold = float(gait_config.get("linear_speed_threshold", 0.12))
        self.angular_speed_threshold = float(gait_config.get("angular_speed_threshold", 0.06))
        self.action_filter = float(gait_config.get("action_filter", 0.5))

    def _init_policy_buffers(self) -> None:
        """Allocate DWAQ two-input ONNX buffers.

        DWAQ consumes both current observation `[1, 58]` and history
        `[1, 4350]` (`75 * 58`).  The buffers stay separate to mirror exported
        ONNX input names and shapes.
        """
        self.single_frame_obs_ = np.zeros(self.obs_size_, dtype=np.float32)
        self.hist_obs_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros_like(self.hist_obs_)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.filtered_actions_ = np.zeros_like(self.actions_)

    def _reset_policy_buffers(self) -> None:
        """Clear DWAQ observation history, previous action, and action filter."""
        self.single_frame_obs_.fill(0.0)
        self.hist_obs_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)
        self.filtered_actions_.fill(0.0)

    def _validate_observation_width(self) -> None:
        """Validate config dimensions against the ONNX graph before runtime use."""
        # Per-frame layout:
        #   ang_vel(3), projected_gravity(3), command(3), gait(4)
        #   q_error(15), qd(15), previous_action(15)
        gait_width = 4
        frame_width = 3 + 3 + 3 + self.action_num_ + self.action_num_ + self.action_num_ + gait_width
        if frame_width != self.obs_size_:
            raise ValueError(f"{self._prefix()} computed obs frame width {frame_width} != config {self.obs_size_}")
        if self.ort_session_ is None:
            return
        model_inputs = self.ort_session_.get_inputs()
        model_outputs = self.ort_session_.get_outputs()
        model_obs = model_inputs[0].shape[-1]
        model_obs_hist = model_inputs[1].shape[-1]
        model_actions = model_outputs[0].shape[-1]
        expected_obs_hist = self.obs_size_ * self.num_hist_
        if model_obs != self.obs_size_:
            raise ValueError(f"{self._prefix()} ONNX input width {model_obs} != {self.obs_size_}")
        if model_obs_hist != expected_obs_hist:
            raise ValueError(f"{self._prefix()} ONNX input history width {model_obs_hist} != {expected_obs_hist}")
        if model_actions != self.action_num_:
            raise ValueError(f"{self._prefix()} ONNX output width {model_actions} != {self.action_num_}")

    def _is_standing(self) -> bool:
        """Return True when command speed is below DWAQ walking thresholds.

        The actor was trained with standing/walking one-hot gait terms.  Small
        joystick noise should not switch the gait phase into walking mode.
        """
        command = np.array(self.robot_data_.get_walk_cmd(), dtype=np.float32)
        linear_speed = np.linalg.norm(command[:2])
        angular_speed = abs(command[2])
        return linear_speed < self.linear_speed_threshold and angular_speed < self.angular_speed_threshold

    def _gait_terms(self) -> np.ndarray:
        """Return the four gait observation channels expected by the actor.

        Layout is `[sin(phase), cos(phase), standing_flag, walking_flag]`.
        Standing uses a fixed phase so zero commands do not continue advancing
        the gait clock inside the observation.
        """
        phase = (self.timer_gait_ / self.gait_cycle) % 1.0
        if self._is_standing():
            phase = self.standing_phase
            gait_state = (1.0, -1.0)
        else:
            gait_state = (-1.0, 1.0)

        angle = 2.0 * np.pi * phase
        return (
            np.array(
                [
                    np.sin(angle),
                    np.cos(angle),
                    gait_state[0],
                    gait_state[1],
                ],
                dtype=np.float32,
            )
            * self.command_scale_
        )

    def compute_observation(self):
        """Build the 58-wide actor frame and roll the 75-frame history buffer.

        Per-frame layout, in LeggedLabUltra DWAQ export order:
        `ang_vel(3), projected_gravity(3), command(3), gait(4),
        q_error(15), qd(15), previous_action(15)`.
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

        # Keep this layout in lockstep with the exported 61950 DWAQ policy.
        # Joint terms use policy/action order; motor data is gathered via lab2mj.
        proprio = np.concatenate(
            [
                ang_vel,
                projected_gravity,
                command,
                self._gait_terms(),
                policy_pos,
                policy_vel,
                self.last_actions_ * self.action_obs_scale_,
            ]
        ).astype(np.float32)
        if proprio.shape != (self.obs_size_,):
            raise ValueError(f"{self._prefix()} observation frame shape {proprio.shape} != ({self.obs_size_},)")

        if self.is_first_obs_:
            # Match RSL-RL export behavior: fill history with the first real
            # frame to avoid a cold-start zero-history transient.
            for i in range(self.num_hist_):
                start = i * self.obs_size_
                self.proprio_hist_buf_[start : start + self.obs_size_] = proprio
            self.is_first_obs_ = False
        else:
            # FIFO one frame left, newest frame appended at the tail.
            shift_size = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift_size] = self.proprio_hist_buf_[self.obs_size_ :]
            self.proprio_hist_buf_[shift_size:] = proprio

        self.single_frame_obs_ = np.clip(proprio, -self.clip_obs_, self.clip_obs_)
        self.hist_obs_ = np.clip(self.proprio_hist_buf_, -self.clip_obs_, self.clip_obs_)
        if self.debug_:
            print(f"{self._prefix()} command={command}")

    def compute_actions(self):
        """Run ONNX inference and clip filtered output actions in policy order.

        The first-order filter is part of this deployment contract, not shared
        runtime behavior, because plain WalkLion does not filter ONNX output.
        """
        if self.ort_session_ is None:
            return
        try:
            input_data_0 = self.single_frame_obs_.reshape(1, -1).astype(np.float32)
            input_name_0 = self.ort_session_.get_inputs()[0].name
            input_data_1 = self.hist_obs_.reshape(1, -1).astype(np.float32)
            input_name_1 = self.ort_session_.get_inputs()[1].name
            outputs = self.ort_session_.run(None, {input_name_0: input_data_0, input_name_1: input_data_1})
            output_data = np.asarray(outputs[0][0], dtype=np.float32)
            if output_data.shape != (self.action_num_,):
                raise ValueError(f"action shape {output_data.shape} != ({self.action_num_},)")
            # Low-pass policy output before clipping.  This smooths deployment
            # commands without changing the exported model itself.
            self.filtered_actions_ = (
                self.action_filter * output_data + (1.0 - self.action_filter) * self.filtered_actions_
            )
            self.actions_[:] = np.clip(self.filtered_actions_, -self.clip_act_, self.clip_act_)
            if self.is_first_action_:
                print(f"{self._onnx_prefix()} first inference ok")
                self.is_first_action_ = False
        except Exception as exc:
            print(f"{self._prefix()} ONNX Runtime inference error: {exc}")

    def on_exit(self):
        """Leave WALKLIONDWAQ state."""
        print(f"{self._prefix()} exit")
