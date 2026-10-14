"""Configurable prep pose before WALKLION and WALKLIONDWAQ policies.

LionZero is a non-policy position-control state.  It moves all 29 motors to a
known pose before a learned WalkLion policy is allowed to start.  This keeps
policy entry conditions explicit and avoids accidental direct transitions from
STOP/ZERO/other policies into policies trained with different initial states.
"""

import os

import numpy as np
import yaml

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData


class FSMStateLionZero(FSMState):
    """Hold one configurable 29-motor pose and gate WalkLion policy entry."""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.LIONZERO

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "lionzero.yaml")
        with open(config_path, "r") as f:
            policy_config = yaml.safe_load(f)

        self.motor_num_ = int(policy_config["motor_num"])
        self.target_position_ = self._as_vector(
            policy_config["target_joint_angles"], self.motor_num_, "target_joint_angles"
        )
        self.kp_pos_ = self._as_vector(policy_config["kp_pos"], self.motor_num_, "kp_pos")
        self.kd_pos_ = self._as_vector(policy_config["kd_pos"], self.motor_num_, "kd_pos")
        self.interp_step_ = float(policy_config.get("interp_step", 0.001))
        self.interp_max_ = float(policy_config.get("interp_max", 1.0))

        self.q_factor_ = 0.0
        self.start_position_ = np.zeros(self.motor_num_, dtype=np.float32)

    @staticmethod
    def _as_vector(value, expected_len: int, label: str) -> np.ndarray:
        """Return a float32 vector and fail fast on config length drift."""
        vector = np.array(value, dtype=np.float32)
        if vector.shape != (expected_len,):
            raise ValueError(f"[FSMStateLionZero] {label} length {len(vector)} != {expected_len}")
        return vector

    def on_enter(self):
        """Cache measured motor pose and start interpolation to target pose."""
        print("[FSMStateLionZero] Enter lionzero prep state")
        self.q_factor_ = 0.0
        self.start_position_ = self.robot_data_.q_a_s_[-self.motor_num_:].copy()

    def run(self, flag: ControlFlag):
        """Write interpolated 29-motor position targets and PD gains."""
        if self.robot_data_ is None:
            return

        if self.q_factor_ < self.interp_max_:
            pos_cmd = (1.0 - self.q_factor_) * self.start_position_ + self.q_factor_ * self.target_position_
            self.q_factor_ = min(self.q_factor_ + self.interp_step_, self.interp_max_)
        else:
            pos_cmd = self.target_position_

        base = self.robot_data_.q_d_s_.shape[0] - self.motor_num_
        self.robot_data_.q_d_s_[base : base + self.motor_num_] = pos_cmd
        self.robot_data_.q_dot_d_s_[base : base + self.motor_num_] = 0.0
        self.robot_data_.tau_d_s_[base : base + self.motor_num_] = 0.0
        self.robot_data_.joint_kp_s_[: self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_s_[: self.motor_num_] = self.kd_pos_

    def on_exit(self):
        """Leave LionZero."""
        print("[FSMStateLionZero] Exit lionzero prep state")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """Allow WalkLion policy start only after entering this prep state."""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        if flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        if flag.fsm_state_command == "gotoLIONZERO":
            return FSMStateName.LIONZERO
        if flag.fsm_state_command == "gotoWALKLION":
            return FSMStateName.WALKLION
        if flag.fsm_state_command == "gotoWALKLIONDWAQ":
            return FSMStateName.WALKLIONDWAQ
        return None
