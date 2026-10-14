"""One-trigger safety bridge between WalkLion-series policies and BeyondMimic.

PolicyZero is intentionally not a learned balance policy.  It is a short
position-control bridge that decides, before leaving the active policy, whether
the target policy pose is close enough to hard-switch or requires a bounded
quintic blend.  Oversized source/target pose deltas are rejected because this
repository does not provide an online FK/support-polygon safety model for
Cartesian-space validation.

The bridge auto-transitions into the target policy once the short blend has
finished.  Operators therefore only press the bridge command once, which avoids
the dangerous "stand in an interpolated pose and wait for a second trigger"
failure mode on the real robot.
"""

import os
from typing import Optional

import numpy as np
import yaml

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData


class FSMStatePolicyZero(FSMState):
    """Safety-gated bridge for WalkLion <-> BeyondMimic transitions."""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.POLICYZERO

        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        config_path = os.path.join(current_dir, "config", "policyzero.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.motor_num_ = int(config["motor_num"])
        self.dt_ = float(config["dt"])
        self.smooth_duration_s_ = float(config["smooth_duration_s"])
        self.direct_switch_delta_rad_ = float(config["direct_switch_delta_rad"])
        self.smooth_switch_max_delta_rad_ = float(config["smooth_switch_max_delta_rad"])
        self.leg_delta_limit_rad_ = float(config["leg_delta_limit_rad"])
        self.waist_delta_limit_rad_ = float(config["waist_delta_limit_rad"])
        self.upper_delta_limit_rad_ = float(config["upper_delta_limit_rad"])
        self.entry_tolerance_rad_ = float(config["entry_tolerance_rad"])
        self.joint_xml = list(config["joint_xml"])
        self.kp_pos_ = self._as_vector(config["kp_pos"], "kp_pos")
        self.kd_pos_ = self._as_vector(config["kd_pos"], "kd_pos")

        self.beyondmimic_target_ = self._load_beyondzero_target(root_dir)
        self.walklion_target_ = self._load_walklion_target(root_dir)

        self.source_state_: Optional[FSMStateName] = None
        self.target_state_: Optional[FSMStateName] = None
        self.source_target_ = self.walklion_target_.copy()
        self.target_position_ = self.beyondmimic_target_.copy()
        self.start_position_ = np.zeros(self.motor_num_, dtype=np.float32)
        self.transition_mode_: Optional[str] = None
        self.step_index_ = 0
        self.total_steps_ = max(1, int(self.smooth_duration_s_ / self.dt_))
        self.completed_ = False
        self.last_block_reason = ""

    def _as_vector(self, value, label: str) -> np.ndarray:
        """Return a 29-wide float32 vector and fail fast on config drift."""
        vector = np.array(value, dtype=np.float32)
        if vector.shape != (self.motor_num_,):
            raise ValueError(f"[FSMStatePolicyZero] {label} length {len(vector)} != {self.motor_num_}")
        return vector

    def _load_beyondzero_target(self, root_dir: str) -> np.ndarray:
        """Load the robot-proven BeyondMimic prep pose from BeyondZero config."""
        path = os.path.join(root_dir, "policy", "beyondzero", "config", "beyondzero.yaml")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        self._validate_joint_xml(config["joint_xml"], "BeyondZero")
        return self._as_vector(config["zero_target"], "beyondzero.zero_target")

    def _load_walklion_target(self, root_dir: str) -> np.ndarray:
        """Build the full 29-motor WalkLion neutral target from policy config."""
        path = os.path.join(root_dir, "policy", "walk_lion", "config", "walk_lion.yaml")
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        self._validate_joint_xml(config["joint_xml"], "WalkLion")
        target = np.zeros(self.motor_num_, dtype=np.float32)
        for joint_name, joint_cfg in config.get("passive_joint_targets", {}).items():
            if joint_name not in self.joint_xml:
                raise ValueError(f"[FSMStatePolicyZero] WalkLion passive joint missing: {joint_name}")
            target[self.joint_xml.index(joint_name)] = float(joint_cfg["angle"])
        joint_names = list(config["joint_names"])
        default_angles = list(config["init_state"]["default_joint_angles"])
        if len(joint_names) != len(default_angles):
            raise ValueError("[FSMStatePolicyZero] WalkLion joint/default length mismatch")
        for joint_name, angle in zip(joint_names, default_angles):
            target[self.joint_xml.index(joint_name)] = float(angle)
        return target

    def _validate_joint_xml(self, joint_xml, label: str) -> None:
        """Ensure all target configs use the same motor/XML order."""
        if list(joint_xml) != self.joint_xml:
            raise ValueError(f"[FSMStatePolicyZero] {label} joint_xml does not match policyzero joint_xml")

    def _plan_for(self, current_state: FSMStateName, flag: ControlFlag):
        """Return source pose, target state, and target pose for a bridge entry."""
        if flag.fsm_state_command != "gotoPOLICYZERO":
            return None
        if current_state == FSMStateName.WALKLION:
            return self.walklion_target_, FSMStateName.BEYONDMIMIC, self.beyondmimic_target_
        if current_state == FSMStateName.BEYONDMIMIC:
            return self.beyondmimic_target_, FSMStateName.WALKLION, self.walklion_target_
        return None

    def can_enter_from(self, current_state: FSMStateName, flag: ControlFlag) -> bool:
        """Reject bridge entry when source pose or target delta is unsafe."""
        plan = self._plan_for(current_state, flag)
        if plan is None:
            self.last_block_reason = f"unsupported source command: {current_state.name}, {flag.fsm_state_command}"
            return False
        source_target, target_state, target_position = plan
        if not self._within_tolerance(source_target, self.entry_tolerance_rad_):
            max_error = self._max_joint_error(source_target)
            self.last_block_reason = (
                f"actual pose differs from {current_state.name} target by {max_error:.3f} rad; "
                f"limit {self.entry_tolerance_rad_:.3f} rad"
            )
            return False
        if self._select_transition_mode(source_target, target_position) is None:
            return False
        self.last_block_reason = ""
        return target_state in (FSMStateName.BEYONDMIMIC, FSMStateName.WALKLION)

    def prepare_transition(self, current_state: FSMStateName, flag: ControlFlag) -> None:
        """Latch bridge direction and mode before entering POLICYZERO."""
        plan = self._plan_for(current_state, flag)
        if plan is None:
            raise ValueError(f"[FSMStatePolicyZero] cannot plan from {current_state} using {flag.fsm_state_command}")
        self.source_target_, self.target_state_, self.target_position_ = plan
        self.source_state_ = current_state
        self.transition_mode_ = self._select_transition_mode(self.source_target_, self.target_position_)
        if self.transition_mode_ is None:
            raise ValueError(f"[FSMStatePolicyZero] unsafe target delta: {self.last_block_reason}")

    def on_enter(self):
        """Start the latched bridge plan from the measured source pose."""
        print(f"[FSMStatePolicyZero] Enter bridge to {self.target_state_.name} ({self.transition_mode_})")
        self.step_index_ = 0
        self.start_position_ = self.robot_data_.get_joint_pos().copy()
        self.completed_ = self.transition_mode_ == "direct"

    def run(self, flag: ControlFlag):
        """Write one bridge command sample and hold position-control gains."""
        if self.robot_data_ is None:
            return
        if self.target_state_ is None or self.transition_mode_ not in ("direct", "smooth"):
            # Defensive hold: normal FSM entry always calls prepare_transition()
            # first, but an unplanned bridge must never invent motion.
            self.last_block_reason = "bridge plan not latched"
            pos_cmd = self.robot_data_.get_joint_pos()
            self.completed_ = False
        elif self.transition_mode_ == "direct":
            pos_cmd = self.target_position_
            self.completed_ = True
        else:
            # Advance before sampling so the bounded blend completes exactly
            # after total_steps_ control ticks and can auto-transition on the
            # next FSM check.
            self.step_index_ = min(self.step_index_ + 1, self.total_steps_)
            t = self.step_index_ / float(self.total_steps_)
            blend = self._quintic_blend(t)
            pos_cmd = (1.0 - blend) * self.start_position_ + blend * self.target_position_
            self.completed_ = self.step_index_ >= self.total_steps_

        base = self.robot_data_.q_d_s_.shape[0] - self.motor_num_
        self.robot_data_.q_d_s_[base : base + self.motor_num_] = pos_cmd
        self.robot_data_.q_dot_d_s_[base : base + self.motor_num_] = 0.0
        self.robot_data_.tau_d_s_[base : base + self.motor_num_] = 0.0
        self.robot_data_.joint_kp_s_[: self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_s_[: self.motor_num_] = self.kd_pos_

    @staticmethod
    def _quintic_blend(t: float) -> float:
        """Quintic smoothstep with zero endpoint velocity and acceleration."""
        t = min(max(float(t), 0.0), 1.0)
        return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5

    def _max_joint_error(self, target: np.ndarray) -> float:
        """Return max absolute measured joint error against a 29-motor target."""
        actual = self.robot_data_.get_joint_pos()
        return float(np.max(np.abs(actual - target)))

    def _within_tolerance(self, target: np.ndarray, tolerance: float) -> bool:
        """Return True when measured joints are close enough to a source pose."""
        return self._max_joint_error(target) <= float(tolerance)

    def _select_transition_mode(self, source_target: np.ndarray, target_position: np.ndarray) -> Optional[str]:
        """Choose direct hard switch, short smooth switch, or reject."""
        max_delta = float(np.max(np.abs(target_position - source_target)))
        if max_delta <= self.direct_switch_delta_rad_:
            return "direct"
        if self._target_delta_is_safe(source_target, target_position):
            return "smooth"
        return None

    def _target_delta_is_safe(self, source_target: np.ndarray, target_position: np.ndarray) -> bool:
        """Gate bridge plans using conservative joint-group delta limits.

        This is a joint-space proxy for Cartesian safety.  The real robot should
        not leave its actively balancing policy if the policy start/end poses
        require large leg, waist, or lion-head mount changes.
        """
        delta = np.abs(target_position - source_target)
        group_limits = (
            ("leg", delta[:12], self.leg_delta_limit_rad_),
            ("waist", delta[12:15], self.waist_delta_limit_rad_),
            ("upper", delta[15:], self.upper_delta_limit_rad_),
        )
        max_delta = float(np.max(delta))
        if max_delta > self.smooth_switch_max_delta_rad_:
            self.last_block_reason = (
                f"target delta {max_delta:.3f} rad exceeds global limit "
                f"{self.smooth_switch_max_delta_rad_:.3f} rad"
            )
            return False
        for group_name, group_delta, group_limit in group_limits:
            group_max = float(np.max(group_delta)) if group_delta.size else 0.0
            if group_max > group_limit:
                self.last_block_reason = (
                    f"target delta {group_max:.3f} rad in {group_name} joints exceeds "
                    f"{group_limit:.3f} rad"
                )
                return False
        self.last_block_reason = ""
        return True

    def on_exit(self):
        """Leave bridge state."""
        print("[FSMStatePolicyZero] Exit bridge state")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """Auto-transition to the target policy after the bridge plan finishes."""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        if flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        if flag.fsm_state_command == "gotoLIONZERO":
            return FSMStateName.LIONZERO
        if self.completed_ and self.target_state_ is not None:
            return self.target_state_
        return None
