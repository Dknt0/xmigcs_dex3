"""Dex29 MULTIAMP: one YAML selects a compatible ONNX and all state parameters."""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from common.control_timing import TickDecimator
from FSM.fsm_base import FSMState, FSMStateName

# Serial motor order from BodyServoIdMap (51..56,61..66,33,32,31,11..17,21..27).
MOTOR_JOINTS = (
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
)


class FSMStateMULTIAMP(FSMState):
    def __init__(self, robot_data):
        super().__init__(robot_data)
        root = Path(__file__).resolve().parent
        cfg = yaml.safe_load((root / "config/multi_amp.yaml").read_text())
        model_dir = root / "model"
        model_path = model_dir / cfg["model_path"]
        self.motor_num = cfg["motor_num"]
        self.action_num = cfg["actions_size"]
        self.num_hist = cfg["size"]["num_hist"]
        self.obs_size = cfg["size"]["observations_size"]
        if (self.motor_num, self.action_num, self.num_hist, self.obs_size) != (
            29,
            29,
            64,
            98,
        ):
            raise ValueError(
                "MULTIAMP requires 29 motors/actions and 64x98 observations"
            )
        if robot_data.motor_num != self.motor_num:
            raise ValueError("MULTIAMP requires 29 motors")
        self.joint_names = cfg["joint_names"]
        if len(self.joint_names) != 29 or set(self.joint_names) != set(MOTOR_JOINTS):
            raise ValueError("MULTIAMP must control all 29 named joints")
        self.policy_to_motor = np.array(
            [MOTOR_JOINTS.index(n) for n in self.joint_names]
        )
        self.offset = self._vector(cfg["init_state"]["default_joint_angles"])
        self.scale = self._vector(cfg["control"]["action_scale"])
        self.kp = self._vector(cfg["gains"]["kp"])[self.policy_to_motor]
        self.kd = self._vector(cfg["gains"]["kd"])[self.policy_to_motor]
        self.torque_limits = self._vector(cfg["gains"]["torque_limits"])[
            self.policy_to_motor
        ]
        if np.any(self.scale <= 0) or np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("Invalid MULTIAMP actuator parameters")
        norm = cfg["normalization"]
        self.action_clip = float(norm["clip_scales"]["clip_actions"])
        if (
            not norm["embedded"]
            or norm["epsilon"] != 0.01
            or norm["clip_scales"]["clip_observations"] != 20.0
            or self.action_clip != 18.0
        ):
            raise ValueError(
                "MULTIAMP requires normalization and action clipping inside ONNX"
            )
        self.dt = float(cfg["dt"])
        self.decimation = cfg["control"]["decimation"]
        if not np.isclose(self.dt * self.decimation, 0.02):
            raise ValueError("MULTIAMP requires 50Hz inference")
        self.clock = TickDecimator(self.decimation, self.dt, robot_data.control_dt_)
        self.warm_start_time = float(cfg["control"]["warm_start_time"])
        if not np.isfinite(self.warm_start_time) or self.warm_start_time < 0:
            raise ValueError("MULTIAMP warm_start_time must be finite and nonnegative")
        # Match WALKDEPTH: ramp on policy inference ticks, not main-loop ticks.
        self._warm_start_steps = (
            max(1, int(self.warm_start_time / (self.dt * self.decimation)))
            if self.warm_start_time > 0 else 0
        )
        self._warmup_inference_counter = 0
        self._warm_start_pose = np.zeros(self.motor_num, dtype=np.float32)
        self.mode_codes = cfg["mode"]["codes"]
        if self.mode_codes != {"walk": [0, 1], "prone": [1, 0]}:
            raise ValueError("Invalid MULTIAMP mode codes")
        self.initial_mode = cfg["mode"]["initial_mode"]
        if self.initial_mode not in self.mode_codes:
            raise ValueError("Invalid MULTIAMP initial mode")
        cooldown = float(cfg["mode"]["switch_cooldown"])
        if not np.isfinite(cooldown) or cooldown < 0:
            raise ValueError("Invalid MULTIAMP mode switch cooldown")
        self.mode_switch_cooldown_ns = round(cooldown * 1e9)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        if [(v.name, v.shape, v.type) for v in self.session.get_inputs()] != [
            ("history", [1, 64, 98], "tensor(float)")
        ] or [(v.name, v.shape, v.type) for v in self.session.get_outputs()] != [
            ("action", [1, 29], "tensor(float)")
        ]:
            raise ValueError("Unexpected MULTIAMP ONNX input/output")
        # Numerical warm-up happens before the control thread starts. No targets sent.
        self.session.run(None, {"history": np.zeros((1, 64, 98), dtype=np.float32)})
        self.history = np.zeros((1, 64, 98), dtype=np.float32)
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.action = np.zeros(29, dtype=np.float32)
        self.mode = self.initial_mode
        self.first_observation = True
        self.inference_count = 0
        self.toggle_count = 0

    @staticmethod
    def _vector(values):
        a = np.asarray(values, dtype=np.float32)
        if a.shape != (29,) or not np.isfinite(a).all():
            raise ValueError("Expected a finite 29-vector")
        return a

    def on_enter(self):
        self.clock.reset()
        self._warmup_inference_counter = 0
        if self._warm_start_steps:
            # Snapshot serial feedback in policy order; never use ZERO or the crawl offset.
            self._warm_start_pose[:] = self._vector(
                self.robot_data_.get_joint_pos()
            )[self.policy_to_motor]
        self.history.fill(0)
        self.previous_action.fill(0)
        self.action.fill(0)
        self.first_observation = True
        self.mode = self.initial_mode
        self.inference_count = 0
        self.toggle_count = 0
        # Entry also starts the window, so a double-click cannot enter then
        # immediately switch mode. Rejected requests never extend this window.
        self._last_mode_change_ns = time.monotonic_ns()
        print(f"[MULTIAMP] enter: {self.mode}")

    def on_reselect(self, flag):
        now = time.monotonic_ns()
        if now - self._last_mode_change_ns < self.mode_switch_cooldown_ns:
            return False  # Drop the event; never defer it until the window ends.
        self._last_mode_change_ns = now
        self.mode = "prone" if self.mode == "walk" else "walk"
        self.toggle_count += 1
        print(f"[MULTIAMP] target mode: {self.mode} (history retained)")
        return True

    def compute_observation(self):
        rd = self.robot_data_
        frame = np.concatenate(
            (
                rd.get_angular_velocity(),
                rd.get_project_gravity(),
                rd.get_walk_cmd(),
                rd.get_joint_pos()[self.policy_to_motor] - self.offset,
                rd.get_joint_vel()[self.policy_to_motor],
                self.previous_action,
                self.mode_codes[self.mode],
            )
        ).astype(np.float32)
        if frame.shape != (98,) or not np.isfinite(frame).all():
            raise FloatingPointError("Invalid MULTIAMP observation")
        if self.first_observation:
            self.history[0, :, :] = frame
            self.first_observation = False
        else:
            self.history[0, :-1] = self.history[0, 1:]
            self.history[0, -1] = frame
        return self.history

    def run(self, flag):
        if not self.clock.step():
            return
        self.compute_observation()
        try:
            result = self.session.run(["action"], {"history": self.history})[0]
        except Exception as exc:
            raise RuntimeError(
                "MULTIAMP inference failed; refusing stale action"
            ) from exc
        if (
            result.shape != (1, 29)
            or not np.isfinite(result).all()
            or np.max(np.abs(result)) > self.action_clip + 1e-5
        ):
            raise FloatingPointError("Invalid MULTIAMP action")
        self.action[:] = result[
            0
        ]  # Graph includes normalization and clip exactly once.
        q_target = self.offset + self.scale * self.action
        if self._warmup_inference_counter < self._warm_start_steps:
            self._warmup_inference_counter += 1
            blend = self._warmup_inference_counter / float(self._warm_start_steps)
            q_target = (1.0 - blend) * self._warm_start_pose + blend * q_target
        rd = self.robot_data_
        base = rd.whole_joint_num - rd.motor_num
        rd.q_d_s_[base + self.policy_to_motor] = q_target
        rd.q_dot_d_s_[base:] = 0
        rd.tau_d_s_[base:] = 0
        rd.joint_kp_s_[self.policy_to_motor] = self.kp
        rd.joint_kd_s_[self.policy_to_motor] = self.kd
        self.previous_action[:] = self.action
        self.inference_count += 1

    def on_exit(self):
        print("[MULTIAMP] exit")

    def check_transition(self, flag):
        return {
            "gotoMULTIAMP": FSMStateName.MULTIAMP,
            # 'gotoDEXDELTA': FSMStateName.DEXDELTA,
            "gotoSTOP": FSMStateName.STOP,
            "gotoZERO": FSMStateName.ZERO,
            # 'gotoDEXBLINDAMP': FSMStateName.DEXBLINDAMP,
            # 'gotoDEXAMP': FSMStateName.DEXAMP,
            "gotoWALKAMP": FSMStateName.WALKAMP,
            "gotoWALKDEPTH": FSMStateName.WALKDEPTH,
            "gotoWALKDEPTH100HZ": FSMStateName.WALKDEPTH100HZ,
            "gotoLIONZERO": FSMStateName.LIONZERO,
        }.get(flag.fsm_state_command)
