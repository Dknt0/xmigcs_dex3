"""
XBOX Controller compatibility layer.
Implements the same FSM modes and control flags as `stdin_keyboard_control.py` / `joystick.py`.
"""
import os
import yaml
import threading
import copy
import time
from typing import Optional
from dataclasses import dataclass
from sensor_msgs.msg import Joy
from .joystick import ControlFlag


class XBOXFlag(ControlFlag):
    def __init__(self):
        super().__init__()
        self.x_speed_command: float = 0.0
        self.y_speed_command: float = 0.0
        self.yaw_speed_command: float = 0.0
        self.motion_number: int = 0
        self.height_cmd: float = 0.89


@dataclass
class XBOXMap:
    # minimal axes/buttons mapping; populated from Joy message in xbox_map_read
    a: float = 0.0
    b: float = 0.0
    x: float = 0.0
    y: float = 0.0
    lb: float = 0.0
    rb: float = 0.0
    select: float = 0.0
    start: float = 0.0
    l_trigger: float = 0.0
    r_trigger: float = 0.0
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    # optional keys for devices with limited buttons
    home: float = 0.0
    # dpad placeholders (axes 6/7 or buttons depending on device)
    dpad_h: float = 0.0
    dpad_v: float = 0.0


class XBOXController:
    """XBOX adapter that mirrors joystick and keyboard behavior.

    It owns only Joy decoding, button chords, and local stale-input handling.
    RobotInterface remains the final safety gate for deciding whether walk
    velocity commands can enter RobotData.
    """

    def __init__(self):
        print("XBOX Controller Start")
        self.map = XBOXMap()
        self.flag = XBOXFlag()
        self.data_mutex = threading.Lock()

        # state tracking
        self.last_select = 0
        self.last_start = 0

        # configuration
        self.initial_height = 0.89
        self.current_height = 0.89
        self.target_height = 0.89
        self.forward_command_offset = 0.0
        self.lateral_command_offset = 0.0
        self.rotation_command_offset = 0.0
        self.max_forward_speed = 0.8
        self.max_backward_speed = 0.5
        self.max_lateral_speed = 0.4
        self.max_rotation_speed = 0.4
        # Missing Joy frames are treated as command loss.  The next fresh Joy
        # frame re-enables the adapter in `xbox_flag_update`.
        self.command_timeout_s = 0.5
        self.last_msg_time = 0.0

        # smoothing
        self.height_step = 0.05

        # default button map indices (can be overridden in config)
        self.button_map = {
            'a': 0, 'b': 1, 'x': 2, 'y': 3,
            'lb': 4, 'rb': 5,  'start': 7,
            'select': 6, 'home': 8
        }
        
        # default axis map indices (can be overridden in config)
        self.axis_map = {
            'lx': 0, 'ly': 1, 'rx': 3, 'ry': 4,
            'l_trigger': 2, 'r_trigger': 5,
            'dpad_h': 6, 'dpad_v': 7
        }
        self._load_config()

    def _load_config(self):
        try:
            config_path = os.path.join('.', 'config', 'dex_config.yaml')
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            xbox_cfg = cfg.get('xbox', {})
            # Allow device-specific Joy layouts without changing controller code.
            bm = xbox_cfg.get('button_map')
            if isinstance(bm, dict):
                for k, v in bm.items():
                    if k in self.button_map:
                        self.button_map[k] = int(v)

            am = xbox_cfg.get('axis_map')
            if isinstance(am, dict):
                for k, v in am.items():
                    if k in self.axis_map:
                        self.axis_map[k] = int(v)
            
            self.initial_height = xbox_cfg.get('initial_height', 0.89)
            self.forward_command_offset = xbox_cfg.get('forward_command_offset', 0.0)
            self.lateral_command_offset = xbox_cfg.get('lateral_command_offset', 0.0)
            self.rotation_command_offset = xbox_cfg.get('rotation_command_offset', 0.0)
            self.height_step = xbox_cfg.get('height_step', 0.05)
            self.max_forward_speed = xbox_cfg.get('max_forward_speed', 0.8)
            self.max_backward_speed = xbox_cfg.get('max_backward_speed', 0.5)
            self.max_lateral_speed = xbox_cfg.get('max_lateral_speed', 0.4)
            self.max_rotation_speed = xbox_cfg.get('max_rotation_speed', 0.4)
            self.command_timeout_s = xbox_cfg.get('command_timeout_s', 0.5)

            self.current_height = self.initial_height
            self.target_height = self.initial_height
            self.flag.height_cmd = self.current_height
            print(f"Loaded XBOX config: initial_height={self.initial_height}")
        except Exception as e:
            print(f"[XBOXController] YAML load error: {e}")

    def xbox_map_read(self, msg: Joy):
        """Populate internal map from a ROS Joy message."""
        with self.data_mutex:
            # axes layout may differ; try safe indexing
            axes = list(msg.axes) + [0.0] * 16
            buttons = list(msg.buttons) + [0] * 32
            
            # common mapping assumptions (best-effort)
            self.map.lx = axes[self.axis_map['lx']]
            self.map.ly = axes[self.axis_map['ly']]
            self.map.rx = axes[self.axis_map['rx']]
            self.map.ry = axes[self.axis_map['ry']]
            # triggers sometimes on axes
            self.map.l_trigger = axes[self.axis_map['l_trigger']]
            self.map.r_trigger = axes[self.axis_map['r_trigger']]
            # dpad may be on axes
            self.map.dpad_h = axes[self.axis_map['dpad_h']]
            self.map.dpad_v = axes[self.axis_map['dpad_v']]
            
            # buttons using button_map indices
            for name, idx in self.button_map.items():
                try:
                    val = buttons[idx]
                except Exception:
                    val = 0
                setattr(self.map, name, val)
            self.last_msg_time = time.monotonic()

    def xbox_flag_update(self):
        """Update ControlFlag from the latest decoded XBOX map.

        This method is called only after a fresh Joy message has been decoded,
        so it is the right place to recover from a prior stale-input disable.
        """
        with self.data_mutex:
            self.flag.enable = True
            # FSM state mapping.  Lion policy commands use shoulder-button
            # chords so single-button STOP/ZERO remain easy to hit:
            #   LB+A -> prep pose, LB+B -> WalkLion, RB+B -> WalkLionDWAQ.
            # POLICYZERO requires isolated LB+X; mixed LB+X+A/B/Home should
            # not steal the Lion prep/start or legacy BeyondMimic chords.
            if self._policyzero_chord_pressed():
                self.flag.fsm_state_command = 'gotoPOLICYZERO'
            elif self.map.lb == 1 and self.map.a == 1:
                self.flag.fsm_state_command = 'gotoLIONZERO'
            elif self.map.lb == 1 and self.map.b == 1:
                self.flag.fsm_state_command = 'gotoWALKLION'
            elif self.map.rb == 1 and self.map.b == 1:
                self.flag.fsm_state_command = 'gotoWALKLIONDWAQ'
            # c -> gotoSTOP
            elif self.map.y == 1:
                self.flag.fsm_state_command = 'gotoSTOP'
            # h -> gotoDH (Left trigger + A)
            # v -> gotoBEYONDMIMIC (Left trigger + home)
            elif self.map.l_trigger < -0.5 and self.map.home == 1:
                self.flag.fsm_state_command = 'gotoBEYONDMIMIC'
            # z -> gotoZERO
            elif self.map.x == 1:
                self.flag.fsm_state_command = 'gotoZERO'

            # detect state change
            if not hasattr(self, '_last_state'):
                self._last_state = self.flag.fsm_state_command
            state_changed = (self.flag.fsm_state_command != self._last_state)
            self._last_state = self.flag.fsm_state_command

            if (state_changed and
                (self.flag.fsm_state_command == 'gotoZERO' or self.flag.fsm_state_command == 'gotoSTOP')):
                self.current_height = self.initial_height
                self.target_height = self.initial_height
                self.flag.height_cmd = self.current_height

            # velocity mapping: continuous (left stick) and small discrete adjustments via buttons
            ly = float(self.map.ly)
            lx = float(self.map.lx)
            rx = float(self.map.rx)

            # continuous stick control (same scaling as joystick)
            if ly >= 0:
                self.flag.x_speed_command = (ly * self.max_forward_speed + self.forward_command_offset)
            else:
                self.flag.x_speed_command = ly * self.max_backward_speed

            self.flag.y_speed_command = (lx * -self.max_lateral_speed + self.lateral_command_offset)
            self.flag.yaw_speed_command = (rx * -self.max_rotation_speed + self.rotation_command_offset)

            # discrete movement adjustments (map buttons to keyboard-like increments)
            # emulate w/s/a/d via shoulder buttons or dpad if needed
            # D-pad (axes 6/7) is common: we'll read them in xbox_map_read if available
            dpad_h = getattr(self.map, 'dpad_h', 0.0)
            dpad_v = getattr(self.map, 'dpad_v', 0.0)
            # left/right dpad emulate arrow keys for height adjust
            motion_add_number = 0
            if dpad_h == -1.0 and getattr(self, 'last_dpad_h', 0.0) == 0.0:
                # left arrow -> increase height
                motion_add_number = 1
            elif dpad_h == 1.0 and getattr(self, 'last_dpad_h', 0.0) == 0.0:
                # right arrow -> decrease height
                motion_add_number = -1

            # Remove select/start height adjustments to use only dpad_h for height control
            # if self.map.select == 1 and self.last_select == 0 and motion_add_number == 0:
            #     motion_add_number = 1
            # elif self.map.start == 1 and self.last_start == 0 and motion_add_number == 0:
            #     motion_add_number = -1

            self.last_select = self.map.select
            self.last_start = self.map.start
            self.last_dpad_h = dpad_h

            self.flag.motion_number = motion_add_number

            if motion_add_number != 0:
                new_target = self.target_height + (motion_add_number * 0.05)
                if new_target > 0.90:
                    new_target = 0.90
                elif new_target < 0.65:
                    new_target = 0.65
                self.target_height = new_target

            # smooth height
            step = 0.01
            if abs(self.current_height - self.target_height) > step:
                if self.current_height < self.target_height:
                    self.current_height += step
                else:
                    self.current_height -= step
            else:
                self.current_height = self.target_height

            self.flag.height_cmd = self.current_height

            # reset movement (r key) -> using START button
            if self.map.start == 1:
                self.flag.x_speed_command = 0.0
                self.flag.y_speed_command = 0.0
                self.flag.yaw_speed_command = 0.0

    def get_xbox_flag(self) -> ControlFlag:
        """Return a snapshot of the current flag, zeroing stale input first."""
        with self.data_mutex:
            if self._is_command_stale():
                self.flag.enable = False
                self._zero_walk_speed_command()
            return copy.deepcopy(self.flag)

    def _is_command_stale(self) -> bool:
        """Return True when the ROS Joy stream has stopped updating."""
        return time.monotonic() - self.last_msg_time > self.command_timeout_s

    def _zero_walk_speed_command(self):
        """Clear velocity commands without changing the requested FSM state."""
        self.flag.x_speed_command = 0.0
        self.flag.y_speed_command = 0.0
        self.flag.yaw_speed_command = 0.0

    def _policyzero_chord_pressed(self) -> bool:
        """Return True only for the isolated LB+X bridge chord."""
        return (
            self.map.lb == 1
            and self.map.x == 1
            and self.map.a == 0
            and self.map.b == 0
            and self.map.y == 0
            and self.map.rb == 0
            and self.map.home == 0
        )

    def init(self) -> int:
        print("XBOX controller initialized")
        return 0
