"""
Websocket backend (mobile console) for the xMIGCS control node.

Speaks the same console protocol as the bitbot-ovinf / efc_sdk backend so the
Bitbot-Android app (and the BitbotCopilot desktop client) can connect:

  HTTP GET /monitor/headers     -> {"kernel": [...], "bus": {"devices": [...]}, "extra": []}
  HTTP GET /monitor/stateslist  -> {"states": [{"id": .., "name": ..}, ..]}
  HTTP GET /setting/control/get -> [{"event": .., "kb_key": ..}, ..]
  WS       /console             -> client polls {"type":"request_data","data":""} (10 Hz) and
                                   gets {"type":"monitor_data","data":"{\\"data\\":[..]}"} back
                                   (data is a JSON string containing JSON, like glaze writes it);
                                   client sends {"type":"events","data":"{\\"events\\":[..]}"} to
                                   control the FSM. Button events carry value 1 (down) / 2 (up);
                                   velocity events (set_vel_x/y/w) carry an IEEE-754 double
                                   bit-cast to int64 in the value field.

The server additionally pushes the current velocity commands back to every
client as periodic "events" messages (value = bit-cast double), which the
Android app simply ignores and the test frontend prints.

Event -> FSM mapping (single place to change, EVENT_TO_FSM):
  stop                   -> gotoSTOP
  start / init_pose      -> gotoZERO
  run_policy             -> gotoWALKAMP
  enable_warking_policy  -> gotoWALKAMP   (historical typo in the app, keep it)
  enable_standing_policy -> gotoSTAND
  enable_robust_policy   -> gotoWALKDEPTH (depth-perceptive policy)
  set_vel_x/y/w          -> absolute walk velocity command (joystick clients;
                            double bit-cast to int64 in the value field)
  velo_x/y/w_increase/decrease -> velocity nudge by velocity_step (0.05 m/s),
                            the keyboard method of ovinf (i/k j/l u/o)
  height_up/height_down  -> height adjust (extra events for the test frontend)

/setting/control/get serves the same kb_key bindings as bitbot-ovinf
backend.json (p=init_pose, 1/2/3=policies, space=stop, i/k/j/l/u/o=nudges).

Enable with `control_tool: websocket` in config/dex_config.yaml.
"""

import asyncio
import json
import math
import os
import queue
import struct
import threading
import time
import copy

import yaml
from aiohttp import WSMsgType, web

from .body_id_map import BodyServoIdMap
from .joystick import ControlFlag

# button event -> FSM state command
EVENT_TO_FSM = {
    "stop": "gotoSTOP",
    "start": "gotoZERO",
    "init_pose": "gotoZERO",
    "run_policy": "gotoWALKAMP",
    "enable_warking_policy": "gotoWALKDEPTH",
    "enable_standing_policy": "gotoWALKAMP",
    # "enable_robust_policy": "gotoWALKDEPTH100HZ",
    "enable_robust_policy": "gotoMULTIAMP",  # dex 盲走 AMP v2 (kb $)
}

# velocity nudge events (keyboard-style clients): name -> (axis, delta)
# axis 0/1/2 = x/y/yaw; step configurable via `websocket.velocity_step`
NUDGE_EVENTS = {
    "velo_x_increase": (0, +1),
    "velo_x_decrease": (0, -1),
    "velo_y_increase": (1, +1),
    "velo_y_decrease": (1, -1),
    "velo_w_increase": (2, +1),
    "velo_w_decrease": (2, -1),
}

# FSM state command -> numeric state id published in the kernel block
COMMAND_TO_STATE_ID = {
    "gotoSTOP": 1,
    "gotoZERO": 2,
    "gotoWALKAMP": 3,
    "gotoWALKDEPTH": 4,
    "gotoWALKDEPTH100HZ": 5,
    "gotoMULTIAMP": 5,
}

# served at /setting/control/get — identical to bitbot-ovinf backend.json
CONTROL_SETTINGS = [
    {"event": "enable_record", "kb_key": "m"},
    {"event": "start", "kb_key": "8"},
    {"event": "power_on", "kb_key": "9"},
    {"event": "stop", "kb_key": " "},
    {"event": "init_pose", "kb_key": "p"},
    {"event": "run_policy", "kb_key": "r"},
    {"event": "enable_standing_policy", "kb_key": "1"},
    {"event": "enable_warking_policy", "kb_key": "2"},
    {"event": "enable_robust_policy", "kb_key": "3"},
    {"event": "policy_switch", "kb_key": "$"},
    {"event": "velo_x_increase", "kb_key": "i"},
    {"event": "velo_x_decrease", "kb_key": "k"},
    {"event": "velo_y_increase", "kb_key": "j"},
    {"event": "velo_y_decrease", "kb_key": "l"},
    {"event": "velo_w_increase", "kb_key": "u"},
    {"event": "velo_w_decrease", "kb_key": "o"},
    {"event": "set_vel_x", "kb_key": "!"},
    {"event": "set_vel_y", "kb_key": "@"},
    {"event": "set_vel_w", "kb_key": "#"},
]

JOINT_HEADERS = [
    "mode",
    "actual_position",
    "target_position",
    "actual_velocity",
    "actual_torque",
    "p_tain",
    "d_gain",
    "temperature",
]
IMU_HEADERS = [
    "roll",
    "pitch",
    "yaw",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
]
CMD_HEADERS = ["x_speed", "y_speed", "yaw_speed"]


# joint group (by index) -> device type; the app makes one tab per type
def _joint_type(index: int) -> str:
    if index < 12:
        return "Leg"
    if index < 15:
        return "Waist"
    return "Arm"


class WebsocketFlag(ControlFlag):
    """Control flag produced by the websocket console."""

    def __init__(self):
        super().__init__()
        self.x_speed_command: float = 0.0
        self.y_speed_command: float = 0.0
        self.yaw_speed_command: float = 0.0
        self.height_cmd: float = 0.89


def _double_to_bits(value: float) -> int:
    """IEEE-754 double bit pattern as a signed int64 (the wire 'value')."""
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


def _bits_to_double(value: int) -> float:
    return struct.unpack("<d", struct.pack("<q", int(value)))[0]


class WebsocketController:
    """HTTP + websocket console server that also acts as a control tool.

    Same interface shape as XBOXController / JoystickHumanoid: init() starts
    the server thread, update_flag() is called by the control loop every cycle
    (drains client events into the flag and snapshots robot_data for the
    monitor dataframe), get_ws_flag() returns the current ControlFlag.
    """

    def __init__(self, robot_data, robot_fsm=None):
        print("Websocket Controller Start")
        self.robot_data = robot_data
        self.fsm = robot_fsm  # optional; used for the state column

        self.flag = WebsocketFlag()
        self.flag_mutex = threading.Lock()
        self.events_queue = queue.Queue()  # (name, value) from clients
        self._last_button_value = {}  # name -> last value, for press/release dedupe

        # monitor snapshot, written by the control thread, read by ws threads
        self._snapshot = []
        self._snapshot_mutex = threading.Lock()
        self._monitor_str = '{"data":[]}'  # cached inner JSON string
        self._snapshot_dirty = False

        # loop statistics for the kernel block
        self._periods = 0
        self._loop_period_ms = 0.0
        self._process_ms = 0.0
        self._last_update_time = None

        # configuration (defaults, overridden by dex_config.yaml `websocket:`)
        self.host = "0.0.0.0"
        self.port = 12888
        self.max_x_plus_speed = 0.8
        self.max_x_minus_speed = 0.5
        self.max_y_speed = 0.4
        self.max_yaw_speed = 0.4
        self.echo_interval = 0.5
        self.velocity_step = 0.05
        self.initial_height = 0.89
        self.height_step = 0.05
        self.current_height = 0.89
        self.target_height = 0.89

        self.id_map = BodyServoIdMap()
        self.id_map.body_can_id_map_init()

        self._load_config()
        self.current_height = self.initial_height
        self.target_height = self.initial_height
        self.flag.height_cmd = self.current_height

        self._headers_json = self._build_headers_json()
        self._states_json = json.dumps(
            {
                "states": [
                    {"id": 0, "name": "kernel_idle"},
                    {"id": 1, "name": "fsm_stop"},
                    {"id": 2, "name": "fsm_zero"},
                    {"id": 3, "name": "fsm_walkamp"},
                    {"id": 4, "name": "fsm_walkdepth"},
                    {"id": 5, "name": "fsm_stand"},
                    {"id": 6, "name": "fsm_multiamp"},
                ]
            }
        )

        # server thread plumbing
        self._loop = None
        self._runner = None
        self._server_thread = None
        self._running = False

    # ------------------------------------------------------------------ setup

    def _load_config(self):
        try:
            config_path = os.path.join(".", "config", "dex_config.yaml")
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            ws_cfg = cfg.get("websocket", {}) or {}
            self.host = ws_cfg.get("host", self.host)
            self.port = int(ws_cfg.get("port", self.port))
            self.max_x_plus_speed = float(
                ws_cfg.get("max_x_plus_speed", self.max_x_plus_speed)
            )
            self.max_x_minus_speed = float(
                ws_cfg.get("max_x_minus_speed", self.max_x_minus_speed)
            )
            self.max_y_speed = float(ws_cfg.get("max_y_speed", self.max_y_speed))
            self.max_yaw_speed = float(ws_cfg.get("max_yaw_speed", self.max_yaw_speed))
            self.echo_interval = float(ws_cfg.get("echo_interval", self.echo_interval))
            self.velocity_step = float(ws_cfg.get("velocity_step", self.velocity_step))
            self.initial_height = float(
                ws_cfg.get("initial_height", self.initial_height)
            )
            self.height_step = float(ws_cfg.get("height_step", self.height_step))
            print(f"Loaded websocket config: {self.host}:{self.port}")
        except Exception as e:
            print(f"[WebsocketController] YAML load error: {e}")

    def _build_headers_json(self) -> str:
        devices = []
        for i in range(self.id_map.whole_motor_nums):
            name = self.id_map.get_name_by_index(i)
            devices.append(
                {"name": name, "type": _joint_type(i), "headers": list(JOINT_HEADERS)}
            )
        devices.append({"name": "imu", "type": "IMU", "headers": list(IMU_HEADERS)})
        devices.append(
            {"name": "walk_cmd", "type": "Command", "headers": list(CMD_HEADERS)}
        )
        return json.dumps(
            {
                "kernel": [
                    "state",
                    "periods_count",
                    "period(ms)",
                    "process_t(ms)",
                    "kernel_t(ms)",
                ],
                "bus": {"devices": devices},
                "extra": [],
            }
        )

    # ------------------------------------------------------- controller iface

    def init(self) -> int:
        self._running = True
        self._server_thread = threading.Thread(
            target=self._server_main, daemon=True, name="ws-console"
        )
        self._server_thread.start()
        print(
            f"Websocket console: http://{self.host}:{self.port}/ "
            f"(ws://...:{self.port}/console)"
        )
        return 0

    def update_flag(self):
        """Called by the control loop every cycle: apply client events, then
        snapshot the robot data for the monitor dataframe."""
        t_start = time.perf_counter()
        now = time.perf_counter()
        if self._last_update_time is not None:
            self._loop_period_ms = (now - self._last_update_time) * 1000.0
        self._last_update_time = now

        # drain client events
        while True:
            try:
                name, value = self.events_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_event(name, value)

        # height smoothing (parity with the xbox controller)
        with self.flag_mutex:
            step = 0.01
            if abs(self.current_height - self.target_height) > step:
                self.current_height += (
                    step if self.current_height < self.target_height else -step
                )
            else:
                self.current_height = self.target_height
            self.flag.height_cmd = self.current_height

        self._periods += 1
        self._update_snapshot()

        self._process_ms = (time.perf_counter() - t_start) * 1000.0

    def get_ws_flag(self) -> ControlFlag:
        with self.flag_mutex:
            return copy.deepcopy(self.flag)

    def stop(self):
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)

    # ------------------------------------------------------------ event logic

    def _apply_event(self, name: str, value):
        with self.flag_mutex:
            # a value=2 right after value=1 of the same key is the release
            # half of a desktop press/release pair -> apply only once
            previous_value = self._last_button_value.get(name)
            is_release = value == 2 and previous_value == 1
            self._last_button_value[name] = value

            if name == "set_vel_x":
                self.flag.x_speed_command = self._clamp_axis(
                    _bits_to_double(value),
                    -self.max_x_minus_speed,
                    self.max_x_plus_speed,
                )
            elif name == "set_vel_y":
                self.flag.y_speed_command = self._clamp_axis(
                    _bits_to_double(value), -self.max_y_speed, self.max_y_speed
                )
            elif name == "set_vel_w":
                self.flag.yaw_speed_command = self._clamp_axis(
                    _bits_to_double(value), -self.max_yaw_speed, self.max_yaw_speed
                )
            elif name in NUDGE_EVENTS and not is_release:
                axis, sign = NUDGE_EVENTS[name]
                delta = sign * self.velocity_step
                if axis == 0:
                    self.flag.x_speed_command = self._clamp_axis(
                        self.flag.x_speed_command + delta,
                        -self.max_x_minus_speed,
                        self.max_x_plus_speed,
                    )
                elif axis == 1:
                    self.flag.y_speed_command = self._clamp_axis(
                        self.flag.y_speed_command + delta,
                        -self.max_y_speed,
                        self.max_y_speed,
                    )
                else:
                    self.flag.yaw_speed_command = self._clamp_axis(
                        self.flag.yaw_speed_command + delta,
                        -self.max_yaw_speed,
                        self.max_yaw_speed,
                    )
            elif name == "height_up":
                self.target_height = min(self.target_height + self.height_step, 0.90)
            elif name == "height_down":
                self.target_height = max(self.target_height - self.height_step, 0.65)
            elif name in EVENT_TO_FSM:
                command = EVENT_TO_FSM[name]

                if command == "gotoMULTIAMP":
                    # 一次按下/松开只计一次，忽略长按产生的重复按下。
                    # 同时支持只发送 value=2 的点击端。
                    if value not in (1, 2):
                        return False
                    if is_release or (value == 1 and previous_value == 1):
                        return False

                self.flag.fsm_state_command = command
                self.flag.fsm_event_id += 1
                return True
            else:
                print(f"[ws-console] ignored event '{name}'")

    @staticmethod
    def _clamp_axis(value: float, low: float, high: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(low, min(high, value))

    # ------------------------------------------------------------- dataframe

    def _current_state_id(self) -> int:
        if self.fsm is not None:
            try:
                # ids must match /monitor/stateslist; enum .value+1 only worked
                # while the exposed states were the first enum members
                return COMMAND_TO_STATE_ID.get(
                    "goto" + self.fsm.get_current_state().name, 0
                )
            except Exception:
                return 0
        with self.flag_mutex:
            return COMMAND_TO_STATE_ID.get(self.flag.fsm_state_command, 0)

    def _update_snapshot(self):
        rd = self.robot_data
        start = rd.whole_joint_num - rd.motor_num
        rad2deg = 180.0 / math.pi
        row = [
            float(self._current_state_id()),
            float(self._periods),
            float(self._loop_period_ms),
            float(self._process_ms),
            float(self._process_ms),
        ]
        for i in range(rd.motor_num):
            j = start + i
            row += [
                1.0,
                float(rd.q_a_s_[j]) * rad2deg,
                float(rd.q_d_s_[j]) * rad2deg,
                float(rd.q_dot_a_s_[j]),
                float(rd.tau_a_s_[j]),
                float(rd.joint_kp_s_[i]),
                float(rd.joint_kd_s_[i]),
                float(rd.temperature_a[i]),
            ]
        imu = rd.imu_data_
        row += [
            float(imu[2]) * rad2deg,
            float(imu[1]) * rad2deg,
            float(imu[0]) * rad2deg,
            float(imu[3]),
            float(imu[4]),
            float(imu[5]),
            float(imu[6]),
            float(imu[7]),
            float(imu[8]),
        ]
        row += [float(rd.walk_cmd_[0]), float(rd.walk_cmd_[1]), float(rd.walk_cmd_[2])]

        with self._snapshot_mutex:
            self._snapshot = row
            self._snapshot_dirty = True

    def _monitor_data_message(self) -> str:
        # ovinf behaviour: re-serialize only when the snapshot changed
        with self._snapshot_mutex:
            if self._snapshot_dirty:
                self._monitor_str = json.dumps({"data": self._snapshot})
                self._snapshot_dirty = False
            inner = self._monitor_str
        return json.dumps({"type": "monitor_data", "data": inner})

    def _echo_events_message(self) -> str:
        with self.flag_mutex:
            events = [
                {
                    "name": "set_vel_x",
                    "value": _double_to_bits(self.flag.x_speed_command),
                },
                {
                    "name": "set_vel_y",
                    "value": _double_to_bits(self.flag.y_speed_command),
                },
                {
                    "name": "set_vel_w",
                    "value": _double_to_bits(self.flag.yaw_speed_command),
                },
            ]
        inner = json.dumps({"events": events})
        return json.dumps({"type": "events", "data": inner})

    # ----------------------------------------------------------- http server

    @staticmethod
    def _cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = (
            "origin, content-type, accept, x-requested-with"
        )
        response.headers["Access-Control-Max-Age"] = "3600"
        return response

    async def _handle_headers(self, request):
        return self._cors(
            web.Response(text=self._headers_json, content_type="application/json")
        )

    async def _handle_stateslist(self, request):
        return self._cors(
            web.Response(text=self._states_json, content_type="application/json")
        )

    async def _handle_control_get(self, request):
        return self._cors(
            web.Response(
                text=json.dumps(CONTROL_SETTINGS), content_type="application/json"
            )
        )

    async def _handle_index(self, request):
        body = (
            "xMIGCS websocket console<br>"
            f"GET /monitor/headers, /monitor/stateslist, /setting/control/get<br>"
            f"WS /console"
        )
        return self._cors(web.Response(text=body, content_type="text/html"))

    # ---------------------------------------------------------- ws handling

    async def _handle_console(self, request):
        ws = web.WebSocketResponse(heartbeat=15)
        await ws.prepare(request)
        print(f"[ws-console] client connected: {request.remote}")
        echo_task = asyncio.ensure_future(self._echo_loop(ws))
        try:
            async for msg in ws:
                if msg.type not in (WSMsgType.TEXT, WSMsgType.BINARY):
                    continue
                try:
                    text = (
                        msg.data.decode("utf-8", "ignore")
                        if isinstance(msg.data, bytes)
                        else msg.data
                    )
                    envelope = json.loads(text)
                    mtype = envelope.get("type", "")
                    if mtype == "request_data":
                        await ws.send_str(self._monitor_data_message())
                    elif mtype == "events":
                        inner = json.loads(envelope.get("data", "") or "{}")
                        for ev in inner.get("events", []):
                            self.events_queue.put(
                                (str(ev.get("name", "")), ev.get("value", 0))
                            )
                    # anything else: ignored, like the ovinf backend
                except (ValueError, TypeError) as e:
                    print(f"[ws-console] bad message: {e}")
        finally:
            echo_task.cancel()
            print(f"[ws-console] client gone: {request.remote}")
        return ws

    async def _echo_loop(self, ws):
        """Push current velocity commands to the client as coded events."""
        try:
            while True:
                await asyncio.sleep(self.echo_interval)
                # await ws.send_str(self._echo_events_message())
        except (asyncio.CancelledError, ConnectionError, RuntimeError):
            pass

    # ----------------------------------------------------------- server loop

    def _server_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/monitor/headers", self._handle_headers)
        app.router.add_get("/monitor/stateslist", self._handle_stateslist)
        app.router.add_get("/setting/control/get", self._handle_control_get)
        app.router.add_get("/console", self._handle_console)
        try:
            self._runner = web.AppRunner(app)
            self._loop.run_until_complete(self._runner.setup())
            site = web.TCPSite(self._runner, self.host, self.port)
            self._loop.run_until_complete(site.start())
            print(f"[ws-console] serving on {self.host}:{self.port}")
            self._loop.run_forever()
        except Exception as e:
            print(f"[ws-console] server error: {e}")
        finally:
            if self._runner is not None:
                self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()
            print("[ws-console] server stopped")
