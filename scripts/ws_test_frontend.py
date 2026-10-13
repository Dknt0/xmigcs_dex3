#!/usr/bin/env python3
"""
Naive python frontend for the xMIGCS websocket console.

Implements the client side of the Bitbot-Android / bitbot-ovinf protocol:
  1. GET /monitor/headers, /monitor/stateslist, /setting/control/get
  2. WS /console: poll {"type":"request_data"} at 10 Hz, print monitor data;
     send {"type":"events"} for FSM commands and velocities (double bit-cast
     to int64 in the value field); print server-pushed events.

Modes:
  demo        scripted sequence with PASS/FAIL checks (default)
  interactive keyboard control (see help printed on start)

Usage (from Deploy_Tienkung, any conda env with aiohttp):
    python3 scripts/ws_test_frontend.py [--host 127.0.0.1] [--port 12888] [--mode demo]
"""
import argparse
import asyncio
import json
import struct
import urllib.request

import aiohttp

POLL_INTERVAL = 0.1


def double_to_bits(value: float) -> int:
    return struct.unpack('<q', struct.pack('<d', float(value)))[0]


def bits_to_double(value: int) -> float:
    return struct.unpack('<d', struct.pack('<q', int(value)))[0]


def http_get_json(host, port, path):
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=5) as r:
        return json.loads(r.read().decode())


def envelope(mtype: str, inner) -> str:
    """Double-encoded message exactly like the app sends it."""
    return json.dumps({"type": mtype, "data": json.dumps(inner)})


def events_msg(events) -> str:
    return envelope("events", {"events": events})


def button(name, value=2):
    return {"name": name, "value": value}


def velocity(name, value):
    return {"name": name, "value": double_to_bits(value)}


class Console:
    """Collects the latest monitor frame and server events."""

    def __init__(self, headers):
        self.headers = headers
        self.state_names = {}
        self.data = []
        self.last_events = None
        self.new_data = asyncio.Event()
        # map device/header name -> flat-array offset
        self.offsets = {}
        offset = len(headers.get("kernel", []))
        for dev in headers.get("bus", {}).get("devices", []):
            for h in dev["headers"]:
                self.offsets[f"{dev['name']}.{h}"] = offset
                offset += 1

    def value(self, key):
        off = self.offsets.get(key)
        if off is None or off >= len(self.data):
            return None
        return self.data[off]

    async def wait_state(self, state_id, timeout):
        """Wait until the kernel state column equals state_id."""
        try:
            await asyncio.wait_for(self._wait_state_helper(state_id), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _wait_state_helper(self, state_id):
        while True:
            if len(self.data) > 0 and int(self.data[0]) == state_id:
                return
            await self._wait_new_data()

    async def _wait_new_data(self):
        self.new_data.clear()
        await self.new_data.wait()


async def poller(ws, console):
    while True:
        await ws.send_str(envelope("request_data", ""))
        await asyncio.sleep(POLL_INTERVAL)


async def receiver(ws, console, verbose):
    async for msg in ws:
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        try:
            outer = json.loads(msg.data)
        except ValueError:
            continue
        mtype = outer.get("type", "")
        if mtype == "monitor_data":
            console.data = json.loads(outer.get("data", "") or '{"data":[]}').get("data", [])
            console.new_data.set()
            if verbose:
                state = console.data[0] if console.data else -1
                j = console.value("l_hip_pitch.actual_position")
                roll = console.value("imu.roll")
                vx = console.value("walk_cmd.x_speed")
                print(f"  [monitor] state={console.state_names.get(state, state)} "
                      f"l_hip_pitch={j:.3f} imu_roll={roll:.3f} cmd_vx={vx:+.2f}")
        elif mtype == "events":
            inner = json.loads(outer.get("data", "") or '{"events":[]}')
            console.last_events = inner.get("events", [])
            if verbose:
                desc = ", ".join(
                    f"{e['name']}={bits_to_double(e['value']):+.3f}"
                    if e["value"] not in (1, 2) else f"{e['name']}={e['value']}"
                    for e in console.last_events)
                print(f"  [server events] {desc}")


async def run_demo(host, port):
    headers = http_get_json(host, port, "/monitor/headers")
    states = http_get_json(host, port, "/monitor/stateslist")
    settings = http_get_json(host, port, "/setting/control/get")
    console = Console(headers)
    console.state_names = {s["id"]: s["name"] for s in states["states"]}
    devices = headers["bus"]["devices"]
    print(f"headers: {len(headers['kernel'])} kernel cols, {len(devices)} devices "
          f"({devices[0]['name']}..{devices[-1]['name']}), extra={headers['extra']}")
    print(f"states:  {console.state_names}")
    print(f"control: {[(s['event'], s['kb_key']) for s in settings]}")
    kb = {s["event"]: s["kb_key"] for s in settings}
    ovinf_ok = (kb.get("init_pose") == "p" and kb.get("stop") == " "
                and kb.get("enable_standing_policy") == "1"
                and kb.get("enable_warking_policy") == "2"
                and kb.get("enable_robust_policy") == "3"
                and kb.get("velo_x_increase") == "i")

    id_by_name = {n: i for i, n in console.state_names.items()}
    results = []

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://{host}:{port}/console") as ws:
            rx = asyncio.ensure_future(receiver(ws, console, verbose=False))
            px = asyncio.ensure_future(poller(ws, console))

            def check(label, ok):
                results.append((label, ok))
                print(f"{'PASS' if ok else 'FAIL'}: {label}")

            await asyncio.sleep(1.0)
            check("ovinf kb bindings (p/space/1/2/3/i...)", ovinf_ok)
            check("monitor data flowing", len(console.data) ==
                  len(headers["kernel"]) + sum(len(d["headers"]) for d in devices))

            print("\n-- demo: init_pose (gotoZERO)")
            await ws.send_str(events_msg([button("init_pose")]))
            check("state -> fsm_zero", await console.wait_state(id_by_name["fsm_zero"], 5))

            print("-- demo: enable_warking_policy (gotoWALKAMP)")
            await ws.send_str(events_msg([button("enable_warking_policy")]))
            check("state -> fsm_walkamp", await console.wait_state(id_by_name["fsm_walkamp"], 5))

            print("-- demo: set_vel_x 0.2 m/s (joystick method)")
            await ws.send_str(events_msg([velocity("set_vel_x", 0.2)]))
            ok = False
            for _ in range(30):
                await asyncio.sleep(POLL_INTERVAL)
                v = console.value("walk_cmd.x_speed")
                if v is not None and abs(v - 0.2) < 0.05:
                    ok = True
                    break
            check("walk_cmd.x_speed ~= 0.2", ok)

            print("-- demo: velo_x_increase x3 (keyboard method, +0.05 each)")
            await ws.send_str(events_msg([velocity("set_vel_x", 0.0)]))
            await asyncio.sleep(0.3)
            for _ in range(3):
                await ws.send_str(events_msg([button("velo_x_increase")]))
                await asyncio.sleep(0.2)
            ok = False
            for _ in range(30):
                await asyncio.sleep(POLL_INTERVAL)
                v = console.value("walk_cmd.x_speed")
                if v is not None and abs(v - 0.15) < 0.01:
                    ok = True
                    break
            check("walk_cmd.x_speed ~= 0.15 after 3 nudges", ok)

            print("-- demo: press/release pair counts once (1 then 2)")
            await ws.send_str(events_msg([{"name": "velo_x_increase", "value": 1}]))
            await asyncio.sleep(0.2)
            await ws.send_str(events_msg([{"name": "velo_x_increase", "value": 2}]))
            await asyncio.sleep(0.3)
            v = console.value("walk_cmd.x_speed")
            check(f"pair applied once (got {v})", v is not None and abs(v - 0.20) < 0.01)

            print("-- demo: set_vel_x 99 (must clamp to max_x_plus)")
            await ws.send_str(events_msg([velocity("set_vel_x", 99.0)]))
            await asyncio.sleep(0.5)
            v = console.value("walk_cmd.x_speed")
            check(f"x_speed clamped (got {v})", v is not None and 0.5 < v <= 1.0)

            print("-- demo: stop")
            await ws.send_str(events_msg([button("stop"), velocity("set_vel_x", 0.0)]))
            check("state -> fsm_stop", await console.wait_state(id_by_name["fsm_stop"], 5))

            print("-- demo: waiting for server velocity echo events")
            console.last_events = None  # ignore echoes from before the stop
            for _ in range(20):
                await asyncio.sleep(POLL_INTERVAL)
                if console.last_events:
                    break
            echo = {e["name"]: bits_to_double(e["value"]) for e in (console.last_events or [])}
            check(f"server echoed velocities {echo}",
                  "set_vel_x" in echo and echo["set_vel_x"] == 0.0)

            px.cancel()
            rx.cancel()
            await asyncio.gather(px, rx, return_exceptions=True)

    print("\n==== demo summary ====")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return all(ok for _, ok in results)


HELP_KEYS = """
ovinf-style keys (press Enter after the key):
  p = init_pose(ZERO)   <space> = stop   r = run_policy(WALKAMP)   8 = start
  1 = standing policy   2 = warking policy   3 = robust policy
  i/k = vx +/-0.05      j/l = vy +/-0.05      u/o = vyaw +/-0.05   (nudges)
  0 = zero all velocities    q = quit
"""


async def run_interactive(host, port):
    headers = http_get_json(host, port, "/monitor/headers")
    states = http_get_json(host, port, "/monitor/stateslist")
    console = Console(headers)
    console.state_names = {s["id"]: s["name"] for s in states["states"]}

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"ws://{host}:{port}/console") as ws:
            rx = asyncio.ensure_future(receiver(ws, console, verbose=True))
            px = asyncio.ensure_future(poller(ws, console))
            loop = asyncio.get_event_loop()
            print(HELP_KEYS)
            nudges = {'i': "velo_x_increase", 'k': "velo_x_decrease",
                      'j': "velo_y_increase", 'l': "velo_y_decrease",
                      'u': "velo_w_increase", 'o': "velo_w_decrease"}
            buttons = {'p': "init_pose", ' ': "stop", 'r': "run_policy",
                       '8': "start", '1': "enable_standing_policy",
                       '2': "enable_warking_policy", '3': "enable_robust_policy"}
            while True:
                key = await loop.run_in_executor(None, input)
                key = key[:1]  # a bare Enter means stop (" " binding); keep one char
                if key == 'q':
                    break
                if key in nudges:
                    await ws.send_str(events_msg([button(nudges[key])]))
                elif key == '0':
                    await ws.send_str(events_msg(
                        [velocity("set_vel_x", 0), velocity("set_vel_y", 0),
                         velocity("set_vel_w", 0)]))
                    print("vel = [0.0, 0.0, 0.0]")
                elif key in buttons:
                    await ws.send_str(events_msg([button(buttons[key])]))
            px.cancel()
            rx.cancel()
            await asyncio.gather(px, rx, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=12888)
    parser.add_argument('--mode', choices=['demo', 'interactive'], default='demo')
    args = parser.parse_args()

    ok = asyncio.run(run_demo(args.host, args.port)
                     if args.mode == 'demo' else run_interactive(args.host, args.port))
    raise SystemExit(0 if ok is not False else 1)


if __name__ == '__main__':
    main()
