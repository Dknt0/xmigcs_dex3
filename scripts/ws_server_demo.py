#!/usr/bin/env python3
"""
Standalone websocket-console demo server with a fake robot.

Runs common/websocket_control.py against a synthetic RobotData (sine waves)
without ROS or the simulator — for testing the mobile app or the test
frontend pipeline on their own.

Usage (from Deploy_Tienkung, any conda env with aiohttp):
    python3 scripts/ws_server_demo.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.robot_data import RobotData
from common.websocket_control import WebsocketController, COMMAND_TO_STATE_ID


def main():
    rd = RobotData()
    ctrl = WebsocketController(rd, robot_fsm=None)  # no FSM: state follows commands
    ctrl.init()

    start = rd.whole_joint_num - rd.motor_num
    t0 = time.time()
    period = 0.01
    print("demo server running (Ctrl-C to quit)")
    try:
        while True:
            loop_start = time.time()
            t = time.time() - t0
            # fake robot: every joint oscillates, imu rolls, commands echo
            for i in range(rd.motor_num):
                rd.q_a_[start + i] = 0.1 * math.sin(t + i * 0.3)
                rd.q_d_[start + i] = 0.1 * math.sin(t + i * 0.3 + 0.5)
                rd.q_dot_a_[start + i] = 0.1 * math.cos(t + i * 0.3)
                rd.tau_a_[start + i] = 5.0 * math.sin(t + i * 0.3)
                rd.tau_d_[start + i] = 5.0 * math.sin(t + i * 0.3 + 0.5)
            rd.imu_data_[0] = 0.1 * math.sin(t)          # yaw
            rd.imu_data_[1] = 0.05 * math.sin(t * 0.7)   # pitch
            rd.imu_data_[2] = 0.15 * math.sin(t * 0.5)   # roll
            rd.imu_data_[3] = 0.2 * math.cos(t)
            rd.imu_data_[6] = 0.1

            ctrl.update_flag()
            # pretend the "robot" accepted the velocity command
            flag = ctrl.get_ws_flag()
            rd.walk_cmd_[:] = [flag.x_speed_command, flag.y_speed_command,
                               flag.yaw_speed_command]
            rd.time_now_ = t

            # 1 Hz status line
            if ctrl._periods % 100 == 0:
                state = COMMAND_TO_STATE_ID.get(flag.fsm_state_command, 0)
                print(f"t={t:6.1f}s cmd={flag.fsm_state_command:14s} "
                      f"state={state} vel=({flag.x_speed_command:+.2f},"
                      f"{flag.y_speed_command:+.2f},{flag.yaw_speed_command:+.2f})")

            sleep = period - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == '__main__':
    main()
