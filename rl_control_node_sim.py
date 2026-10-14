import os
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
import yaml

import signal
from rl_control_node import XMIGCSControlNode
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class XMIGCSControlNode_sim(XMIGCSControlNode):
    def __init__(self, debug=False):
        # # sim=True 经基类构造函数在控制线程启动前生效: 跳过真机并联踝关节
        # # 求解(MuJoCo 模型踝关节为串联表达, 见 CLAUDE.md 坑 6)
        # super().__init__(debug, sim=True)
        super().__init__(debug)
        print("rewrite sim")
        self.robot_interface.sim = True

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = None
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        node = XMIGCSControlNode_sim(debug=False)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # SIGTERM 关闭 context 后, spin 中未完成的订阅 take 会抛
        # "Unable to convert call argument"; 上下文仍在则属真实错误, 放行
        if rclpy.ok():
            raise
    finally:
        try:
            node.destroy_node()
        except Exception as e:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
