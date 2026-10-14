# xMIGCS

xMIGCS (eXtensible Motion Intelligence Group Control System) 是一个用于机器人控制的软件系统，专注于通过有限状态机（FSM）和策略模块实现对机器人的灵活控制。项目服务于运动智能领域的研究与开发，支持多种控制模式和外部输入方式（如键盘、手柄等）。

## 功能特性

- **状态机管理**: 基于 FSM 模块实现机器人行为的状态流转控制
- **多策略支持**: 提供多种控制策略（如 dh, mlp, pbhc, zero 等）
- **人机交互控制**: 支持键盘、手柄等外设进行机器人实时操控
- **配置驱动**: 使用 YAML 文件进行参数配置，支持不同场景下的快速部署
- **模块化设计**: 各策略独立封装，便于扩展与维护

## 安装与运行

### 环境要求

- 当前 `requirements.txt` 按 Python 3.12 环境使用，搭配 Ubuntu 24.04 / ROS 2 Jazzy。Python ABI 必须与本地 wheel 和 ROS 一致。
- Git (用于版本控制)
- bodyctrl_msgs 下载地址：http://10.20.0.252:5302/installation.html

### 安装步骤

```bash
cd your_project_folder
unzip xmigcs.zip
cd xmigcs/
pip install -r requirements.txt
# Ubuntu 24.04 / ROS Jazzy / Python 3.12
pip install lib/24/sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl
sudo dpkg -i lib/24/ros-jazzy-bodyctrl-msgs_0.0.0-0noble_amd64.deb
# lib/22 提供 Humble / Python 3.10 对应库；使用它时须另外匹配 Python 依赖版本。
```

复制来的 `.venv` 若仍是 Python 3.10，不能直接加载 `lib/24` 的 CPython 3.12 wheel；应在与目标 ROS 匹配的环境安装依赖。

### 3.0 踝解算与图像接口

踝转换使用 `sptlib_python.funcSPTrans`。反馈顺序保持
`set_p_est → calcFK → calcIK → get_s_state`；命令顺序保持
`set_s_des → calc_joint_pos_ref → calc_joint_tor_des → get_p_des`。
新库 getter 返回 `(success, q, dq, tau)`；失败或非有限/错误维度结果会抛出异常，当前调用不会继续下发。

`RobotData` 的 `_s_` 串联缓冲供策略和现有 WebSocket 监控读取，`_p_` 并联缓冲保存实际执行器反馈和目标，`_motor_` 缓冲保存零偏/方向转换后的下发量。转换只写相应缓冲，不覆盖串联策略目标。踝串联 PD 力矩经新库转换后，仍按原来的并联 PD 公式折算位置目标；踝下发前馈力矩保持为零，避免重复施加力矩。主循环顺序、29 关节映射及腿/臂/腰三组 ROS 话题保持原样。

`common/robot_interface.py` 中 Orbbec SDK 导入、初始化、采集及释放均已注释。图像由外部相机节点发布，配置使用 `robot_interface.depth.type: ros2` 和 `robot_interface.depth.ros2.topic`；支持 `16UC1` 毫米及 `32FC1` 米制深度。`refresh_depth()` 保留为空钩子，ROS 回调继续完成预处理。也可配置 `type: disable`。

### 运行项目

以下示例以 Humble 为例；Jazzy 环境使用对应的 `/opt/ros/jazzy/setup.bash`。

```bash
# body节点, 只有真机才需要，仿真启动xsim_mujoco就行
sudo su
source /home/ubuntu/ros2ws/install/setup.bash
source /opt/ros/humble/setup.sh
ros2 launch body_control body.launch.py

# XBOX手柄节点
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
source /opt/ros/humble/setup.bash
ros2 run joy joy_node --ros-args --remap joy:=xbox_data

# 云卓手柄节点
# 安装: https://zitd5je6f7j.feishu.cn/wiki/Y6zxwVA5WisMwskzeHdcqQpQnQx?fromScene=spaceOverview
source ~/ros2ws/install/setup.bash
ros2 launch usb_sbus usb_sbus.launch.py

# 启动主控制节点
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
source /opt/ros/humble/setup.bash
source /home/ubuntu/ros2ws/install/setup.bash

# 实机
cd xmigcs
python3 rl_control_node.py
# 仿真
python3 rl_control_node_sim.py

touch /tmp/rl_start_signal
```

## 控制器使用说明

### XBOX手柄键位映射

xMIGCS支持标准XBOX手柄控制，以下是详细键位映射关系：

#### 状态映射关系

##### 单按钮状态切换

| 按钮 | 对应状态 | 功能说明 |
|------|----------|----------|
| X | gotoZERO | 回到零位状态 |
| Y | gotoSTOP | 停止状态 |



### 云卓手柄键位映射

xMIGCS支持标准云卓手柄控制，开始使用前先确保所有键都回中，以下是详细键位映射关系：

#### 状态映射关系

##### 单按钮状态切换

| 按钮 | 对应状态 | 功能说明 |
|------|----------|----------|
| C | gotoSTOP | 停止状态 |

##### 组合按钮状态切换

| 切入策略按钮组合| 策略内使用按键 | 对应状态 | 功能说明 |
|------------|----------------|--------------|-------------|
|   所有键(拨中)   |         D      | gotoZERO     | 回到零位状态  |
|   所有键(拨中)   |         A      | gotoWALKAMP      | WALKAMP策略状态  |
|   E(上拨)   |         A      | gotoBEYONDMIMIC   | BEYONDMIMIC策略状态   |
|   E(上拨)   |         D      | gotoBEYONZERO   | BEYONDMIMIC零位状态   |
|   F(上拨)   |         无      | 手柄控制失能，只有停止键可用      | |

##### 基础运动控制

| 控制方式 | 功能 |
|----------|------|
| 左摇杆Y1轴 | 前后移动控制（正向为前进） |
| 左摇杆X1轴 | 左右移动控制 |
| 右摇杆X2轴 | 机身旋转控制 |

## 项目结构

```
.
├── FSM                 # 有限状态机模块
├── common              # 通用功能模块
├── config              # 配置文件
├── policy              # 控制策略模块
├── test                # 测试文件
└── rl_control_node.py  # 主控制节点
```

## 如何添加新的控制策略
1. 在 policy 目录下创建新的策略文件夹，例如 my_new_policy

2. 在新文件夹中创建以下文件：
   - fsm_mypolicy.py - 实现具体的FSM状态类
   - config/mypolicy.yaml - 策略配置文件（可选）
3. 在 fsm_mypolicy.py 中实现 FSMState 类：
```python
    from FSM.fsm_base import FSMState, FSMStateName, ControlFlag
    from common.robot_data import RobotData

    class FSMStateMyPolicy(FSMState):
        def __init__(self, robot_data: RobotData):
            super().__init__(robot_data)
            # 初始化策略特定变量
        
        def on_enter(self):
            # 进入状态时的初始化操作
            pass
        
        def run(self, flag: ControlFlag):
            # 策略的主要运行逻辑
            pass
        
        def on_exit(self):
            # 退出状态时的清理操作
            pass
        
        def check_transition(self, flag: ControlFlag) -> FSMStateName:
            # 检查是否需要转换到其他状态
            pass
```
1. 在 FSM/robot_fsm.py 中注册新状态：
   - 导入新策略类
   - 在 _init_states() 方法中初始化状态对象
   - 在 FSMStateName 枚举中添加新状态
2. 在控制接口（如 common/joystick.py中添加对应的状态映射
3. 更新配置文件dex_config.yaml 以支持新策略的相关参数
## 开发与贡献

欢迎对项目进行贡献，开发前请确保：

1. 遵循项目代码规范
2. 添加适当的测试用例
3. 提交前运行所有测试确保无误

## 许可证

本项目仅供内部使用。

## 项目状态

项目正在积极开发中。
