#!/usr/bin/env bash
# Manual launcher for the Tienkung sim stack. Requires the xhumanoid conda env
# active — its activation hook already sources ROS, sets ROS_DOMAIN_ID and
# auto-detects DISPLAY/XAUTHORITY for the MuJoCo viewer.
# Installed as `tiensim` by scripts/setup_conda_env.sh.
set -e
XH=/home/dknt/Project/xhumanoid

case "${1:-}" in
  sim)  # MuJoCo simulator (tg3_test stairs course, 64x36 depth = training cam)
    cd "$XH/xSIM_MUJOCO"
    exec python -u scripts/simulator_decoupled.py -m tg3_test --depth-width 64 --depth-height 36
    ;;
  ctrl)  # FSM control node (sim mode)
    cd "$XH/Deploy_Tienkung"
    exec python3 rl_control_node_sim.py
    ;;
  joy)  # xbox pad bridge
    exec ros2 run joy joy_node --ros-args --remap joy:=xbox_data
    ;;
  stack)  # full 3-pane tmux session (sim + ctrl + joy)
    tmux kill-session -t xhumanoid 2>/dev/null || true
    tmux new-session -d -s xhumanoid -x 220 -y 55
    tmux split-window -v -t xhumanoid:0.0
    tmux split-window -h -t xhumanoid:0.1
    tmux resize-pane -t xhumanoid:0.0 -y 36
    tmux select-pane -t xhumanoid:0.0
    tmux send-keys -t xhumanoid:0.1 -l "source ~/Software/miniconda3/etc/profile.d/conda.sh && conda activate xhumanoid && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY && cd $XH/xSIM_MUJOCO && python -u scripts/simulator_decoupled.py -m tg3_test --depth-width 64 --depth-height 36"
    tmux send-keys -t xhumanoid:0.1 Enter
    sleep 3
    tmux send-keys -t xhumanoid:0.0 -l "source ~/Software/miniconda3/etc/profile.d/conda.sh && conda activate xhumanoid && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID && cd $XH/Deploy_Tienkung && python3 rl_control_node_sim.py"
    tmux send-keys -t xhumanoid:0.0 Enter
    tmux send-keys -t xhumanoid:0.2 -l "source ~/Software/miniconda3/etc/profile.d/conda.sh && conda activate xhumanoid && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID && ros2 run joy joy_node --ros-args --remap joy:=xbox_data"
    tmux send-keys -t xhumanoid:0.2 Enter
    echo "stack running in tmux session 'xhumanoid' — attach with: tmux attach -t xhumanoid"
    echo "press ESC in the MuJoCo window to release the elastic band, then pad X -> RB -> left stick"
    ;;
  stop)
    tmux kill-session -t xhumanoid 2>/dev/null || true
    echo "stopped"
    ;;
  *)
    echo "usage: tiensim {sim|ctrl|joy|stack|stop}"
    echo "  sim    MuJoCo simulator (tg3_test, 64x36 depth) in this terminal"
    echo "  ctrl   rl_control_node_sim.py FSM node in this terminal"
    echo "  joy    joy_node xbox bridge in this terminal"
    echo "  stack  all three in tmux session 'xhumanoid'"
    echo "  stop   kill the tmux session"
    exit 1
    ;;
esac
