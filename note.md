# How to Config Dex 3.0 Robot

## Setup venv

Create a venv in xmigcs project folder.

```
(.venv) ubuntu@ubuntu:~/dknt/xmigcs_dex3$ printf '/opt/ros/humble/lib/python3.10/site-packages\n/opt/ros/jazzy/local/lib/python3.12/dist-packages\n'     > .venv/lib/python3.12/site-packages/ros_paths.pth # 暴露 rclpy/bodyctrl_msgs

(.venv) ubuntu@ubuntu:~/dknt/xmigcs_dex3$ .venv/bin/python -m pip install -r requirements_24.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

(.venv) ubuntu@ubuntu:~/dknt/xmigcs_dex3$ .venv/bin/python -m pip install lib/24/sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl --no-deps
```


## Remote Window Forwarding

Setup x11 forwarding.

```
sudo apt install -y xauth libqt5widgets5 libqt5gui5 libqt5core5a libqt5opengl5 \
  libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0 libgl1 libgl1-mesa-dri
```

Connect via ssh:

```
ssh -XY ubuntu@192.168.41.1
```

## Update Camera Firmware

Setup x11 forward first.

Then install OrbbecViewer on Orin.

```
mkdir -p ~/orbbec_viewer && cd ~/orbbec_viewer
wget https://github.com/orbbec/OrbbecSDK_v2/releases/download/v2.9.3/OrbbecViewer_v2.9.3_202607160623_b7e38ef_linux_arm64.tar.gz
tar xf OrbbecViewer_v2.9.3_*.tar.gz
find . -type f -name "OrbbecViewer"    # 记下二进制的实际路径
```

Connect with x11 forwarding.

