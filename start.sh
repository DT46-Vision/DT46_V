# 1. 定义工作空间绝对路径 (这样无论在哪个目录下运行该脚本都能成功)
WS_PATH="/home/kie-dt46/DT46_V"

# 2. 加载系统环境
source /opt/ros/humble/setup.bash

# 3. 加载工作空间环境 (替代了 si 命令)
if [ -f "$WS_PATH/install/setup.bash" ]; then
    source "$WS_PATH/install/setup.bash"
else
    echo "错误: 未找到 $WS_PATH/install/setup.bash"
    exit 1
fi

# 4. 显式指定显示器，确保 rqt 能弹出
export DISPLAY=:0

# 5. 直接执行启动指令 (替代了你的 rmhik 命令)
# ros2 launch rm_vision_bringup hik.launch.py
