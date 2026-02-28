#!/bin/bash
# watch_dog.sh
TIMEOUT=3  # 设定超时时间为10秒
NAMESPACE="" # 命名空间 例如 "/infantry_3" 注意要有"/"
NODE_NAMES=("rm_detector" "rm_tracker" "usb_camera")  # 列出所有需要监控的节点名称，注意是用空格分隔
USER="$(whoami)" #用户名
HOME_DIR=$(eval echo ~$USER)
WORKING_DIR="$HOME_DIR/DT46_VISION" # 代码目录
LAUNCH_FILE="rm_vision_bringup usb_nav.launch.py" # launch 文件
OUTPUT_FILE="$WORKING_DIR/screen.output" # 终端输出记录文件

declare -A NODE_PACKAGE=(
    ["rm_detector"]="rm_detector"
    ["rm_tracker"]="rm_tracker"
    ["rm_serial_python"]="rm_serial_python"
    ["usb_camera"]="usb_camera"
    ["hik_camera"]="hik_camera"
    ["mindvision_camera"]="mindvision_camera"
)

declare -A NODE_PACKAGE_WAY=(
    ["rm_detector"]="detector.launch.py"
    ["rm_tracker"]="tracker.launch.py"
    ["rm_serial_python"]="rm_serial.launch.py" 
    ["usb_camera"]="usb_camera.launch.py"
    ["hik_camera"]="hik_camera.launch.py"
    ["mindvision_camera"]="mv_launch.py"
)

declare -A NODE_Heartbeat=(
    ["rm_detector"]="/detector/heartbeat"
    ["rm_tracker"]="/tracker/heartbeat"
    ["rm_serial_python"]="/serial/heartbeat"
    ["usb_camera"]="/usb/heartbeat"
    ["hik_camera"]="/hik/heartbeat"
    ["mindvision_camera"]="/camera_info"
)

declare -A NODE_Last_time=(
    ["rm_detector"]="rm_detector_lasttime"
    ["rm_tracker"]="rm_tracker_lasttime"
    ["rm_serial_python"]="rm_serial_python_lasttime"
    ["usb_camera"]="ros2_usb_camera_lasttime"
    ["launch hik_camera"]="launch hik_camera_lasttime"
    ["launch mindvision_camera"]="launch mindvision_camera_lasttime"
)
for node in "${NODE_NAMES[@]}"; do
    NODE_Last_time["$node"]=0  # 键：节点名，值：初始时间戳0
done

rmw="rmw_fastrtps_cpp" #RMW
export RMW_IMPLEMENTATION="$rmw" # RMW实现

export ROS_HOSTNAME=$(hostname)
export ROS_HOME=${ROS_HOME:=$HOME_DIR/.ros}
export ROS_LOG_DIR="/tmp"

source /opt/ros/humble/setup.bash
source $WORKING_DIR/install/setup.bash

rmw_config=""
if [[ "$rmw" == "rmw_fastrtps_cpp" ]]
then
  if [[ ! -z $rmw_config ]]
  then
    export FASTRTPS_DEFAULT_PROFILES_FILE=$rmw_config
  fi
elif [[ "$rmw" == "rmw_cyclonedds_cpp" ]]
then
  if [[ ! -z $rmw_config ]]
  then
    export CYCLONEDDS_URI=$rmw_config
  fi
fi

function bringup_Single() {
    local node_name=$1
    local package_name="${NODE_PACKAGE[$node_name]}"
    local package_way="${NODE_PACKAGE_WAY[$node_name]}"

    source /opt/ros/humble/setup.bash
    source $WORKING_DIR/install/setup.bash

    nohup ros2 launch "$package_name" "$package_way" > "$WORKING_DIR/${node_name}.log" 2>&1 &

}

function bringup() {
    source /opt/ros/humble/setup.bash
    source $WORKING_DIR/install/setup.bash
    nohup ros2 launch $LAUNCH_FILE > "$OUTPUT_FILE" 2>&1 &
}

function restart() {
    local node_name=$1
    echo "正在重启节点: $node_name"
    pkill -f "/$node_name$" 2>/dev/null
    # pkill -f ros  # 杀掉所有ROS2进程
    # ros2 daemon stop
    # ros2 daemon start
    bringup_Single "$node_name"
}

bringup
sleep $TIMEOUT
sleep $TIMEOUT

# 监控每个节点的心跳
while true; do
    for node in "${NODE_NAMES[@]}"; do
        topic="${NODE_Heartbeat[$node]}" #获取心跳包发送的话题
        echo "- Check $node"
        if ros2 topic list 2>/dev/null | grep -q $topic 2>/dev/null; then
            data_value=$(timeout 10 ros2 topic echo $topic --once | grep -o "heartbeat_time: [0-9]*" | awk '{print $2}' 2>/dev/null)
            if [ ! -z "$data_value" ]; then
                if [ "$data_value" != "${NODE_Last_time[$node]}" ]; then
                    echo "    $node is OK! Heartbeat Count: $data_value"
                    NODE_Last_time["$node"]=$data_value
                else
                    echo "${NODE_Last_time[$node]}"
                    echo "    aaaaa Heartbeat lost for $topic, restarting $topic nodes..."
                    restart "$node"
                    break
                fi
            else
                echo "    bbbbb Heartbeat lost for $topic, restarting $topic nodes..."
                restart "$node"
                break
            
            fi
        else
            echo "    cccccc Heartbeat topic $topic does not exist, restarting $topic nodes..."
            restart "$node"
            break
        fi
    done
    sleep $TIMEOUT
done
