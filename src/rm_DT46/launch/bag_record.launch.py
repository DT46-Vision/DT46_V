import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ---------------- 路径配置 ----------------
    # 默认 Bag 路径
    default_bag_path = os.path.expanduser("~/ros2_img_msg_to_mp4/autoaim_data_bag")

    # 算法配置文件路径
    detector_params_file = os.path.join(
        get_package_share_directory("rm_detector"), "config", "detector_params.yaml"
    )
    tracker_params_file = os.path.join(
        get_package_share_directory("rm_tracker"), "config", "tracker_params.yaml"
    )
    perspective_file = os.path.expanduser(
        "Kielas_Vision.perspective"
    )

    # ---------------- Launch 配置变量 ----------------
    bag_file = LaunchConfiguration('bag_file')

    return LaunchDescription([
        # ----------- 1. 参数声明 -----------
        # 允许命令行动态指定 bag 路径: ros2 launch xxx.py bag_file:=/path/to/new_bag
        DeclareLaunchArgument(
            'bag_file',
            default_value=default_bag_path,
            description='Path to the rosbag file to play'
        ),

        DeclareLaunchArgument(name="detector_params_file", default_value=detector_params_file),
        DeclareLaunchArgument(name="tracker_params_file", default_value=tracker_params_file),

        # ----------- 2. 自动播放 Bag (模拟数据源) -----------
        # --loop: 循环播放，方便反复调试算法
        # --clock: 发布模拟时钟 (如果算法依赖仿真时间需加上，实物录制的通常不需要)
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_file, '--loop'],
            output='screen'
        ),

        # ----------- 3. 启动装甲板检测节点 (接收 Bag 中的图像) -----------
        Node(
            package="rm_detector",
            executable="rm_detector_node",
            name="rm_detector",
            output="screen",
            emulate_tty=True,
            parameters=[LaunchConfiguration("detector_params_file")],
        ),

        # # ----------- 4. 启动装甲板追踪节点 (接收识别结果 + Bag 中的 IMU) -----------
        # Node(
        #     package="rm_tracker",
        #     executable="rm_tracker_node",
        #     name="rm_tracker",
        #     output="screen",
        #     emulate_tty=True,
        #     parameters=[LaunchConfiguration("tracker_params_file")],
        # ),

        # ----------- 5. 启动 RQT -----------
        Node(
            package="rqt_gui",
            executable="rqt_gui",
            name="rqt_gui",
            arguments=["--perspective-file", perspective_file],
            output="screen",
        ),
    ])
