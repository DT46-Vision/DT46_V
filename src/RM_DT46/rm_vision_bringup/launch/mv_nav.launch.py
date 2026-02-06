import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ---------------- 相机参数 ----------------
    params_file = os.path.join(
        get_package_share_directory("mindvision_camera"), "config", "camera_params.yaml"
    )
    camera_info_url = "package://mindvision_camera/config/camera_info.yaml"

    # ---------------- 装甲板检测参数 ----------------
    detector_params_file = os.path.join(
        get_package_share_directory("rm_detector"), "config", "detector_params.yaml"
    )

    # ---------------- 装甲板追踪参数 ----------------
    tracker_params_file = os.path.join(
        get_package_share_directory("rm_tracker"), "config", "tracker_params.yaml"
    )

    # ---------------- rqt 界面配置 ----------------
    perspective_file = os.path.expanduser(
        "~/ros_vision/Kielas_Vision.perspective"
    )

    return LaunchDescription([
        # ----------- 可配置的 launch 参数 -----------
        DeclareLaunchArgument(name="params_file", default_value=params_file),
        DeclareLaunchArgument(name="camera_info_url", default_value=camera_info_url),
        DeclareLaunchArgument(name="use_sensor_data_qos", default_value="false"),
        DeclareLaunchArgument(name="detector_params_file", default_value=detector_params_file),
        DeclareLaunchArgument(name="tracker_params_file", default_value=tracker_params_file),

        # ----------- 启动装甲板检测节点 -----------
        Node(
            package="rm_detector",
            executable="rm_detector_node",
            name="rm_detector",
            output="screen",
            emulate_tty=True,
            parameters=[LaunchConfiguration("detector_params_file")],
        ),

        # ----------- 启动装甲板追踪节点 -----------
        Node(
            package="rm_tracker",
            executable="rm_tracker_node",
            name="rm_tracker",
            output="screen",
            emulate_tty=True,
            parameters=[LaunchConfiguration("tracker_params_file")],
        ),

        # ----------- 启动 MV 相机节点 -----------
        Node(
            package="mindvision_camera",
            executable="mindvision_camera_node",
            name="mv_camera",
            output="screen",
            emulate_tty=True,
            parameters=[
                LaunchConfiguration("params_file"),
                {
                    "camera_info_url": LaunchConfiguration("camera_info_url"),
                    "use_sensor_data_qos": LaunchConfiguration("use_sensor_data_qos"),
                },
            ],
        ),

        # ----------- 启动 rqt 并加载布局 -----------
        Node(
            package="rqt_gui",
            executable="rqt_gui",
            name="rqt_gui",
            arguments=["--perspective-file", perspective_file],
            output="screen",
        ),

    ])
