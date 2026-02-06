import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 获取安装后的包路径，拼接出 YAML 文件的绝对路径
    config_file_path = os.path.join(
        get_package_share_directory('rm_detector'), # 包名
        'config',                                   # 目录名 (对应 CMakeLists install 的目标)
        'detector_params.yaml'                      # 文件名
    )

    print(f"DEBUG: Loading config from: {config_file_path}")

    return LaunchDescription([
        Node(
            package='rm_detector',
            executable='rm_detector_node',
            name='rm_detector',            # 必须与 yaml 文件的根节点 rm_detector: 保持一致
            output='screen',
            emulate_tty=True,              # 更好的日志颜色显示
            parameters=[config_file_path]  # <--- 核心修改：这里必须把路径传进去
        )
    ])
