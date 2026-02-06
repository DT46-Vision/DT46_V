from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 找到参数文件路径
    config = os.path.join(
        get_package_share_directory('rm_serial_python'),
        'config',
        'rm_serial_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='rm_serial_python',
            executable='rm_serial_node',   # setup.py 里 console_scripts 注册的名字
            name='rm_serial',
            output='screen',
            parameters=[config]
        )
    ])
