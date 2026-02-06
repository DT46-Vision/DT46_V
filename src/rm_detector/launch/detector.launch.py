from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rm_detector',
            executable='rm_detector_node',
            name='rm_detector',
            output='screen',
        )
    ])
