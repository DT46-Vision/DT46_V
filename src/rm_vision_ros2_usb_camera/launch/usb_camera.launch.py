from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('usb_camera')

    # 参数文件路径
    params_file = os.path.join(pkg_share, 'config', 'camera_params.yaml')
    camera_info_url = 'package://usb_camera/config/camera_info.yaml'

    node = Node(
        package='usb_camera',
        executable='usb_camera_node',
        name='usb_camera',
        output='screen',
        parameters=[
            params_file,
            {
                'camera_info_url': camera_info_url
            }
        ]
    )

    return LaunchDescription([node])
