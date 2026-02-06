from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 找到参数文件路径 (安装后的 share/rm_tracker/config/tracker_params.yaml)
    config = os.path.join(
        get_package_share_directory('rm_tracker'),
        'config',
        'tracker_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='rm_tracker',
            executable='rm_tracker_node',   # setup.py 里注册的入口
            name='rm_tracker',              # 节点名
            output='screen',                # 输出到终端
            parameters=[config]             # 加载参数文件
        )
    ])
