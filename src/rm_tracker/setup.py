from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rm_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament 资源索引
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # 包的元信息
        ('share/' + package_name, ['package.xml']),

        # 安装 launch 文件 (到 install/rm_tracker/share/rm_tracker/launch/)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),

        # 安装 config 文件 (到 install/rm_tracker/share/rm_tracker/config/)
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kielas',
    maintainer_email='c1470759@outlook.com',
    description='Armor tracker node (IMM-3D + Ballistics)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 映射可执行节点名 -> Python 主函数
            'rm_tracker_node = rm_tracker.node:main',
        ],
    },
)
