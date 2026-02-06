from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rm_serial_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kielas',
    maintainer_email='c1470759@outlook.com',
    description='串口通信驱动 (RMSerialDriver)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rm_serial_node = rm_serial_python.rm_serial_node:main',
        ],
    },
)
