from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'gnn_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[
        'gnn_inference',         # ROS node logic
        'gnn_inference.*',
        'gnn_modules',           # Your GNN code
        'gnn_modules.*'
    ]),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', ['config/config.yml']),
        ('share/' + package_name + '/weights', ['weights/graph_based_detector.pt']),
    ],
    install_requires=['setuptools','rclpy', 'torch', 'gnn_interfaces'],
    zip_safe=True,
    maintainer='flw-6gem-dev',
    maintainer_email='flw-6gem-dev@todo.todo',
    description='GNN model inference node that consumes GraphData messages and outputs predictions',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = gnn_inference.inference_node:main',
        ],
    },
)
