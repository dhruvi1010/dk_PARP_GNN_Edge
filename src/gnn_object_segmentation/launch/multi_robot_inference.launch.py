from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition


def generate_launch_description():
    return LaunchDescription([
        # --- Launch Arguments ---
        DeclareLaunchArgument(
            'ep03_bag',
            default_value='/home/asfy/flw/Robo_FUSE_Dataset/CPPS_Vertical/Robot_1/rosbag/20250219_171846/20250219_171846_0.db3',
            #default_value='/home/flw-6gem-dev/dev/RoboFUSE_Dataset/CPPS_Static_Scenario/CPPS_Horizontal/Robot_1/rosbag_logs/20250221_120046_v2/20250221_120046_0.db3',
            description='Rosbag path for ep03 (Robot 1)'
        ),
        DeclareLaunchArgument(
            'ep05_bag',
            default_value='/home/asfy/flw/Robo_FUSE_Dataset/CPPS_Vertical/Robot_2/rosbag/20250219_171851/20250219_171851_0.db3',
            #default_value='/home/flw-6gem-dev/dev/RoboFUSE_Dataset/CPPS_Static_Scenario/CPPS_Horizontal/Robot_2/rosbag_logs/20250221_120054_v2/20250221_120054_0.db3',
            description='Rosbag path for ep05 (Robot 2)'
        ),
        DeclareLaunchArgument(
            'visualize',
            default_value='false',
            description='Enable RViz2 radar and robot position visualization'
        ),
        DeclareLaunchArgument(
            'use_rosbags',
            default_value='false',
            description='Play rosbag files if true, else assume live data'
        ),
        DeclareLaunchArgument(
            'simulation',
            default_value='false',
            description='Play rosbag files if true, else assume live data'
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value='src/gnn_object_segmentation/rviz/flw_hall_gnn.rviz',
            description='Path to RViz config file'
        ),

        # --- Rosbag Playback (conditionally launched) ---
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', LaunchConfiguration('ep03_bag'), '--rate', '1.0', '--clock'],
            condition=IfCondition(LaunchConfiguration('use_rosbags')),
            output='screen'
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', LaunchConfiguration('ep05_bag'), '--rate', '1.0', '--clock'],
            condition=IfCondition(LaunchConfiguration('use_rosbags')),
            output='screen'
        ),

        # --- Data Merger Node (with or without visualization) ---
        Node(
            package='gnn_object_segmentation',
            executable='data_merge',
            name='data_merge',
            output='screen',
            arguments=[
                '--visualize', LaunchConfiguration('visualize'),
                '--simulation', LaunchConfiguration('simulation')
            ],
            condition=IfCondition(LaunchConfiguration('visualize'))
        ),
        Node(
            package='gnn_object_segmentation',
            executable='data_merge',
            name='data_merge',
            output='screen',
            arguments=[
                '--visualize', LaunchConfiguration('visualize'),
                '--simulation', LaunchConfiguration('simulation')
            ],
            condition=IfCondition(PythonExpression(['not ', LaunchConfiguration('visualize')]))
        ),

        # --- RViz2 (optional) ---
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('visualize'))
        ),

        Node(
            package='gnn_object_segmentation',
            executable='arena_marker_node',
            name='arena_marker_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('visualize'))
        ),

        # --- Octomap (optional) ---
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[{
                'frame_id': 'map',
                'resolution': 0.1,
                'sensor_model_max_range': 10.0,
            }],
            remappings=[
                ('cloud_in', '/octomap_input_points')
            ]
        ),
    ])
