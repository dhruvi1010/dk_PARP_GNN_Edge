from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch.actions import ExecuteProcess
import datetime
import os


def generate_launch_description():
    # Generate timestamp at launch time
    run_id = LaunchConfiguration('run_id')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    bag_dir = PathJoinSubstitution([
        TextSubstitution(text='datalogging/rosbags/'),
        run_id,
        TextSubstitution(text=f'_{timestamp}_bag')
    ])
    return LaunchDescription([
        # --- Launch Arguments ---
        DeclareLaunchArgument(
            'visualize',
            default_value='False',
            description='Enable RViz2 radar and robot position visualization'
        ),
        DeclareLaunchArgument(
            'simulation',
            default_value='False',
            description='Use simulation mode (affects transform handling)'
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value='src/gnn_object_segmentation/rviz/flw_hall_gnn.rviz',
            description='Path to RViz config file'
        ),
        DeclareLaunchArgument(
            'run_id',
            default_value='default_run',
            description='Run ID for log correlation'
        ),

        # --- rosbag record 🟢 ---
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record', '-o', bag_dir,
                'clock',
                '/rm03/ti_mmwave/radar_scan_pcl', '/rm04/ti_mmwave/radar_scan_pcl',
                '/tf', '/tf_static', 
                'rm03/odom','rm04/odom',
                '/rm03/vicon_pose','/rm04/vicon_pose',
                '/tracked_polygons', 'gnn_objetcs',
                '/rm03/global_costmap/costmap_raw','/rm04/global_costmap/costmap_raw',
                '/graph_data', 
                '/navigate_to_pose/feedback', '/navigate_to_pose/result',
                '/rm03/plan', '/rm04/plan', 
                '/rm03/path', '/rm04/path', 
                '/rm03/cmd_vel','/rm04/cmd_vel', 
                '/rm03/behavior_tree_log', '/rm04/behavior_tree_log',
                '/tracked_polygons','gnn_objects'
                # '--qos-profile-overrides-path', 'src/gnn_object_segmentation/qos_overrides.yaml'
            ],
            output='screen'
        ),

        # --- Data Merger Node ---
        Node(
            package='gnn_object_segmentation',
            executable='data_merge',
            name='data_merge',
            parameters=[
                {"run_id": LaunchConfiguration('run_id')},
                {"window_size": 5}
            ],
            output='screen',
            arguments=[
                '--visualize', LaunchConfiguration('visualize'),
                '--simulation', LaunchConfiguration('simulation')
            ]
        ),

        # --- Tracked Polygon Marker Publisher ---
        Node(
            package='gnn_object_segmentation',
            executable='tracked_polygon_visualizer',
            name='tracked_polygon_visualizer',
            output='screen',
            parameters=[{
                'input_topic': '/tracked_polygons',
                'output_topic': '/tracked_polygon_markers'
            }]
        ),

        # --- Arena Static Markers (for testing or mapping) ---
        # Node(
        #     package='gnn_object_segmentation',
        #     executable='arena_marker_node',
        #     name='arena_marker_node',
        #     output='screen',
        #     condition=IfCondition(LaunchConfiguration('visualize'))
        # ),

        # --- RViz2 (Visualization) ---
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     output='screen',
        #     arguments=['-d', LaunchConfiguration('rviz_config')],
        #     condition=IfCondition(LaunchConfiguration('visualize'))
        # ),


    ])
