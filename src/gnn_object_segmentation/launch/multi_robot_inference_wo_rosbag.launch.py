from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition

def generate_launch_description():
    return LaunchDescription([
        # --- Config Params ---
        DeclareLaunchArgument('visualize', default_value='True'),
        DeclareLaunchArgument('simulation', default_value='True'), # Default to True for safety
        
        # --- CRITICAL: Add this argument so we can pass it from terminal ---
        DeclareLaunchArgument('use_sim_time', default_value='True'), 
        
        DeclareLaunchArgument('rviz_config', default_value='src/gnn_object_segmentation/rviz/flw_hall_gnn.rviz'),
        
        # --- Robot Configuration ---
        # Pass the list as a string. The Python script will parse it with JSON.
        DeclareLaunchArgument('robot_list', default_value='["rm04", "rm03"]', description='List of robot namespaces'),

        # --- Dynamic Data Merger ---
        Node(
            package='gnn_object_segmentation',
            executable='data_merge_dynamic', 
            name='data_merge',
            parameters=[
                {"run_id": "default_run"},
                {"window_size": 5},
                # --- PASSING THE PARAMS CORRECTLY ---
                {"robot_list": LaunchConfiguration('robot_list')},
                {"use_sim_time": LaunchConfiguration('use_sim_time')} # <--- THE FIX
            ],
            output='screen',
            arguments=[
                '--visualize', LaunchConfiguration('visualize'),
                '--simulation', LaunchConfiguration('simulation')
            ]
        ),

        # --- RViz2 ---
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            # Pass use_sim_time to RViz too so it doesn't flicker
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            condition=IfCondition(LaunchConfiguration('visualize'))
        ),

        # --- Markers (Visuals) ---
        Node(
            package='gnn_object_segmentation',
            executable='arena_marker_node',
            name='arena_marker_node',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            condition=IfCondition(LaunchConfiguration('visualize'))
        ),

        Node(
            package='gnn_object_segmentation',
            executable='tracked_polygon_visualizer',
            name='tracked_polygon_visualizer',
            output='screen',
            parameters=[{
                'input_topic': '/tracked_polygons',
                'output_topic': '/tracked_polygon_markers',
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }]
        ),
    ])