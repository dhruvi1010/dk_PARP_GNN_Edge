from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="nav2_costmap_2d",
            executable="costmap_2d_node",
            name="costmap_test_node",
            output="screen",
            parameters=[
                {
                    "use_sim_time": False,
                },
                # Plugin parameters
                {
                    "plugins": ["gnn_costmap_layer"],
                    "gnn_costmap_layer": {
                        "plugin": "gnn_objects_layer::GNNObjectsLayer",
                        "topic": "/tracked_polygons"
                    }
                },
                {
                    "global_frame": "map",
                    "robot_base_frame": "base_link",
                    "resolution": 0.05,
                    "update_frequency": 1.0,
                    "publish_frequency": 1.0,
                    "width": 10.0,
                    "height": 10.0,
                    "origin_x": -5.0,
                    "origin_y": -5.0,
                }
            ]
        )
    ])
