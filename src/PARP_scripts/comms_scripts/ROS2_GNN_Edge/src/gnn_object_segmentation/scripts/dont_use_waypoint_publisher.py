#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient

import yaml
import os
import time
from datetime import datetime
from ament_index_python.packages import get_package_share_directory


class WaypointPublisher(Node):

    def __init__(self):
        super().__init__('waypoint_publisher')
        print("[DEBUG] 🚀 WaypointPublisher launched!")

        # Declare and get parameters
        self.declare_parameter('run_id', 'R1-N1')
        self.declare_parameter('robot_id', 'rm03')
        self.declare_parameter('delay_start_sec', 5)
        self.declare_parameter('waypoints_file', 'waypoints.yaml')

        self.run_id = self.get_parameter('run_id').get_parameter_value().string_value
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.delay_start_sec = self.get_parameter('delay_start_sec').get_parameter_value().integer_value
        self.waypoints_file = self.get_parameter('waypoints_file').get_parameter_value().string_value

        # Create NavigateToPose action client
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join('datalogging/nav_logs', f'{self.run_id}_{timestamp}')
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, f'waypoint_log_{self.robot_id}.csv')

        with open(self.log_file, 'w') as f:
            f.write('goal_index,start_time,end_time,status\n')

        # Load waypoints
        self.load_waypoints()

        # Delay start
        self.get_logger().info(f'Delaying start by {self.delay_start_sec} seconds...')
        time.sleep(self.delay_start_sec)

        # Start sending goals
        self.send_next_goal()

    def load_waypoints(self):
        # Search for the YAML file in config or absolute path
        if not os.path.isfile(self.waypoints_file):
            config_dir = os.path.join(get_package_share_directory('gnn_object_segmentation'), 'config')
            self.waypoints_file = os.path.join(config_dir, self.waypoints_file)

        with open(self.waypoints_file, 'r') as f:
            all_data = yaml.safe_load(f)

        self.waypoints = all_data.get(self.run_id, {}).get(self.robot_id, [])
        if not self.waypoints:
            self.get_logger().warn(f"No waypoints found for run_id={self.run_id}, robot_id={self.robot_id}")
        else:
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints for {self.robot_id} in {self.run_id}")

        self.current_index = 0

    def send_next_goal(self):
        if self.current_index >= len(self.waypoints):
            self.get_logger().info("All waypoints completed.")
            return

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available!')
            return

        wp = self.waypoints[self.current_index]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = wp['x']
        goal_msg.pose.pose.position.y = wp['y']
        goal_msg.pose.pose.orientation.z = self.yaw_to_quat_z(wp['theta'])

        self.start_time = time.time()
        self.get_logger().info(f"Sending goal {self.current_index + 1}/{len(self.waypoints)}: {wp}")
        self._action_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)

    def yaw_to_quat_z(self, yaw):
        import math
        return math.sin(yaw / 2.0)  # simplified planar quaternion z

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected.')
            return

        self.get_logger().info('Goal accepted.')
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        status = future.result().status
        status_str = self.translate_status(status)

        end_time = time.time()
        duration = end_time - self.start_time
        self.get_logger().info(f"Goal {self.current_index + 1} {status_str} in {duration:.2f} sec.")

        with open(self.log_file, 'a') as f:
            f.write(f"{self.current_index},{self.start_time},{end_time},{status_str}\n")

        self.current_index += 1
        self.send_next_goal()

    def translate_status(self, code):
        from action_msgs.msg import GoalStatus
        status_map = {
            GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
            GoalStatus.STATUS_ACCEPTED: 'ACCEPTED',
            GoalStatus.STATUS_EXECUTING: 'EXECUTING',
            GoalStatus.STATUS_CANCELING: 'CANCELING',
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
        }
        return status_map.get(code, 'INVALID')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()