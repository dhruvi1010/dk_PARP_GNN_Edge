#!/usr/bin/env python3
import os
import time
import math
import yaml
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, GoalResponse
from rclpy.executors import ExternalShutdownException
from rclpy.task import Future

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import GetCostmap
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory


class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('waypoint_publisher')

        # defaults so timers can't race
        self.current_index = 0
        self.current_retries = 0
        self.active_goal_handle = None
        self.goal_start_time = None
        self.waypoints = []
        # ---- Parameters
        self.declare_parameter('run_id', 'R1-N1')
        self.declare_parameter('robot_id', 'rm03')
        self.declare_parameter('delay_start_sec', 5)
        self.declare_parameter('waypoints_file', 'waypoints.yaml')
        self.declare_parameter('check_radius_m', 0.25)     # radius to test around goal in costmap
        self.declare_parameter('lethal_threshold', 254)     # costmap lethal cost
        self.declare_parameter('use_local_costmap', True)   # True -> /local_costmap/get_costmap
        self.declare_parameter('max_retries', 2)            # retries per waypoint when not blocked
        self.declare_parameter('goal_timeout_sec', 120.0)   # cancel + retry if exceeds this
        #self.declare_parameter('use_sim_time', True)
        p = self.get_parameter
        self.run_id = p('run_id').get_parameter_value().string_value
        self.robot_id = p('robot_id').get_parameter_value().string_value
        self.delay_start_sec = p('delay_start_sec').get_parameter_value().integer_value
        self.waypoints_file = p('waypoints_file').get_parameter_value().string_value
        self.check_radius_m = p('check_radius_m').get_parameter_value().double_value
        self.lethal_threshold = p('lethal_threshold').get_parameter_value().integer_value
        self.use_local_costmap = p('use_local_costmap').get_parameter_value().bool_value
        self.max_retries = p('max_retries').get_parameter_value().integer_value
        self.goal_timeout_sec = p('goal_timeout_sec').get_parameter_value().double_value
        # add a param (default true)
        self.declare_parameter('auto_heading', True)
        self.auto_heading = self.get_parameter('auto_heading').value
        # ---- Action client
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available!')
            return

        # ---- Costmap client
        robot_ns = self.robot_id
        srv_name = f"/{robot_ns}/local_costmap/get_costmap" if self.use_local_costmap \
                else f"/{robot_ns}/global_costmap/get_costmap"
        self._costmap_client = self.create_client(GetCostmap, srv_name)
        if not self._costmap_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'Costmap service {srv_name} not available yet. Skipping block checks will be FALSE until it appears.')

        # ---- Logging
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join('datalogging/nav_logs', f'{self.run_id}_{timestamp}')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f'waypoint_log_{self.robot_id}.csv')
        with open(self.log_file, 'w') as f:
            f.write('goal_index,start_time,end_time,status,retries,skipped_blocked\n')

        self.load_waypoints()

        # AFTER (ROS time; no wall sleep)
        self.get_logger().info(f'Delaying start by {self.delay_start_sec} seconds (ROS time)...')
        self.kickoff_timer = self.create_timer(float(self.delay_start_sec), self._kickoff_once)


        # watchdog timer for timeout
        self.watchdog = self.create_timer(0.5, self._check_goal_timeout)

    def _segment_yaw(self, i):
        # i: current waypoint index
        n = len(self.waypoints)
        if n <= 1:
            return 0.0
        if i < n - 1:
            p0, p1 = self.waypoints[i], self.waypoints[i+1]
        else:
            p0, p1 = self.waypoints[i-1], self.waypoints[i]
        return math.atan2(float(p1['y']) - float(p0['y']),
                        float(p1['x']) - float(p0['x']))


    def _kickoff_once(self):
        # cancel so it fires only once
        self.kickoff_timer.cancel()
        # init & go
        self.current_index = 0
        self.current_retries = 0
        self.active_goal_handle = None
        self.goal_start_time = None
        self.send_next_goal()
    # ---------------- Waypoints I/O ----------------
    def load_waypoints(self):
        if not os.path.isfile(self.waypoints_file):
            config_dir = os.path.join(get_package_share_directory('gnn_object_segmentation'), 'config')
            self.waypoints_file = os.path.join(config_dir, self.waypoints_file)

        with open(self.waypoints_file, 'r') as f:
            all_data = yaml.safe_load(f)

        self.waypoints = all_data.get(self.run_id, {}).get(self.robot_id, [])
        if not self.waypoints:
            self.get_logger().warn(f'No waypoints found for run_id={self.run_id}, robot_id={self.robot_id}')
        else:
            self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints for {self.robot_id} in {self.run_id}')

    # ---------------- Navigation flow ----------------
    def send_next_goal(self):
        if self.current_index >= len(self.waypoints):
            self.get_logger().info('All waypoints completed.')
            return

        wp = self.waypoints[self.current_index]

        yaw = self._segment_yaw(self.current_index) if self.auto_heading else float(wp['theta'])

        # Skip waypoint only if blocked in the costmap
        blocked = self.is_goal_blocked(wp['x'], wp['y'])
        if blocked:
            self.get_logger().warn(f"Waypoint {self.current_index} appears BLOCKED in costmap. Skipping.")
            self._log_status('SKIPPED_BLOCKED', skipped=True)
            self.current_index += 1
            self.current_retries = 0
            self._call_later(0.05, self.send_next_goal)
            return

        # Build and send goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._pose_stamped(wp['x'], wp['y'], yaw)

        self.goal_start_time = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().info(f"Sending goal {self.current_index + 1}/{len(self.waypoints)}: {wp}, retry {self.current_retries}/{self.max_retries}")
        send_future = self._action_client.send_goal_async(goal_msg, feedback_callback=None)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by server.')
            self._handle_failure('REJECTED')
            return

        self.get_logger().info('Goal accepted.')
        self.active_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def result_callback(self, future: Future):
        self.active_goal_handle = None
        result = future.result()
        status = result.status if result else GoalStatus.STATUS_UNKNOWN

        status_str = self._status_to_string(status)
        self.get_logger().info(f"Goal {self.current_index + 1} finished with status: {status_str}")

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._log_status('SUCCEEDED')
            self.current_index += 1
            self.current_retries = 0
            self._call_later(0.05, self.send_next_goal)
            return

        # For ABORTED/CANCELED: check if it is blocked; if yes, skip; else retry
        wp = self.waypoints[self.current_index]
        if self.is_goal_blocked(wp['x'], wp['y']):
            self.get_logger().warn(f"Waypoint {self.current_index} blocked after attempt → skipping.")
            self._log_status('SKIPPED_BLOCKED', skipped=True)
            self.current_index += 1
            self.current_retries = 0
        else:
            self._handle_failure(self._status_to_string(status))

        self._call_later(0.05, self.send_next_goal)

    # ---------------- Helpers ----------------
    def _handle_failure(self, label: str):
        if self.current_retries < self.max_retries:
            self.current_retries += 1
            self._log_status(f'RETRY_{label}', retry=True)
            self.get_logger().info(f"Retrying waypoint {self.current_index} (attempt {self.current_retries}/{self.max_retries})")
        else:
            self.get_logger().warn(f"Max retries reached for waypoint {self.current_index}. Marking as FAILED and moving on.")
            self._log_status('FAILED_MAX_RETRIES')
            self.current_index += 1
            self.current_retries = 0

    def _check_goal_timeout(self):
        if self.active_goal_handle is None or self.goal_start_time is None:
            return
        if (self.get_clock().now().nanoseconds / 1e9 - self.goal_start_time) > self.goal_timeout_sec:
            self.get_logger().warn(f"Goal {self.current_index} timeout after {self.goal_timeout_sec:.1f}s — canceling.")
            self.active_goal_handle.cancel_goal_async()

    def _pose_stamped(self, x, y, yaw) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps

    def _status_to_string(self, code: int) -> str:
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

    def _log_status(self, status: str, skipped: bool = False, retry: bool = False):
        end_time = self.get_clock().now().nanoseconds / 1e9
        with open(self.log_file, 'a') as f:
            f.write(f"{self.current_index},{self.goal_start_time or 0.0},{end_time},{status},{self.current_retries},{int(skipped)}\n")

    def _call_later(self, delay_sec: float, fn, *args, **kwargs):
        def _cb():
            fn(*args, **kwargs)
            timer.cancel()
        timer = self.create_timer(delay_sec, _cb)

    # ---------------- Costmap block check ----------------
    def is_goal_blocked(self, gx: float, gy: float) -> bool:
        """Returns True if any cell within check_radius_m around (gx, gy) is >= lethal_threshold."""
        if not self._costmap_client.service_is_ready():
            # If the service isn't ready yet, be conservative: do NOT skip
            return False

        req = GetCostmap.Request()
        try:
            future = self._costmap_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            if not future.done() or future.result() is None:
                return False
            msg = future.result().map  # nav2_msgs/Costmap
        except Exception as e:
            self.get_logger().warn(f"Costmap service error: {e}")
            return False

        # World->map index conversion
        res = msg.metadata.resolution
        origin_x = msg.metadata.origin.position.x
        origin_y = msg.metadata.origin.position.y
        width = msg.metadata.size_x
        height = msg.metadata.size_y
        data = msg.data  # uint8 list of length width*height

        mx = int((gx - origin_x) / res)
        my = int((gy - origin_y) / res)

        if mx < 0 or my < 0 or mx >= width or my >= height:
            # Out of costmap bounds → treat as not blocked (let planner handle)
            return False

        radius_cells = max(1, int(self.check_radius_m / res))
        lethal = self.lethal_threshold

        for iy in range(max(0, my - radius_cells), min(height - 1, my + radius_cells) + 1):
            for ix in range(max(0, mx - radius_cells), min(width - 1, mx + radius_cells) + 1):
                idx = iy * width + ix
                if data[idx] >= lethal:
                    return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
