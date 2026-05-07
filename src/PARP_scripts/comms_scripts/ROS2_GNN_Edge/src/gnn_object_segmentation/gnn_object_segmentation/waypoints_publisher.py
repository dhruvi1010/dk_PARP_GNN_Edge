#!/usr/bin/env python3
import os
import time
import math
import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus

def yaw_to_quat(yaw: float):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return (0.0, 0.0, qz, qw)

class WaypointNavigatorSync(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        # Params (minimal)
        self.declare_parameter('run_id', 'R1-N1')
        self.declare_parameter('robot_id', 'rm03')
        self.declare_parameter('waypoints_file', 'waypoints.yaml')
        self.declare_parameter('dwell_sec', 5.0)           # wall-time pause between goals
        self.declare_parameter('loop', False)              # loop forever
        self.declare_parameter('goal_timeout_sec', 120.0)  # cancel if a goal exceeds this

        p = self.get_parameter
        self.run_id          = p('run_id').value
        self.robot_id        = p('robot_id').value
        self.wp_file         = p('waypoints_file').value
        self.dwell_sec       = float(p('dwell_sec').value)
        self.loop            = bool(p('loop').value)
        self.goal_timeout    = float(p('goal_timeout_sec').value)

        self.waypoints = self._load_waypoints()
        if not self.waypoints:
            self.get_logger().error(f'No waypoints for run_id={self.run_id}, robot_id={self.robot_id}')
            return

        # Action name: respect node namespace if set (e.g. __ns:=/rm04)
        ns = self.get_namespace()
        self.action_name = (ns.rstrip('/') + '/navigate_to_pose') if (ns and ns != '/') else 'navigate_to_pose'
        self.client = ActionClient(self, NavigateToPose, self.action_name)

    # ---------- I/O ----------
    def _load_waypoints(self):
        path = self.wp_file
        if not os.path.isfile(path):
            try:
                cfg_dir = os.path.join(get_package_share_directory('gnn_object_segmentation'), 'config')
                cand = os.path.join(cfg_dir, self.wp_file)
                if os.path.isfile(cand):
                    path = cand
            except Exception:
                pass

        if not os.path.isfile(path):
            self.get_logger().error(f'Waypoints file not found: {self.wp_file}')
            return []

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Nested (run_id/robot_id) or flat
        wps = []
        if isinstance(data, dict) and self.run_id in data:
            robots = data[self.run_id]
            if isinstance(robots, dict) and self.robot_id in robots:
                wps = robots[self.robot_id]
        elif isinstance(data, dict) and 'waypoints' in data:
            wps = data['waypoints']
        elif isinstance(data, list):
            wps = data
        else:
            self.get_logger().warn('Unrecognized YAML structure; expected nested run_id/robot_id or a "waypoints" list.')

        # Normalize & fill yaw if zero (path tangent)
        norm = []
        for i, wp in enumerate(wps):
            x = float(wp.get('x', 0.0))
            y = float(wp.get('y', 0.0))
            th = float(wp.get('theta', 0.0) or 0.0)
            if abs(th) < 1e-6:
                if i < len(wps) - 1:
                    nx, ny = float(wps[i+1].get('x', x)), float(wps[i+1].get('y', y))
                    th = math.atan2(ny - y, nx - x)
                elif i > 0:
                    px, py = float(wps[i-1].get('x', x)), float(wps[i-1].get('y', y))
                    th = math.atan2(y - py, x - px)
                else:
                    th = 0.0
            norm.append({'x': x, 'y': y, 'theta': th, 'name': wp.get('name', f'wp_{i}')})
        self.get_logger().info(f"Loaded {len(norm)} waypoints for {self.robot_id} in {self.run_id}")
        return norm

    # ---------- Core (synchronous with timeout/cancel) ----------
    def run(self):
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'Action server not available: {self.action_name}')
            return

        self.get_logger().info(f"Waypoints loaded: {len(self.waypoints)}")  # quick sanity

        def send_and_wait(i, wp):
            goal = NavigateToPose.Goal()
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = wp['x']
            ps.pose.position.y = wp['y']
            qx, qy, qz, qw = yaw_to_quat(wp['theta'])
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            goal.pose = ps

            self.get_logger().info(
                f"➡️  Goal {i+1}/{len(self.waypoints)} → {wp['name']} "
                f"(x={wp['x']:.2f}, y={wp['y']:.2f}, yaw={wp['theta']:.2f} rad)"
            )

            # send & wait acceptance
            f_goal = self.client.send_goal_async(goal)
            if not rclpy.spin_until_future_complete(self, f_goal, timeout_sec=5.0):
                self.get_logger().warn("Goal send timed out waiting for acceptance")
                return False
            handle = f_goal.result()
            if not handle or not handle.accepted:
                self.get_logger().warn("Goal rejected by server")
                return False

            # wait for result with timeout; cancel if exceeded
            t0 = time.time()
            f_res = handle.get_result_async()
            while rclpy.ok():
                done = rclpy.spin_until_future_complete(self, f_res, timeout_sec=0.5)
                if done:
                    resp = f_res.result()   # NavigateToPose_GetResult.Response
                    status = getattr(resp, 'status', GoalStatus.STATUS_UNKNOWN)
                    self.get_logger().info(f"✅ Result received (status={status})")
                    return status == GoalStatus.STATUS_SUCCEEDED

                if time.time() - t0 > self.goal_timeout:
                    self.get_logger().warn(f"⏱️ Goal timeout after {self.goal_timeout:.1f}s — canceling.")
                    cancel_future = handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                    return False

        # run once (unless loop:=true)
        while rclpy.ok():
            for i, wp in enumerate(self.waypoints):
                ok = send_and_wait(i, wp)

                # move on regardless; you can add retries here if you like
                if self.dwell_sec > 0:
                    self.get_logger().info(f"⏳ Waiting {self.dwell_sec:.1f}s before next…")
                    time.sleep(self.dwell_sec)  # wall time on purpose

            if not self.loop:
                break


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigatorSync()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
