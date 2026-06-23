#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer
import time # <--- Use standard time, not asyncio

class FakeNav2Server(Node):
    def __init__(self):
        super().__init__('fake_nav2_server')
        self._server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_cb
        )
        self.get_logger().info("Fake Nav2 Action Server Ready!")

    # Removed 'async' keyword
    def execute_cb(self, goal_handle):
        # 1. Log the goal
        pos = goal_handle.request.pose.pose.position
        self.get_logger().info(f"Received goal: x={pos.x:.2f}, y={pos.y:.2f}")
        
        # 2. Simulate robot movement (Blocking sleep is fine for a fake server)
        time.sleep(2.0)
        
        # 3. Mark as successful
        goal_handle.succeed()
        
        # 4. Return result
        result = NavigateToPose.Result()
        return result

def main(args=None):
    rclpy.init(args=args)
    node = FakeNav2Server()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()