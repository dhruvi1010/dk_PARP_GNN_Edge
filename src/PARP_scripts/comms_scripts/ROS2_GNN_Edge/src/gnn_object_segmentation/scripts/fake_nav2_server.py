#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer
from geometry_msgs.msg import PoseStamped

class FakeNav2Server(Node):
    def __init__(self):
        super().__init__('fake_nav2_server')
        self._server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_cb
        )

    async def execute_cb(self, goal_handle):
        self.get_logger().info(f"Received goal: {goal_handle.request.pose.pose.position}")
        # Simulate success after a delay
        import asyncio
        await asyncio.sleep(2.0)
        result = NavigateToPose.Result()
        goal_handle.succeed()
        return result

def main(args=None):
    rclpy.init(args=args)
    node = FakeNav2Server()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
