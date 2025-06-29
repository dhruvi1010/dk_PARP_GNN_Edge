#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class ArenaBoundaryPublisher(Node):
    def __init__(self):
        super().__init__('arena_boundary_publisher')
        self.publisher = self.create_publisher(Marker, '/arena_marker', 1)
        self.frame_id = "map"  # or any other global frame
        self.timer = self.create_timer(2.0, self.publish_arena)  # every 2 seconds

    def publish_arena(self):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "arena"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.lifetime.sec = 0

        marker.scale.x = 0.05
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Arena boundary (rectangle)
        corners = [(-10, -5), (10, -5), (10, 5), (-10, 5), (-10, -5)]
        marker.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in corners]

        self.publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = ArenaBoundaryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
