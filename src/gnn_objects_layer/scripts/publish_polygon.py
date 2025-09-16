import rclpy
from rclpy.node import Node
from gnn_interfaces.msg import TrackedPolygon
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
import time

class PolygonPublisher(Node):
    def __init__(self):
        super().__init__('polygon_publisher')
        self.publisher_ = self.create_publisher(TrackedPolygon, '/tracked_polygons', 10)
        self.timer = self.create_timer(1.0, self.publish_polygon)

    def publish_polygon(self):
        msg = TrackedPolygon()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # Define a square polygon
        msg.polygon.points = [
            Point32(x=1.0, y=1.0),
            Point32(x=2.0, y=1.0),
            Point32(x=2.0, y=2.0),
            Point32(x=1.0, y=2.0)
        ]

        self.publisher_.publish(msg)
        self.get_logger().info("Published test polygon.")

def main(args=None):
    rclpy.init(args=args)
    node = PolygonPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
