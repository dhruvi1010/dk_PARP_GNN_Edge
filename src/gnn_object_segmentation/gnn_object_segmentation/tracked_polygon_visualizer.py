#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gnn_interfaces.msg import TrackedPolygon
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class TrackedPolygonVisualizer(Node):
    def __init__(self):
        super().__init__('tracked_polygon_visualizer')

        self.declare_parameter("input_topic", "/tracked_polygons")
        self.declare_parameter("output_topic", "/tracked_polygon_markers")

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        self.sub = self.create_subscription(
            TrackedPolygon,
            input_topic,
            self.callback,
            50
        )
        self.pub = self.create_publisher(MarkerArray, output_topic, 10)

        self.marker_id_counter = 0
        self.get_logger().info(f"Listening on {input_topic}, publishing RViz markers on {output_topic}")

    def callback(self, msg: TrackedPolygon):
        marker_array = MarkerArray()

        marker = Marker()
        marker.header = msg.header
        marker.ns = f"tracked_cls_{msg.label}"
        marker.id = self.marker_id_counter
        self.marker_id_counter += 1

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05

        r, g, b, a = self.class_to_color(msg.label)
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a

        marker.points = [Point(x=pt.x, y=pt.y, z=pt.z) for pt in msg.polygon.points]
        if marker.points:
            marker.points.append(marker.points[0])  # Close loop

        marker_array.markers.append(marker)
        self.pub.publish(marker_array)

    def class_to_color(self, cls_id):
        palette = [
            (0.5, 0.5, 0.5),    # gray
            (1.0, 0.0, 1.0),    # magenta
            (1.0, 0.65, 0.0),   # orange
            (0.0, 1.0, 1.0),    # cyan
            (1.0, 0.0, 0.0),    # red
        ]
        color = palette[cls_id % len(palette)]
        return (*color, 1.0)  # RGBA

def main(args=None):
    rclpy.init(args=args)
    node = TrackedPolygonVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
