from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, ColorRGBA
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import rclpy
from rclpy.node import Node
#import torch
from sensor_msgs.msg import PointField
from geometry_msgs.msg import Point
from gnn_interfaces.msg import TrackedPolygon

class GraphVisualizer:
    def __init__(self, node: Node, frame_id="map"):
        self.node = node
        self.frame_id = frame_id
        self.pc_pub = node.create_publisher(PointCloud2, '/graph_node_cloud', 10)
        self.robot_pub = node.create_publisher(MarkerArray, '/robot_pose_markers', 10)
        self.accumulate_points = True
        self.radar_history = []
        self.max_points = 10  # max to avoid RViz overload
        
        #self.publisher = node.create_publisher(Marker, '/arena_marker', 1)
        #self.timer = self.create_timer(2.0, self.publish_arena)
        #self.arena_pub = self.node.create_publisher(Marker, '/arena_marker', 1)
        # Timer for periodic sync every 50 ms
        #self.create_timer(5, self.publish_arena)

    def store_visualization_data(self, robot_name, points):
        if not hasattr(self, "viz_points"):
            self.viz_points = {"rm04": [], "rm03": []}

        self.viz_points[robot_name].extend(points)
        self.viz_points[robot_name] = self.viz_points[robot_name][-self.max_points:]

    def publish_arena(self):
        def make_point(x, y, z=0.01):
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            return p
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns = "arena"
        m.id = 999
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        # Set rectangle corners
        corners = [(-10, -5), (10, -5), (10, 6), (-10, 6), (-10, -5)]
        m.points = [make_point(x, y) for x, y in corners]

        m.scale.x = 0.1  # line width
        m.color = self.make_color(0.8, 0.8, 0.8, 1.0)
        m.lifetime.sec = 0  # permanent
        self.arena_pub.publish(m)


    def publish_radar_points(self):
        if not hasattr(self, "viz_points"):
            return

        color_map = {
            "rm04": (1.0, 0.5, 0.0),  # orange
            "rm03": (0.0, 1.0, 1.0)   # cyan
        }

        points = []
        for robot_name, pts in self.viz_points.items():
            r, g, b = color_map.get(robot_name, (0.5, 0.5, 0.5))
            for p in pts:
                pt = (p['x'], p['y'], p['z'], r, g, b)
                points.append(pt)

        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = pc2.create_cloud(header, fields, points)
        self.pc_pub.publish(cloud)

        
    def make_color(self, r, g, b, a=1.0):
        c = ColorRGBA()
        c.r, c.g, c.b, c.a = r, g, b, a
        return c

    def publish_robot_positions(self, poses_dict: dict):
        marker_array = MarkerArray()

        colors = {
            "rm04": self.make_color(1.0, 0.0, 0.0),
            "rm03": self.make_color(0.0, 0.0, 1.0)
        }

        for i, (name, pose) in enumerate(poses_dict.items()):
            # Sphere (robot position)
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns = "robot_sphere"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pose['translation'][0])
            m.pose.position.y = float(pose['translation'][1])
            m.pose.position.z = float(pose['translation'][2])
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color = colors.get(name, self.make_color(0.5, 0.5, 0.5))
            m.lifetime.sec = 1
            marker_array.markers.append(m)

            # Arrow (robot orientation)
            a = Marker()
            a.header.frame_id = self.frame_id
            a.header.stamp = m.header.stamp
            a.ns = "robot_arrows"
            a.id = i + 10
            a.type = Marker.ARROW
            a.action = Marker.ADD
            a.pose.position = m.pose.position
            a.pose.orientation.x = float(pose['rotation'][0])
            a.pose.orientation.y = float(pose['rotation'][1])
            a.pose.orientation.z = float(pose['rotation'][2])
            a.pose.orientation.w = float(pose['rotation'][3])
            a.scale.x = 0.4  # arrow shaft length
            a.scale.y = 0.05
            a.scale.z = 0.05
            a.color = m.color
            a.lifetime.sec = 1
            marker_array.markers.append(a)

        self.robot_pub.publish(marker_array)

    # def publish_robot_positions(self, poses_dict: dict):
    #     """poses_dict: {'rm04': {'translation': np.array, ...}, 'rm03': {...}}"""
    #     marker_array = MarkerArray()
    #     def make_color(r, g, b, a=1.0):
    #         c = ColorRGBA()
    #         c.r, c.g, c.b, c.a = r, g, b, a
    #         return c

    #     colors = {
    #         "rm04": make_color(1.0, 0.0, 0.0),
    #         "rm03": make_color(0.0, 0.0, 1.0)
    #     }

    #     for i, (name, pose) in enumerate(poses_dict.items()):
    #         m = Marker()
    #         m.header.frame_id = 'flw_hall'#self.frame_id
    #         m.header.stamp = self.node.get_clock().now().to_msg()
    #         m.ns = "robot_positions"
    #         m.id = i
    #         m.type = Marker.SPHERE
    #         m.action = Marker.ADD
    #         m.pose.position.x = float(pose['translation'][0])
    #         m.pose.position.y = float(pose['translation'][1])
    #         m.pose.position.z = float(pose['translation'][2])
    #         m.scale.x = m.scale.y = m.scale.z = 0.15
    #         m.color = colors.get(name)
    #         m.lifetime.sec = 1  # auto expire in RViz
    #         marker_array.markers.append(m)
    #     self.robot_pub.publish(marker_array)

