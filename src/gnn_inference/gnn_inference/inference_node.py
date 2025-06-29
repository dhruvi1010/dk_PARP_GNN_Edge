#!/usr/bin/env python3
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from gnn_interfaces.msg import GraphData
import torch
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2
import networkx as nx
from collections import defaultdict
from scipy.spatial import ConvexHull
from gnn_modules.set_configurations.set_param_for_inference_gnn import set_parameters_for_inference
from gnn_modules.set_configurations.set_config_gnn import config
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
from gnn_modules.tracker.kalman_tracker import ObjectTracker
import inspect
#from gnn_inference.model_utils import load_model_pipeline




class GNNInferenceNode(Node):
    def __init__(self):
        super().__init__('gnn_inference_node')

        self.octomap_pub = self.create_publisher(PointCloud2, '/octomap_input_points', 10)

        self.subscription = self.create_subscription(
            GraphData,
            '/graph_data',
            self.graph_callback,
            10
        )
        self.get_logger().info('✅ GNN Inference Node initialized and listening on /graph_data')
        # ⚙️ Inference setup
        config_file = 'configuration_flw_gnn_v1.yml'  # update this
        self.trained_weights_path = 'graph_based_detector.pt'  # update this
        module_rootdir = '.'
#        self.model, self.device = load_model_pipeline(
#            config_path='config/config.yaml',
#            weights_path='weights/model_weights.pt',
#            module_root='.'
#       )
        self.config_obj = config(config_file)
        eps = 1.0

        param_obj = set_parameters_for_inference(module_rootdir, self.config_obj, self.trained_weights_path)
        self.device = param_obj['device']
        grid = param_obj['grid']
        self.detector = param_obj['detector']
        self.detector.set_param_for_proposal_extraction(eps, compute_adj_mat_from_links=False) 
        self.marker_pub = self.create_publisher(MarkerArray, '/gnn_objects', 10)
        self.object_pub = self.create_publisher(PointCloud2, '/gnn_objects_pc', 10)

        self.tracked_objects = defaultdict(list)
        self.max_robot_lifetime = 3.0  # seconds
        
        self.trackers_per_class = defaultdict(list)
        self.tracker_id_counter = 0
        self.max_tracker_age = 1.0  # seconds

        self.get_logger().info("🧠 GNN Inference Node ready.")

    def graph_callback(self, msg: GraphData):
        try:
            N, D = msg.num_nodes, msg.node_feature_dim
            E, D_e = msg.num_edges, msg.edge_attr_dim
            node_feats = np.array(msg.node_features, dtype=np.float32).reshape((N, D))
            edge_index = np.array(msg.edge_index, dtype=np.int64).reshape((2, E))
            edge_attr = np.array(msg.edge_attr, dtype=np.float32).reshape((E, D_e))
            # Replace this with your actual GNN model inference
            pred_class, pred_edge_class, pred_class_conf = self.run_inference(node_feats, edge_index, edge_attr)
        except Exception as e:
            self.get_logger().error(f"❌ Failed inference: {e}")

        self.use_graph_clusters = True
        if self.use_graph_clusters:
            self.process_predictions(node_feats, edge_index, pred_class, pred_edge_class, pred_class_conf)
        else:
            self.process_predictions_v2(node_feats, pred_class)

    def run_inference(self, node_feats, edge_index, edge_attr):
        # Placeholder dummy inference: 4-class logits per node

        # ✅ Convert to tensors if needed for PyTorch model
        node_feats = torch.from_numpy(node_feats).to(self.device)
        edge_index = torch.from_numpy(edge_index).to(self.device)
        edge_attr = torch.from_numpy(edge_attr).to(self.device)
        node_cls_predictions, _, edge_cls_predictions, _, _ = self.detector(
            node_features=node_feats,
            edge_features=edge_attr,
            #other_features=None,
            edge_index=edge_index,
            adj_matrix=None
        )

        cls_prob = F.softmax(node_cls_predictions, dim=-1)
        conf_scores, cls_idx = torch.max(cls_prob, dim=-1)

       # Probability that edge is \"connected\"\n",
        edge_connected_prob = torch.sigmoid(edge_cls_predictions[:, 1])
        threshold = 0.88  # You can tune this later for best F1\n",
        pred_edge_class = (edge_connected_prob >= threshold).long()

        pred_class = cls_idx.detach().cpu().numpy()
        conf_scores = conf_scores.detach().cpu().numpy()
        pred_edge_class = pred_edge_class.detach().cpu().numpy()


        return pred_class,pred_edge_class,conf_scores
    
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
    
    def create_pointcloud2(self, points, frame_id="map"):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        return pc2.create_cloud(header, fields, points)
    
    def process_predictions(self, node_feats, edge_index, pred_class, pred_edge_class, conf_scores):
        # flips the points back to ROS
        # 🔁 Revert the manual radar remapping
        #node_feats[:, 0], node_feats[:, 1] = -node_feats[:, 1], -node_feats[:, 0]


        G = nx.Graph()
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i], edge_index[1, i]
            if pred_edge_class[i] == 1 and pred_class[src] == pred_class[tgt]:
                G.add_edge(src, tgt)

        current_time = self.get_clock().now().nanoseconds / 1e9
        clusters = list(nx.connected_components(G))

        new_detections = defaultdict(list)
        for cluster in clusters:
            points = np.array([node_feats[i][:2] for i in cluster])
            cls = pred_class[list(cluster)[0]]
            conf = np.mean([conf_scores[i] for i in cluster])

            if cls == 0 or conf < 0.7:
                continue
            if len(points) < 4 or np.linalg.matrix_rank(points - points[0]) < 2:
                continue

            try:
                hull = ConvexHull(points)
                hull_pts = points[hull.vertices]
                centroid = np.mean(points, axis=0)
                new_detections[cls].append((centroid, hull_pts, conf))
            except Exception as e:
                self.get_logger().warn(f"Could not compute hull for cluster (class {cls}): {e}")

        for cls, detections in new_detections.items():
            updated_ids = set()
            for centroid, shape, conf in detections:
                best_tracker = None
                best_dist = float('inf')

                for tracker in self.trackers_per_class[cls]:
                    pred_pos = tracker.predict(current_time=current_time)
                    dist = np.linalg.norm(pred_pos - centroid)
                    if dist < 0.5 and dist < best_dist:
                        best_dist = dist
                        best_tracker = tracker

                if best_tracker:
                    best_tracker.update(centroid,current_time)
                    best_tracker.last_seen = current_time
                    best_tracker.shape = shape
                    updated_ids.add(id(best_tracker))
                else:
                    tracker = ObjectTracker(initial_pos=centroid)
                    tracker.shape = shape
                    tracker.last_seen = current_time
                    self.trackers_per_class[cls].append(tracker)

            self.trackers_per_class[cls] = [
                t for t in self.trackers_per_class[cls] if (current_time - t.last_seen) < self.max_tracker_age
            ]

        self.tracked_objects.clear()
        for cls, trackers in self.trackers_per_class.items():
            for tracker in trackers:
                self.tracked_objects[cls].append({
                    'timestamp': current_time,
                    'centroid': tracker.get_state(),
                    'shape': tracker.shape,
                    'confidence': 1.0
                })
        self.publish_tracked_objects_as_pointcloud()

        # RViz Visualization
        marker_array = MarkerArray()
        marker_id = 0
        for cls, objects in self.tracked_objects.items():
            for obj in objects:
                shape = obj['shape']
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"tracked_cls_{cls}"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.05
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.class_to_color(cls)
                marker.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in shape]
                marker.points.append(Point(x=float(shape[0][0]), y=float(shape[0][1]), z=0.05))
                marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def publish_tracked_objects_as_pointcloud(self):
        import numpy as np
        from sensor_msgs.msg import PointField
        from sensor_msgs_py.point_cloud2 import create_cloud
        import std_msgs.msg

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=12, datatype=PointField.UINT32, count=1),
            PointField(name='confidence', offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        point_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('label', np.uint32),
            ('confidence', np.float32),
        ])

        # Collect points
        total_count = sum(len(v) for v in self.tracked_objects.values())
        points = np.zeros(total_count, dtype=point_dtype)

        idx = 0
        for cls, objects in self.tracked_objects.items():
            for obj in objects:
                x, y = obj['centroid']
                conf = float(obj.get('confidence', 1.0))
                points[idx] = (x, y, 0.0, int(cls), conf)
                idx += 1

        # Build message
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        cloud_msg = create_cloud(header, fields, points)
        self.object_pub.publish(cloud_msg)


    def process_predictions_v2(self, node_feats, pred_class, merge_dist=0.5):
        """
        Alternate prediction processor using DBSCAN clustering and convex hulls.
        Avoids dependency on edge_index. More robust to noisy graphs.
        Also publishes static class points to OctoMap.
        """

        LABEL_MAP = {
            1: "Workstation",
            2: "Robot",
            3: "Boundary",
            4: "Forklift"
        }

        marker_array = MarkerArray()
        marker_id = 0

        static_classes = {1, 3, 4}  # workstation, boundary, forklift
        static_points = []  # for octomap input

        for cls in np.unique(pred_class):
            if cls == 0:
                continue  # Skip "Other"

            mask = pred_class == cls
            points = node_feats[mask][:, :2]  # take only x, y

            if len(points) < 3:
                continue

            db = DBSCAN(eps=merge_dist, min_samples=4).fit(points)
            cluster_labels = db.labels_

            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    continue  # noise

                cluster_pts = points[cluster_labels == cluster_id]

                if len(cluster_pts) < 3:
                    continue

                try:
                    hull = ConvexHull(cluster_pts, qhull_options="QJ")
                    hull_pts = cluster_pts[hull.vertices]
                except:
                    hull_pts = cluster_pts

                # # ---- Store for OctoMap if static ----
                # if cls in static_classes:
                #     for pt in cluster_pts:
                #         static_points.append((pt[0], pt[1], 0.05))  # Add Z

                # ---- RViz Marker ----
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "gnn_objects_v2"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.08
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.class_to_color(cls)

                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                marker.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in hull_pts]
                marker_array.markers.append(marker)

                # ---- Optional Text Labels ----
                label = LABEL_MAP.get(cls, f"Class{cls}")
                cx, cy = np.mean(cluster_pts[:, 0]), np.mean(cluster_pts[:, 1])
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "gnn_labels_v2"
                text_marker.id = 1000 + marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = float(cx)
                text_marker.pose.position.y = float(cy)
                text_marker.pose.position.z = 0.5
                text_marker.scale.z = 0.3
                text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                text_marker.text = label
                marker_array.markers.append(text_marker)

        # ---- Publish visualization markers ----
        self.marker_pub.publish(marker_array)

        # # ---- Publish static points to OctoMap ----
        # if static_points and hasattr(self, 'octomap_pub'):
        #     cloud = self.create_pointcloud2(static_points, frame_id="map")
        #     self.octomap_pub.publish(cloud)

    def publish_octomap_from_predictions(self, node_feats, edge_index, pred_class, pred_edge_class):
        G = nx.Graph()
        for i in range(edge_index.shape[1]):
            if pred_edge_class[i] == 1:
                src, tgt = edge_index[0, i], edge_index[1, i]
                if pred_class[src] == pred_class[tgt] and pred_class[src] in {1, 3, 4}:
                    G.add_edge(src, tgt)

        components = list(nx.connected_components(G))
        octo_points = []

        for comp in components:
            comp = list(comp)
            points = node_feats[comp][:, :2]
            if len(points) < 3:
                continue
            try:
                hull = ConvexHull(points, qhull_options="QJ")
                for idx in hull.vertices:
                    x, y = points[idx]
                    octo_points.append((float(x), float(y), 0.05))  # Add Z
            except:
                continue

        if octo_points and hasattr(self, 'octomap_pub'):
            cloud = self.create_pointcloud2(octo_points, frame_id="map")
            self.octomap_pub.publish(cloud)





def main(args=None):
    rclpy.init(args=args)
    node = GNNInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()