import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped,PoseWithCovarianceStamped
from collections import deque
import numpy as np
import json
import math
import struct
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import pickle
from .visualizer import GraphVisualizer
import argparse
from gnn_interfaces.msg import GraphData
from rclpy.parameter import Parameter
import argparse
from rclpy.parameter import Parameter
import tf2_ros
import os ,csv
from datetime import datetime
import time
from rclpy.clock import Clock


def create_pointcloud2_from_numpy(points, frame_id="map"):
    header = Header()
    header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    pc_data = pc2.create_cloud(header, fields, points)
    return pc_data


def statistical_outlier_removal(points, k=10, std_ratio=1.0):
    if len(points) < k:
        return points

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_distances = np.mean(distances[:, 1:], axis=1)

    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)

    threshold = global_mean + std_ratio * global_std
    mask = mean_distances < threshold
    return points[mask]

def pointcloud2_to_xyz_intensity(msg: PointCloud2):
    """Convert PointCloud2 message to an (N,4) numpy array of [x, y, z, intensity]."""
    points = []
    data = msg.data
    for i in range(msg.width):
        offset = i * msg.point_step
        x, = struct.unpack_from('<f', data, offset + 0)
        y, = struct.unpack_from('<f', data, offset + 4)
        z, = struct.unpack_from('<f', data, offset + 8)
        intensity, = struct.unpack_from('<f', data, offset + 16)
        points.append([x, y, z, intensity])
    return np.array(points, dtype=np.float32)


class DataMergerNode(Node):


    def __init__(self, visualize=True, simulation=False):
        super().__init__('data_merger_node')

        self.simulation = simulation

        self.robot_names = ['rm04', 'rm03']

        self.pose_buffer = {'rm04': deque(maxlen=1000), 'rm03': deque(maxlen=1000)}  # Always safe to define


        self.clock = Clock()

        # if self.simulation:
        #     self.timestamps_rm04 = self.load_timestamps('timestamps_rm04.json')
        #     self.timestamps_rm03 = self.load_timestamps('timestamps_rm03.json')
        #     self.current_idx = {'rm04': 0, 'rm03': 0}

        # else:
        self.timestamps_rm04 = []
        self.timestamps_rm03 = []
        self.current_idx = {'rm04': 0, 'rm03': 0}

        self.create_subscription(PoseWithCovarianceStamped, '/rm04/vicon_pose', self.vicon_callback_rm04, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/rm03/vicon_pose', self.vicon_callback_rm03, 10)

        self.radar_buffer = {'rm04': deque(maxlen=1000), 'rm03': deque(maxlen=1000)}

        self.create_subscription(PointCloud2, '/rm04/ti_mmwave/radar_scan_pcl', self.radar_callback_rm04, 10)
        self.create_subscription(PointCloud2, '/rm03/ti_mmwave/radar_scan_pcl', self.radar_callback_rm03, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_timer(0.03, self.sync_and_merge)
        self.graph_pub = self.create_publisher(GraphData, '/graph_data', 10)

        self.get_logger().info("Data Merger Node Initialized")
        self.merged_data_buffer = deque(maxlen=100)
        # self.N = 5
        self.temporal_threshold = 2.0

        with open('normalization_weights_unified.pkl', 'rb') as f:
            self.norm_weights = pickle.load(f)
            print(f"✅ Norm weight has been loaded ")

        # self.raw_pc_pub = self.create_publisher(PointCloud2, "/radar_points_raw", 10)
        self.visualizer = GraphVisualizer(self, frame_id="map") if visualize else None

        # --- Retrieve parameters ---
        self.declare_parameter("run_id", "default_run")
        self.run_id = self.get_parameter("run_id").get_parameter_value().string_value

        self.declare_parameter("window_size", 3)
        self.N = self.get_parameter("window_size").get_parameter_value().integer_value

        # --- Create logging folder ---
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join("datalogging/data_merge", f"{self.run_id}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # --- CSV log file path ---
        self.csv_path = os.path.join(self.log_dir, "data_merger_log.csv")

        # --- Initialize CSV log file ---
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_ros", "run_id", "window_size",
                "rm04_points", "rm03_points",
                "node_count", "edge_count", "inter_robot_edge_count",
                "graph_build_time_ms", "megre_latency", "vicon_delay_rm04_ms", "vicon_delay_rm03_ms"
            ])

        self.last_radar_timestamp = {'rm04': 0.0, 'rm03': 0.0}
        self.last_vicon_timestamp = {'rm04': 0.0, 'rm03': 0.0}




    def load_timestamps(self, filepath):
        with open(filepath, 'r') as f:
            ts_dict = json.load(f)
        # Convert keys to int and sort by key to create ordered list of timestamps
        sorted_items = sorted(ts_dict.items(), key=lambda x: int(x[0]))
        timestamps = [float(v) for k, v in sorted_items]
        return timestamps

    def radar_callback_rm04(self, msg):
        #if self.current_idx['rm04'] >= len(self.timestamps_rm04):
        #    self.get_logger().warning("No more rm04 radar timestamps available.")
        #    return
        #timestamp = self.timestamps_rm04[self.current_idx['rm04']]
        #self.current_idx['rm04'] += 1
        #print('callback for rm04')
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        points = pointcloud2_to_xyz_intensity(msg)
        # if len(points) > 0:
        #     p = points[0]
        #     self.get_logger().info(f"[DEBUG] RM03 first raw point: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}, snr={p[3]:.2f})")
            
        self.radar_buffer['rm04'].append((timestamp, points))
        self.radar_buffer['rm04'] = deque([(ts, pts) for ts, pts in self.radar_buffer['rm04'] if timestamp - ts < 2.0], maxlen=1000)
        self.last_radar_timestamp['rm04'] = timestamp
        # self.get_logger().debug(f"rm04 radar buffered with timestamp {timestamp:.6f}")

    def radar_callback_rm03(self, msg):
        #if self.current_idx['rm03'] >= len(self.timestamps_rm03):
        #    self.get_logger().warning("No more rm03 radar timestamps available.")
        #    return
        #timestamp = self.timestamps_rm03[self.current_idx['rm03']]
        #self.current_idx['rm03'] += 1
        #print("Got radar data from 3")
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        points = pointcloud2_to_xyz_intensity(msg)
        # if len(points) > 0:
        #     p = points[0]
        #     self.get_logger().info(f"[DEBUG] RM03 first raw point: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}, snr={p[3]:.2f})")

        self.radar_buffer['rm03'].append((timestamp, points))
        self.radar_buffer['rm03'] = deque([(ts, pts) for ts, pts in self.radar_buffer['rm03'] if timestamp - ts < 2.0], maxlen=1000)
        self.last_radar_timestamp['rm03'] = timestamp
        #self.get_logger().debug(f"rm03 radar buffered with timestamp {timestamp:.6f}")

    def vicon_callback_rm04(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = {
            'timestamp': timestamp,
            'translation': np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]),
            'rotation': np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])
        }
        self.pose_buffer['rm04'].append((timestamp, pose))
        self.last_vicon_timestamp['rm04'] = timestamp

    def vicon_callback_rm03(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = {
            'timestamp': timestamp,
            'translation': np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]),
            'rotation': np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])
        }
        self.pose_buffer['rm03'].append((timestamp, pose))
        self.last_vicon_timestamp['rm03'] = timestamp
        #self.get_logger().debug(f"rm03 Vicon pose buffered with timestamp {timestamp:.6f}")

    def log_graph_stats(self, metadata):
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metadata["timestamp_ros"],
                metadata["run_id"],
                metadata["window_size"],
                metadata["rm04_points"],
                metadata["rm03_points"],
                metadata["node_count"],
                metadata["edge_count"],
                metadata["inter_robot_edge_count"],
                metadata["graph_valid"],
                round(metadata["build_time_ms"], 2),
                round(metadata["vicon_delay_rm04_ms"], 2),
                round(metadata["vicon_delay_rm03_ms"], 2),
            ])


    def get_closest(self, buffer, ref_time, max_diff=1.0):
        """Return the closest data in buffer to ref_time within max_diff seconds."""
        closest = None
        min_diff = float('inf')
        for ts, data in buffer:
            diff = abs(ts - ref_time)
            if diff < min_diff:
                min_diff = diff
            if diff < max_diff:
                closest = (ts, data)
        if not closest:
            self.get_logger().warn(f"[SYNC] No match found within tolerance ({max_diff}s). Closest diff: {min_diff:.4f}s")
        return closest


    def transform_to_vicon(self, x, y, z):
        """Transform ROS coordinates to Vicon frame."""
        return -y, -x, z

    def calculate_metrics(self, x, y, z, snr, robot_name, bag_timestamp, robot_id):
        """Calculate range, azimuth, elevation from coords."""
        range_val = math.sqrt(x * x + y * y + z * z)

        if y == 0:
            detectedAzimuth = 90.0 if x >= 0 else -90.0
        else:
            detectedAzimuth = round(math.atan2(x, y) * 180 / math.pi, 3)

        if x == 0 and y == 0:
            detectedElevAngle = 90.0 if z >= 0 else -90.0
        else:
            detectedElevAngle = round(math.atan2(z, math.sqrt(x * x + y * y)) * 180 / math.pi, 3)

        return {
            f'{robot_name}_timestamp': bag_timestamp,
            'range': range_val,
            'azimuth': detectedAzimuth,
            'elevation': detectedElevAngle,
            'x': x,
            'y': y,
            'z': z,
            'snr': snr,
            'robot_id': robot_id,
            'robot_prefix_num': robot_name,
        }
    def transform_local_to_global(self,p_local, translation, quaternion=None):
        """
        Transforms a local 3D point to the global Vicon frame using rigid body transform.
        - p_local: np.array([x, y, z])
        - translation: np.array([tx, ty, tz])
        - quaternion: np.array([qx, qy, qz, qw])
        Returns: np.array([x, y, z]) in global frame
        """
        rot = R.from_quat(quaternion)
        return rot.apply(p_local) + translation
        #return p_local + translation
    

    def process_radar_points(self, points, robot_name, bag_timestamp, robot_id):
        radar_points = []
        viz_points = [] if self.visualizer else None

        # if self.simulation:
        #     latest_pose = self.pose_buffer[robot_name][-1][1]
        #     translation = latest_pose['translation']
        #     quaternion = latest_pose['rotation']
        # else:
        #self.get_logger().info(f"[PROCESS] {robot_name.upper()} processing {len(points)} raw points")

        try:
            tf = self.tf_buffer.lookup_transform(
                "map", f"{robot_name}/base_link",
                rclpy.time.Time(seconds=int(bag_timestamp), nanoseconds=int((bag_timestamp % 1) * 1e9))
            )
            translation = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ])
            quaternion = np.array([
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w
            ])
            self.get_logger().debug(f"[TF] {robot_name} TF @ {bag_timestamp:.3f} → trans=({translation[0]:.2f}, {translation[1]:.2f}) rot={quaternion}")

        except Exception as e:
            self.get_logger().warn(f"[TF] Failed to get transform for {robot_name} at {bag_timestamp:.3f}: {e}")
            return [], []

        for pt in points:
            x, y, z, snr = pt
            if (x < 0.3  and y <= 0.3):
                self.get_logger().debug(f"[SKIP] {robot_name} point: too close to the robot → x={x:.2f}, y={y:.2f}, z={z:.2f}")
                continue
            x_local, y_local, z_local = x, y, z
            p_local = np.array([x_local, y_local, z_local])
            p_global = R.from_quat(quaternion).apply(p_local) + translation
            
            xg, yg, zg = p_global
            # if not (-10.0 <= xg <= 10.0 and -5.0 <= yg <= 5.0):
            #     self.get_logger().debug(f"[SKIP] {robot_name} point out of bounds → x={xg:.2f}, y={yg:.2f}, z={zg:.2f}")
            #     continue
            if not (-12.0 <= xg <= 10.0 and -5.0 <= yg <= 7.0):
                self.get_logger().debug(f"[SKIP] {robot_name} point out of bounds → x={xg:.2f}, y={yg:.2f}, z={zg:.2f}")
                continue


            point_data = self.calculate_metrics(
                xg, yg, zg, snr, robot_name, bag_timestamp, robot_id
            )
            radar_points.append(point_data)

            if self.visualizer:
                p_vis = R.from_quat(quaternion).apply(np.array([x, y, z])) + translation
                xv, yv, zv = p_vis
                if not (-10.0 <= xv <= 10.0 and -5.0 <= yv <= 5.0):
                    continue
                viz_points.append({
                    'x': xv, 'y': yv, 'z': zv,
                    'robot_prefix_num': robot_name
                })
        # self.get_logger().info(f"[PROCESS] {robot_name.upper()} processing {len(points)} raw points")
        # self.get_logger().info(f"[PROCESS] {robot_name.upper()} → radar_points={len(radar_points)}, viz_points={len(viz_points) if viz_points else 'N/A'}")

        return radar_points, viz_points


    def get_recent_window_frames(self, current_time):
        """
        Get up to N merged frames within the 1-second threshold of current_time.
        Always include the current frame.
        """
        valid_frames = []
        for frame in reversed(self.merged_data_buffer):
            if abs(current_time - frame['timestamp']) <= self.temporal_threshold:
                valid_frames.append(frame)
            if len(valid_frames) == self.N:
                break
        if not valid_frames:
            self.get_logger().warning("No valid frames within threshold. Using current frame only.")
            return [self.merged_data_buffer[-1]]  # use latest only
        return list(valid_frames)  
    
    def normalize_column(self, vals, params):
        method = params.get("method", "")
        if method == "minmax":
            return (vals - params["min"]) / (params["max"] - params["min"] + 1e-7)
        elif method == "zscore":
            return (vals - params["mean"]) / (params["std"] + 1e-7)
        return np.zeros_like(vals)

    def normalize_angle(self, vals, params):
        sin = np.sin(np.radians(vals))
        cos = np.cos(np.radians(vals))
        sin_norm = (sin - params.get("sin_mean", 0)) / (params.get("sin_std", 1) + 1e-7)
        cos_norm = (cos - params.get("cos_mean", 0)) / (params.get("cos_std", 1) + 1e-7)
        return sin_norm, cos_norm
    
    def build_edge_index_and_features(self, positions, snr_norm, timestamps, base_k=8):
        N = len(positions)
        k = min(base_k, N)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(positions)
        distances, indices = nbrs.kneighbors(positions)

        edge_index = []
        edge_attr = []

        for i in range(N):
            for j in indices[i]:
                if i == j:
                    continue
                delta_pos = positions[j] - positions[i]
                delta_snr = snr_norm[j] - snr_norm[i]
                delta_t = timestamps[j] - timestamps[i]
                edge_index.append([i, j])
                edge_attr.append(np.hstack((delta_pos, delta_snr, delta_t)))

        #edge_index = torch.tensor(edge_index, dtype=torch.long).T  # shape (2, E)
        #edge_attr = torch.tensor(edge_attr, dtype=torch.float)     # shape (E, D)
        #edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        # ✅ Rewritten using only NumPy
        edge_index = np.array(edge_index, dtype=np.int64).T   # shape (2, E)
        edge_attr = np.array(edge_attr, dtype=np.float32)     # shape (E, D)
        return edge_index, edge_attr


    def build_graph_from_window(self, frames, norm_weights, base_k=8):
        all_points = []
        
        # Step 1: Flatten radar points directly from merged_data
        for frame in frames:
            ts = frame['timestamp']
            for radar_source, radar_pts in frame['radar_points'].items():
                if radar_pts is None:
                    continue
                for pt in radar_pts:
                    pt = pt.copy()
                    pt['timestamp'] = ts  # attach frame timestamp
                    all_points.append(pt)
        
        if len(all_points) == 0:
            return None  # nothing to build

        # Step 2: Extract positions and normalize features
        positions = np.array([[p['x'], p['y'], p['z']] for p in all_points])
        
        snr_vals = np.array([p['snr'] for p in all_points])
        range_vals = np.array([p['range'] for p in all_points])
        azimuth_vals = np.array([p['azimuth'] for p in all_points])
        elevation_vals = np.array([p['elevation'] for p in all_points])
        robot_ids = np.array([
            1 if p['robot_prefix_num'] in ('rm04', 'robot_1') else 2
            for p in all_points
        ])
        timestamps = np.array([p['timestamp'] for p in all_points])

        # Normalize features
        snr_norm = self.normalize_column(snr_vals, norm_weights.get("snr", {}))
        range_norm = self.normalize_column(range_vals, norm_weights.get("range", {}))
        az_sin_norm, az_cos_norm = self.normalize_angle(azimuth_vals, norm_weights.get("azimuth", {}))
        el_sin_norm, el_cos_norm = self.normalize_angle(elevation_vals, norm_weights.get("elevation", {}))

        # Combine all node features (positions + normalized features)
        node_features_np = np.stack([
            positions[:, 0], positions[:, 1], positions[:, 2],
            snr_norm, range_norm,
            az_sin_norm, az_cos_norm,
            el_sin_norm, el_cos_norm,
            robot_ids
        ], axis=1)

        #node_features = torch.tensor(node_features_np, dtype=torch.float)
        node_features = np.array(node_features_np, dtype=np.float32)

        # Step 3: Edge index + attributes using KNN
        edge_index, edge_attr = self.build_edge_index_and_features(
            positions, snr_norm, timestamps, base_k=base_k
        )

        return node_features, edge_index, edge_attr

    def publish_graph(self, node_features, edge_index, edge_attr):
        msg = GraphData()
        # msg.header = std_msgs.msg.Header()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.node_features = node_features.flatten().tolist()
        msg.node_feature_dim = node_features.shape[1] if node_features.ndim == 2 else 0

        msg.edge_index = edge_index.flatten().tolist()
        msg.edge_attr = edge_attr.flatten().tolist()

        if edge_attr.ndim == 2 and edge_attr.shape[0] > 0:
            msg.edge_attr_dim = edge_attr.shape[1]
        else:
            msg.edge_attr_dim = 0  # Safe default fallback

        msg.num_nodes = node_features.shape[0]
        msg.num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0

        self.graph_pub.publish(msg)

    def prune_buffer(self, buffer_dict, max_age, now):
        for key in buffer_dict:
            buffer_dict[key] = [
                (t, data) for (t, data) in buffer_dict[key]
                if now - t < max_age
            ]

    def prune_merged_data_buffer(self, max_age, now):
        self.merged_data_buffer = [
            frame for frame in self.merged_data_buffer
            if now - frame['timestamp'] < max_age
        ]

    def sync_and_merge(self):
        merge_start_time = time.perf_counter()

        #self.get_logger().info("[SYNC] Using TF-based poses; no internal pose_buffer")

        # Abort if no radar data at all
        if not self.radar_buffer['rm04'] and not self.radar_buffer['rm03']:
            self.get_logger().warn("Skipping graph: No radar data in either buffer.")
            return

        now = self.clock.now().nanoseconds * 1e-9
        fresh_threshold = 0.75  # seconds

        # Determine latest timestamps per robot (0 if buffer empty)
        latest_rm04_ts = self.radar_buffer['rm04'][-1][0] if self.radar_buffer['rm04'] else 0
        latest_rm03_ts = self.radar_buffer['rm03'][-1][0] if self.radar_buffer['rm03'] else 0

        # Ensure at least one radar message is fresh
        if (now - latest_rm04_ts > fresh_threshold) and (now - latest_rm03_ts > fresh_threshold):
            self.get_logger().warn("Skipping graph: No fresh radar data from rm04 or rm03.")
            return

        # Choose the reference timestamp for merging (latest available)
        if self.radar_buffer['rm04']:
            ref_timestamp = self.radar_buffer['rm04'][-1][0]
        elif self.radar_buffer['rm03']:
            ref_timestamp = self.radar_buffer['rm03'][-1][0]
        else:
            self.get_logger().warn("Skipping graph: Could not determine reference timestamp.")
            return

        # Get the radar scans closest to the reference timestamp
        rm04_radar = self.get_closest(self.radar_buffer['rm04'], ref_timestamp)
        rm03_radar = self.get_closest(self.radar_buffer['rm03'], ref_timestamp)

        # if rm03_radar:
        #     self.get_logger().info(f"[SYNC] Found RM03 radar at {rm03_radar[0]:.3f}")
        # else:
        #     self.get_logger().warn("[SYNC] No RM03 radar found near reference timestamp")

        merged_data = {
            'timestamp': ref_timestamp,
            'poses': {},
            'radar_points': {}
        }

        if rm04_radar:
            gnn_pts, viz_pts = self.process_radar_points(rm04_radar[1], "rm04", rm04_radar[0], "robot_1")
            merged_data['radar_points']['rm04'] = gnn_pts
            #if self.visualizer:
            #    self.visualizer.store_visualization_data('rm04', viz_pts)

        if rm03_radar:
            gnn_pts, viz_pts = self.process_radar_points(rm03_radar[1], "rm03", rm03_radar[0], "robot_2")
            merged_data['radar_points']['rm03'] = gnn_pts
            #if self.visualizer:
            #    self.visualizer.store_visualization_data('rm03', viz_pts)

        rm04_pts = len(merged_data['radar_points'].get('rm04', []))
        rm03_pts = len(merged_data['radar_points'].get('rm03', []))
        self.get_logger().info(f"[SYNC] Points merged: rm04={rm04_pts}, rm03={rm03_pts}")

        if rm04_pts == 0 and rm03_pts == 0:
            return

        for robot_id in ['rm04', 'rm03']:
            radar_data = merged_data['radar_points'].get(robot_id)
            if radar_data:
                points_array = np.array([[pt['x'], pt['y'], pt['z']] for pt in radar_data])
                if len(points_array) >= 15:
                    filtered = statistical_outlier_removal(points_array, k=8, std_ratio=1.0)
                    filtered_set = set(map(tuple, filtered))
                    filtered_points = [pt for pt in radar_data if (pt['x'], pt['y'], pt['z']) in filtered_set]
                    # self.get_logger().info(f"[FILTER] {robot_id}: {len(radar_data)} → {len(filtered_points)} after filtering.")
                    merged_data['radar_points'][robot_id] = filtered_points
                else:
                    self.get_logger().info(f"[FILTER] {robot_id}: Not enough points to filter.")

        self.merged_data_buffer.append(merged_data)
        #self.get_logger().info(f"[BUFFER] Added to merged buffer. Current size: {len(self.merged_data_buffer)}")
        graph_start_time = time.time()
        window_frames = self.get_recent_window_frames(merged_data['timestamp'])
        graph_data = self.build_graph_from_window(window_frames, self.norm_weights)
        graph_end_time = time.time()
        if graph_data is None:
            self.get_logger().warning("[GNN] Graph construction failed — not enough data.")
            return
        node_feats, edge_index, edge_attr = graph_data
        if edge_attr.shape[0] == 0 or edge_index.shape[1] == 0:
            self.get_logger().warning("[GNN] Graph invalid — no edges or edge attributes.")
            return

        merge_end_time = time.perf_counter()
        internal_merge_latency_ms = (merge_end_time - merge_start_time) * 1000

        #now = time.time()
        now = self.clock.now().nanoseconds * 1e-9  # ROS time in seconds
        vicon_delay_rm04 = (now - self.last_vicon_timestamp.get('rm04', now)) * 1000
        vicon_delay_rm03 = (now - self.last_vicon_timestamp.get('rm03', now)) * 1000
        radar_delay_rm04 = (now - self.last_radar_timestamp.get('rm04', now)) * 1000
        radar_delay_rm03 = (now - self.last_radar_timestamp.get('rm03', now)) * 1000
        
        inter_robot_edges = int(np.sum([
            1 for i, j in edge_index.T
            if node_feats[i, -1] != node_feats[j, -1]
        ]))
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                now,                      # timestamp_ros
                self.run_id,
                self.N,
                rm04_pts,
                rm03_pts,
                node_feats.shape[0],
                edge_index.shape[1],
                inter_robot_edges,
                round((graph_end_time - graph_start_time) * 1000, 2),  # build time
                internal_merge_latency_ms,  # compute merge latency if tracked
                round(vicon_delay_rm04, 2),
                round(vicon_delay_rm03, 2)
            ])
                

        #positions_only = node_feats[:, :3]
        #raw_pc_msg = create_pointcloud2_from_numpy(positions_only)
        #self.raw_pc_pub.publish(raw_pc_msg)

        self.publish_graph(node_feats, edge_index, edge_attr)
        self.get_logger().info(f"[GNN] Published graph: {node_feats.shape[0]} nodes, {edge_index.shape[1]} edges")

        max_buffer_age = 1.5  # seconds

        self.prune_buffer(self.radar_buffer, max_buffer_age, now)
        self.prune_buffer(self.pose_buffer, max_buffer_age, now)
        self.prune_merged_data_buffer(max_buffer_age, now)

        # if self.visualizer:
            # self.visualizer.publish_radar_points()
            # self.visualizer.publish_robot_positions(merged_data["poses"])


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="Data merger and graph builder node")
    parser.add_argument('--visualize', action='store_true', default=True ,help='Enable RViz visualization')
    parser.add_argument('--simulation', action='store_true', default=False, help='Run in simulation mode with rosbag timestamps')
    parsed_args, unknown = parser.parse_known_args()

    node = DataMergerNode(visualize=parsed_args.visualize, simulation=parsed_args.simulation)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
