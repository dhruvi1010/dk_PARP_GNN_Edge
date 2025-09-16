# ROS_GNN_Edge – Shared Perception for Obstacle Avoidance

This workspace provides a **multi-robot perception and obstacle avoidance pipeline** using **Graph Neural Networks (GNNs)**.  
Radar and pose data are merged across robots, converted into graph structures, processed by a GNN, and used within the **Nav2 costmap layer** for shared obstacle avoidance.  

---

## 1️⃣ Build & Setup

```bash
cd ~/dev/ROS_GNN_ws
colcon build
source install/setup.bash
```

---

## 2️⃣ Running the System

### Data Merger Node (with rosbag recording)
```bash
ros2 launch gnn_object_segmentation multi_robot_inference.launch.py run_id:=Horizontal
```

### GNN Inference Node
```bash
ros2 run gnn_inference inference_node --ros-args -p run_id:=Horizontal
```

### Waypoint Publishers

- **rm03**
```bash
ros2 run gnn_object_segmentation waypoint_publisher   --ros-args -r __ns:=/rm03   -p robot_id:=rm03   -p waypoints_file:=waypoints.yaml   -p dwell_sec:=5.0   -p loop:=false   -p run_id:=Horizontal
```

- **rm04**
```bash
ros2 run gnn_object_segmentation waypoint_publisher   --ros-args -r __ns:=/rm04   -p robot_id:=rm04   -p waypoints_file:=waypoints.yaml   -p dwell_sec:=5.0   -p loop:=false   -p run_id:=Horizontal
```

### Visualization (RViz2)
```bash
rviz2
```

Configs stored in:
```
ROS_GNN_ws/src/gnn_object_segmentation/configs
```

---

## 3️⃣ System Pipeline

### High-Level Flow

1. **Reference Pose Selection** – use EP03 Vicon pose as anchor  
2. **Temporal Alignment** – retrieve closest radar + poses for all robots  
3. **Radar Processing** – transform into GNN-compatible features + RViz visualization points  
4. **Noise Filtering (Optional)** – statistical outlier removal  
5. **Sliding Window Buffer** – maintain recent frames for temporal graphs  
6. **Graph Construction** – build nodes & edges with spatial-temporal features  
7. **Validation** – skip if graph is empty  
8. **Publish Graph** – send `GraphData` to `/graph_data`  
9. **Visualization** – update radar & robot poses in RViz  

### Flowchart

```
START
  |
  ├── Is EP03 pose available?
  │      └── No → Abort
  │      └── Yes → ref_timestamp
  |
  ├── Retrieve closest radar + poses
  ├── Filter + transform radar
  ├── Add to sliding buffer
  ├── Build graph from window
  ├── Publish GraphData
  └── Update RViz
```

---

## 4️⃣ Node Logic Overview

### 🔹 Data Merger Node (`gnn_object_segmentation` → `data_merge` executable)

**Purpose**  
Synchronize multi-robot radar + pose, build windowed graphs for GNN inference, and publish to `/graph_data`. Logs timing and graph stats to CSV for analysis.

**Subscriptions**
- `/rm04/ti_mmwave/radar_scan_pcl` (`sensor_msgs/PointCloud2`)
- `/rm03/ti_mmwave/radar_scan_pcl` (`sensor_msgs/PointCloud2`)
- `/rm04/vicon_pose` (`geometry_msgs/PoseWithCovarianceStamped`)
- `/rm03/vicon_pose` (`geometry_msgs/PoseWithCovarianceStamped`)
- TF: `map` ←→ `<robot>/base_link` (via `tf2_ros`)

**Publications**
- `/graph_data` (`gnn_interfaces/GraphData`) — flattened node/edge arrays, `frame_id: "map"`
- (Visualization helpers exist in code via `GraphVisualizer`, currently disabled by default in launch)

**Parameters**
- `run_id: string` — experiment/run label (used in logs & paths)
- `window_size: int` — number of frames used to build the temporal graph window (default in launch: 5)
- CLI flags parsed by the node:
  - `--visualize` (default `True` in code, controlled by launch arg)
  - `--simulation` (default `False`)

**Internal Buffers & Timing**
- Radar buffers: `deque(maxlen=1000)` per robot, recent ≤ **2.0 s**
- Pose buffers: `deque(maxlen=1000)` per robot (used for delays/logging)
- Merged frames buffer: `deque(maxlen=100)`
- **Fresh data requirement:** at least one radar topic with msg newer than **0.75 s**
- **Temporal window threshold:** include frames within **2.0 s** of current ref time
- **Buffer pruning:** old entries dropped after **1.5 s**

**Normalization & Features**
- Loads `normalization_weights_unified.pkl` at node startup
- Node features per radar point:
  - Position: **x, y, z**
  - Normalized: **snr**, **range**
  - Encoded angles: **sin/cos(azimuth)**, **sin/cos(elevation)**
  - **robot_id** (1 for `rm04`/`robot_1`, 2 for `rm03`/`robot_2`)
- Angle normalization uses sin/cos with mean/std from weights file

**Edge Construction (KNN)**
- Base K: **8** (reduced automatically if N < K)
- For each directed edge i→j:
  - `edge_attr = [dx, dy, dz, Δsnr_norm, Δt]`
- Produces:
  - `edge_index` shape **(2, E)** (flattened on publish)
  - `edge_attr` shape **(E, 5)**

**Radar → Global Transform & Filtering**
- Looks up TF: `map` ← `<robot>/base_link` at the radar timestamp
- Skips points too close to robot: **(x < 0.3 and y ≤ 0.3)**
- Global bounds kept (others skipped):
  - `-12.0 ≤ x ≤ 10.0`, `-5.0 ≤ y ≤ 7.0`
- Optional Statistical Outlier Removal (SOR) per robot if **≥ 15** points:
  - k=8, `std_ratio=1.0` (keeps points with mean-NN-distance < mean + std_ratio·std)

**Graph Publishing**
- `GraphData.header.frame_id = "map"`
- `node_features`: flattened `float32` list; `node_feature_dim = 10`
- `edge_index`: flattened `int64` list (2×E)
- `edge_attr`: flattened `float32` list; `edge_attr_dim = 5`
- `num_nodes`, `num_edges` filled from arrays
- Skips publish if no nodes or no edges

**CSV Logging**
- Folder: `datalogging/data_merge/<run_id>_<timestamp>/`
- File: `data_merger_log.csv`
- Appended per publish with:
  - `timestamp_ros, run_id, window_size, rm04_points, rm03_points, node_count, edge_count, inter_robot_edge_count, build_time_ms, merge_latency_ms, vicon_delay_rm04_ms, vicon_delay_rm03_ms`

**Key Methods (high level)**
- `pointcloud2_to_xyz_intensity` → (N×4) array
- `process_radar_points` → TF to map, bounds & proximity culling, metrics (range/angles), build dicts
- `get_recent_window_frames` → last N frames within 2.0s
- `build_graph_from_window` → assemble features, KNN edges
- `publish_graph` → populate `GraphData` message and publish
- `sync_and_merge` (timer @ 33Hz) → main loop to align, filter, build, log, publish, prune

---

### 🔹 GNN Inference Node (`gnn_inference` → `inference_node` executable)

**Purpose**  
Consumes `GraphData`, runs a trained GNN to classify nodes & edges, clusters nodes into objects, tracks them over time, and publishes polygons + RViz markers. Also logs inference stats for offline analysis.

**Subscriptions**
- `/graph_data` (`gnn_interfaces/GraphData`)

**Publications**
- `/gnn_objects` (`visualization_msgs/MarkerArray`) — outline polygons per detected object/class
- `/tracked_polygons` (`gnn_interfaces/TrackedPolygon`) — polygons + label, confidence, and contributor info

**Parameters**
- `run_id: string` — for logging directory and CSV filename

**Model/Config Files (expected in working dir)**  
- `configuration_flw_gnn_v1.yml` — model/config
- `graph_based_detector.pt` — trained weights

**Inference Pipeline**
1. **Tensor prep**: convert node/edge arrays to tensors on selected device.  
2. **Forward pass**: `detector()` returns node logits and edge logits.  
3. **Node class**: softmax → argmax → `pred_class` and confidences.  
4. **Edge class**: sigmoid on link-prob for edge[:,1]; threshold **0.91** → `pred_edge_class`.  
5. **Graph clusters**: build undirected graph using only edges with:  
   - same node class,  
   - `pred_edge_class == 1`,  
   - pairwise distance < **1.2 m**.  
6. **Polygon extraction**: for each connected component (with ≥6 pts and full-rank), compute **alpha shape** (`alpha=0.01`) to get boundary polygon; compute centroid and average confidence.  
7. **Shared perception**: compute per-object **contributor ratios** from node `robot_id` feature; attach to message.  
8. **Tracking**: per-class **Kalman trackers**, matched by centroid distance < **0.5 m**; trackers expire after **1.0 s** of no updates.  
9. **Publishing**:  
   - RViz `MarkerArray` outlines (LINE_STRIP).  
   - `TrackedPolygon` messages with `label`, `confidence`, `polygon`, `contributor_ids`, `contributor_ratios`.  
10. **Logging** (`datalogging/inference/<run_id>_<timestamp>/gnn_inference_log.csv`):  
    - `timestamp_ros, run_id, graph_timestamp, inference_latency_ms, cluster_count, object_class_distribution, mean_confidence, new_objects, removed_objects, gnn_pipeline_latency_ms, shared_objects_perc`.

**Alternate Processor (optional)**  
`process_predictions_v2`: DBSCAN on node positions class-wise, convex hull per cluster, and RViz labels. (Off by default.)

**Utilities**
- `class_to_color` — fixed palette → RGBA
- `create_pointcloud2` — helper for optional OctoMap (currently disabled)
- `publish_tracked_objects_as_pointcloud` — builds a labeled PointCloud2 (disabled by default)

---

### 🔹 Waypoint Publisher (`gnn_object_segmentation` → `waypoint_publisher` executable)

**Purpose**  
Sends sequential navigation goals to Nav2’s **`NavigateToPose`** action using waypoints from YAML. Supports per-run/per-robot waypoint selection, auto-fills yaw if missing, optional dwell time between goals, and looping.

**Action Client**
- Server name respects namespace: `/<ns>/navigate_to_pose` (e.g., `/rm03/navigate_to_pose`)

**Parameters**
- `run_id: string` — run label; used to pick a nested section from YAML (default: `R1-N1` in code)
- `robot_id: string` — `rm03`, `rm04`, … (default: `rm03`)
- `waypoints_file: string` — file name or path; searched in package share `gnn_object_segmentation/config/` if relative
- `dwell_sec: double` — pause between goals in wall time (default: `5.0`)
- `loop: bool` — repeat the sequence forever (default: `False`)
- `goal_timeout_sec: double` — cancel a goal if it exceeds this time (default: `120.0`)

**Waypoint YAML Structure (supported)**
1) **Nested by run/robot**
```yaml
Horizontal:
  rm03:
    - {name: wp1, x: 1.0, y: 0.0, theta: 0.0}
    - {name: wp2, x: 2.0, y: 1.0, theta: 1.57}
  rm04:
    - {name: wp1, x: -1.0, y: 0.5, theta: -1.57}
```
2) **Flat list**
```yaml
waypoints:
  - {name: A, x: 0.0, y: 0.0, theta: 0.0}
  - {name: B, x: 3.0, y: 0.0, theta: 0.0}
```
3) **Plain list**
```yaml
- {name: A, x: 0.0, y: 0.0, theta: 0.0}
- {name: B, x: 3.0, y: 0.0, theta: 0.0}
```

**Yaw Handling**
- If `theta` is `0` or missing, it is auto-filled using the path tangent (next or previous waypoint).

**Behavior**
- Loads waypoints once at startup; logs count.  
- For each waypoint: sends a `NavigateToPose` goal and waits synchronously for result.  
- If result not received within `goal_timeout_sec`, cancels the goal and proceeds.  
- Sleeps `dwell_sec` seconds (wall time) between goals.  
- If `loop:=false`, exits after one pass; otherwise, repeats.

**Waypoint Scenarios & Intended Paths (per `waypoints.yaml`)**

Below are the waypoint **scenarios** and what each is designed to test for our shared-perception experiments. Coordinates are in the `map` frame (meters). If a waypoint’s `theta` is zero/missing, the publisher auto-fills yaw from the path tangent.

- **R1-N1 — Straight Opposed Traverse**  
  - `rm03`: left-to-right sweep (x: −8 → +8, y ≈ 0).  
  - `rm04`: right-to-left sweep (x: +8 → −8, y ≈ 0).  
  - *Intent:* head-on, same corridor, to stress **opposing-traffic avoidance** via shared detections.

- **R1-N2 — Skewed Cross**  
  - `rm03`: diagonal up-right (≈ [−7,−3] → [7,3]).  
  - `rm04`: partial diagonal + return to origin (≈ [7,−3] → [0,0]).  
  - *Intent:* **angled crossing** and merge at a common rendezvous.

- **R1-N3 — Straight vs. Patrol Loop**  
  - `rm03`: straight pass (−8 → +8 along y≈0).  
  - `rm04`: right corridor patrol (to [8,0] → up to [0,3] → down to [0,−3] → back).  
  - *Intent:* **one passer, one patroller**; tests persistence and **spatio-temporal fusion**.



- **Diagonal — Dense Diagonal Weave (both robots)**  
  - `rm03` & `rm04`: long multi-point diagonals with alternating offsets.  
  - *Intent:* **high waypoint density** to evaluate **tracking continuity** and polygon stability under rapid viewpoint changes.

- **Horizontal — Lane Following (east–west)**  
  - `rm03` & `rm04`: multi-segment horizontal sweeps at several y-bands (≈ +3, +1, 0, −1, −2…).  
  - *Intent:* **parallel-lane coverage** and repeated passes for **map consistency** and **false-positive suppression**.

- **Vertical — Aisle Sweeps (north–south)**  
  - `rm03` & `rm04`: columnar sweeps with many turns along x-bands.  
  - *Intent:* **aisle/column coverage**; tests **edge-case TF alignment** and long-horizon tracking.

- **S4-F8 — Figure-Eight (mirrored)**  
  - Both robots trace loops that compose a figure-eight across quadrants.  
  - *Intent:* classic **crossing paths** with curvature; stresses **trajectory intersection handling** and **object identity maintenance**.

- **S5-NarrowLane — Tapered Corridor**  
  - `rm03`: climbs from bottom-left through a narrowing central lane to top-right.  
  - `rm04`: mirrored descent.  
  - *Intent:* **bottleneck/narrow-passage negotiation**, testing **shared situational awareness** at constrictions.

- **S6-PerimeterCross — Perimeter + Diagonal Cross**  
  - `rm03`: left perimeter up, across top, then diagonal to right; `rm04`: symmetric inverse.  
  - *Intent:* **boundary following** plus **mid-field crossing** for mixed context transitions.


**CLI Examples**
- **rm03**
```bash
ros2 run gnn_object_segmentation waypoint_publisher   --ros-args -r __ns:=/rm03   -p robot_id:=rm03   -p waypoints_file:=waypoints.yaml   -p dwell_sec:=5.0   -p loop:=false   -p run_id:=Horizontal
```
- **rm04**
```bash
ros2 run gnn_object_segmentation waypoint_publisher   --ros-args -r __ns:=/rm04   -p robot_id:=rm04   -p waypoints_file:=waypoints.yaml   -p dwell_sec:=5.0   -p loop:=false   -p run_id:=Horizontal
```

---

### 🔹 Launch: `multi_robot_inference.launch.py`
Starts rosbag recording, the data merger, and visualization helpers.

**Launch Arguments**
- `visualize` (default `False`) — enable RViz/markers
- `simulation` (default `False`)
- `rviz_config` (default `src/gnn_object_segmentation/rviz/flw_hall_gnn.rviz`)
- `run_id` (default `default_run`)

**Rosbag Recording**
- Output directory: `datalogging/rosbags/<run_id>_<YYYYMMDD_HHMMSS>_bag`
- Topics recorded (subset shown as in file):
  - `clock`
  - `/rm03/ti_mmwave/radar_scan_pcl`, `/rm04/ti_mmwave/radar_scan_pcl`
  - `/tf`, `/tf_static`
  - `rm03/odom`, `rm04/odom`
  - `/rm03/vicon_pose`, `/rm04/vicon_pose`
  - `/graph_data`, `/tracked_polygons`, `gnn_objects`
  - `/rm03/global_costmap/costmap_raw`, `/rm04/global_costmap/costmap_raw`
  - `/navigate_to_pose/feedback`, `/navigate_to_pose/result`
  - `/rm03/plan`, `/rm04/plan`, `/rm03/path`, `/rm04/path`
  - `/rm03/cmd_vel`, `/rm04/cmd_vel`
  - `/rm03/behavior_tree_log`, `/rm04/behavior_tree_log`

**Nodes**
- **Data Merger**
  - `package='gnn_object_segmentation'`, `executable='data_merge'`, `name='data_merge'`
  - Parameters: `{run_id, window_size=5}`
  - Arguments: `--visualize`, `--simulation` (from launch args)
- **Tracked Polygon Marker Publisher**
  - `package='gnn_object_segmentation'`, `executable='tracked_polygon_visualizer'`
  - Parameters: `input_topic=/tracked_polygons`, `output_topic=/tracked_polygon_markers`
- *(Optional commented blocks for RViz and arena markers are present in the launch file.)*

---

## 5️⃣ Message Definitions (custom)

### `gnn_interfaces/TrackedPolygon.msg`
```
std_msgs/Header header
geometry_msgs/Polygon polygon
uint32 label
float32 confidence

uint32[] contributor_ids     # robot IDs that contributed to this object
float32[] contributor_ratios # normalized ratios per contributor
```

### `gnn_object_segmentation/GraphData.msg`
```
std_msgs/Header header
# Tensor-like flat arrays with size metadata
float32[] node_features       # flattened [N x 10]
uint32 node_feature_dim       # always 10

int64[] edge_index            # flattened [2 x E]
float32[] edge_attr           # flattened [E x 5]
uint32 edge_attr_dim          # always 5

# Optional metadata
uint32 num_nodes
uint32 num_edges
```

---

## 6️⃣ Parameters (quick reference)

- `run_id` → experiment ID (e.g., `Horizontal`)  
- `robot_id` → namespace ID (`rm03`, `rm04`)  
- `waypoints_file` → YAML with navigation waypoints  
- `dwell_sec` → pause at waypoint  
- `loop` → loop through waypoints or not  
- `window_size` → temporal window length for graph (default 5 via launch)  
- `visualize` / `simulation` → launch flags forwarding to node args

---

## 7️⃣ Outputs

- `/graph_data` → `GraphData.msg` (node features, edge index, edge attributes)  
- `/gnn_objects` → `MarkerArray` outlines of detected objects  
- `/tracked_polygons` → `TrackedPolygon` messages (class, confidence, polygon, contributor info)  
- **Logs**:  
  - `datalogging/data_merge/<run_id>_<timestamp>/data_merger_log.csv`  
  - `datalogging/inference/<run_id>_<timestamp>/gnn_inference_log.csv`

---

## 8️⃣ Notes & Assumptions

- TF tree must provide `map` ↔ `<robot>/base_link` for `rm03` and `rm04`. This would be done via the vicon pose publisher node.
- Normalization weights file `normalization_weights_unified.pkl` must be present for the **Data Merger** node.
- Inference node expects `configuration_flw_gnn_v1.yml` and `graph_based_detector.pt` in its working directory.
- Waypoint publisher searches `gnn_object_segmentation/config/` if `waypoints_file` is not an absolute path.
- Bounds and thresholds (proximity/bounds/SOR/window/edge distance/edge threshold) are tuned for the current arena; adjust as needed.
- If neither radar stream is fresh (older than 0.75s), graph construction is skipped for that tick.





