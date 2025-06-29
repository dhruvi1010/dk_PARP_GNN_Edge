# gnn_object_segmentation/ros_fusion_buffer.py

from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R

class RadarPoseBuffer:
    def __init__(self, history_len=5):
        self.history_len = history_len
        self.radar_data = {
            'ep03': deque(maxlen=history_len),
            'ep05': deque(maxlen=history_len)
        }
        self.poses = {'ep03': None, 'ep05': None}

    def update_pose(self, robot_id, pose_msg):
        self.poses[robot_id] = pose_msg

    def add_radar_frame(self, robot_id, pcl_np_array, timestamp):
        """
        pcl_np_array: NxD numpy array with fields like x, y, z, snr, range, azim, elev
        timestamp: float (sec)
        """
        if self.poses[robot_id] is None:
            return False

        tf = self.poses[robot_id].transform
        t = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        q = np.array([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])
        R_mat = R.from_quat(q).as_matrix()

        # Transform to global frame
        points_xyz = pcl_np_array[:, :3]
        points_global = (R_mat @ points_xyz.T).T + t

        pcl_np_array[:, :3] = points_global
        robot_id_col = np.full((pcl_np_array.shape[0], 1), 3 if robot_id == 'ep03' else 5)
        timestamp_col = np.full((pcl_np_array.shape[0], 1), timestamp)

        full = np.hstack([pcl_np_array, robot_id_col, timestamp_col])
        self.radar_data[robot_id].append(full)
        return True

    def get_combined_window(self):
        combined = []
        for robot_id in ['ep03', 'ep05']:
            for frame in self.radar_data[robot_id]:
                combined.append(frame)
        if not combined:
            return None
        return np.vstack(combined)
