# kalman_tracker.py
import numpy as np

class ObjectTracker:
    def __init__(self, initial_pos, init_time=None, dt=0.1):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])
        self.last_update_time = init_time
        self.P = np.eye(4) * 0.1
        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.R = np.eye(2) * 0.05
        self.Q = np.eye(4) * 0.01
        self.last_seen = init_time if init_time else 0.0
        self.shape = None  # ← Added

    def predict(self, current_time=None):
        # If too much time passed since last seen, dampen the velocity
        if current_time is not None and self.last_seen is not None:
            time_since_update = current_time - self.last_seen
            if time_since_update > 0.5:  # or 1.0 sec, depending on desired decay
                self.x[2:] *= 0.5  # reduce velocity by half
            if time_since_update > 1.5:
                self.x[2:] = 0.0  # stop completely

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]


    def update(self, measurement, current_time=None):
        z = np.array(measurement)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        if current_time is not None:
            self.last_seen = current_time
        return self.x[:2]


    def get_state(self):
        return self.x[:2]
