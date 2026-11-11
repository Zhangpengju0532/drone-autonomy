import numpy as np

import drone
import sensors

ASSUMED_ACCEL_NOISE = 0.5
ASSUMED_BIAS_DRIFT = 0.002
INITIAL_BIAS_DOUBT = 0.2


class PositionFilter:
    def __init__(self, first_gps):
        dt = drone.DT
        self.x = np.zeros((3, 3))
        self.x[:, 0] = first_gps
        self.P = np.array([np.diag([sensors.GPS_NOISE[axis] ** 2, 1.0, INITIAL_BIAS_DOUBT ** 2]) for axis in range(3)])
        self.F = np.array([[1.0, dt, -0.5 * dt * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0]])
        self.B = np.array([0.5 * dt * dt, dt, 0.0])
        self.Q = np.outer(self.B, self.B) * ASSUMED_ACCEL_NOISE ** 2
        self.Q[2, 2] = ASSUMED_BIAS_DRIFT ** 2 * dt
        self.H = np.array([1.0, 0.0, 0.0])

    def predict(self, accel_reading, angles):
        world_accel = drone.rotation(angles) @ accel_reading + drone.DOWN
        for axis in range(3):
            self.x[axis] = self.F @ self.x[axis] + self.B * world_accel[axis]
            self.P[axis] = self.F @ self.P[axis] @ self.F.T + self.Q

    def correct(self, gps_reading):
        for axis in range(3):
            P = self.P[axis]
            K = P @ self.H / (self.H @ P @ self.H + sensors.GPS_NOISE[axis] ** 2)
            self.x[axis] = self.x[axis] + K * (gps_reading[axis] - self.H @ self.x[axis])
            self.P[axis] = P - np.outer(K, self.H @ P)

    @property
    def pos(self):
        return self.x[:, 0].copy()

    @property
    def vel(self):
        return self.x[:, 1].copy()

    @property
    def bias(self):
        return self.x[:, 2].copy()