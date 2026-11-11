import numpy as np

import drone

GPS_EVERY = 50
COMPASS_EVERY = 10
GPS_NOISE = np.array([0.5, 0.5, 0.7])
ACCEL_NOISE = 0.2
ACCEL_BIAS = np.array([0.05, -0.04, 0.06])
GYRO_NOISE = 0.01
GYRO_BIAS = np.array([0.010, -0.008, 0.005])
COMPASS_NOISE = np.radians(3.0)


class Sensors:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)

    def accelerometer(self, vehicle):
        r = drone.rotation(vehicle.angles)
        specific_force = r.T @ (vehicle.accel - drone.DOWN)
        return specific_force + ACCEL_BIAS + self.rng.normal(0.0, ACCEL_NOISE, 3)

    def gyroscope(self, vehicle):
        return vehicle.rates + GYRO_BIAS + self.rng.normal(0.0, GYRO_NOISE, 3)

    def gps(self, vehicle):
        return vehicle.pos + self.rng.normal(0.0, 1.0, 3) * GPS_NOISE

    def compass(self, vehicle):
        return vehicle.angles[2] + self.rng.normal(0.0, COMPASS_NOISE)