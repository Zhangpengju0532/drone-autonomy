import math

import numpy as np

import world

ORDER = 8
CRUISE_SPEED = 1.6
MIN_SEGMENT_TIME = 1.2


def derivative_row(t, order):
    row = np.zeros(ORDER)
    for k in range(order, ORDER):
        row[k] = math.factorial(k) / math.factorial(k - order) * t ** (k - order)
    return row


def segment_times(points):
    times = []
    last = len(points) - 2
    for i, (a, b) in enumerate(zip(points, points[1:])):
        duration = max(MIN_SEGMENT_TIME, float(np.linalg.norm(b - a)) / CRUISE_SPEED)
        if i == 0 or i == last:
            duration += 1.0
        times.append(duration)
    return times


class Trajectory:
    def __init__(self, points, times=None):
        self.points = [np.array(p, dtype=float) for p in points]
        self.times = segment_times(self.points) if times is None else list(times)
        self.starts = np.concatenate([[0.0], np.cumsum(self.times)])
        self.duration = float(self.starts[-1])
        n = len(self.times)
        A = np.zeros((ORDER * n, ORDER * n))
        b = np.zeros((ORDER * n, 3))
        row = 0
        for i in range(n):
            cols = slice(ORDER * i, ORDER * (i + 1))
            A[row, cols] = derivative_row(0.0, 0)
            b[row] = self.points[i]
            row += 1
            A[row, cols] = derivative_row(self.times[i], 0)
            b[row] = self.points[i + 1]
            row += 1
        for order in (1, 2, 3):
            A[row, 0:ORDER] = derivative_row(0.0, order)
            row += 1
            A[row, ORDER * (n - 1):ORDER * n] = derivative_row(self.times[-1], order)
            row += 1
        for i in range(n - 1):
            for order in range(1, 7):
                A[row, ORDER * i:ORDER * (i + 1)] = derivative_row(self.times[i], order)
                A[row, ORDER * (i + 1):ORDER * (i + 2)] = -derivative_row(0.0, order)
                row += 1
        self.coefficients = np.linalg.solve(A, b).reshape(n, ORDER, 3)

    def evaluate(self, t):
        t = min(max(t, 0.0), self.duration)
        i = min(int(np.searchsorted(self.starts, t, side="right")) - 1, len(self.times) - 1)
        local = t - self.starts[i]
        c = self.coefficients[i]
        return tuple(derivative_row(local, order) @ c for order in (0, 1, 2))

    def sample(self, step=0.02):
        ts = np.arange(0.0, self.duration + step / 2, step)
        values = [self.evaluate(t) for t in ts]
        return ts, np.array([v[0] for v in values]), np.array([v[1] for v in values]), np.array([v[2] for v in values])

    def clearance(self):
        ts, pos, _, _ = self.sample(0.05)
        gaps = np.array([world.distance_to_obstacles(p) for p in pos])
        return float(gaps.min()), float(ts[int(gaps.argmin())])


def smooth(points, wanted_clearance):
    points = [np.array(p, dtype=float) for p in points]
    for rounds in range(12):
        candidate = Trajectory(points)
        gap, when = candidate.clearance()
        if gap >= wanted_clearance:
            return candidate, rounds
        i = min(int(np.searchsorted(candidate.starts, when, side="right")) - 1, len(points) - 2)
        points.insert(i + 1, (points[i] + points[i + 1]) / 2)
    return candidate, rounds