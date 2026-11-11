import numpy as np

SIZE = np.array([20.0, 20.0, 6.0])
CELL = 0.5
SAFETY = 0.5
BOXES = [
    (0.0, 9.5, 0.0, 8.0, 10.5, 6.0),
    (10.5, 9.5, 0.0, 20.0, 10.5, 6.0),
    (8.0, 9.5, 3.5, 10.5, 10.5, 6.0),
    (12.0, 14.0, 0.0, 19.0, 14.8, 2.5),
    (3.0, 3.5, 0.0, 4.2, 4.7, 6.0),
    (6.5, 6.0, 0.0, 7.7, 7.2, 6.0),
    (10.0, 3.0, 0.0, 11.2, 4.2, 6.0),
    (13.5, 6.5, 0.0, 14.7, 7.7, 6.0),
    (16.5, 2.5, 0.0, 17.7, 3.7, 6.0),
    (4.0, 13.0, 0.0, 5.2, 14.2, 6.0),
    (8.5, 15.5, 0.0, 9.7, 16.7, 6.0),
    (15.0, 17.0, 0.0, 16.2, 18.2, 4.0),
    (2.0, 17.0, 0.0, 3.2, 18.2, 6.0),
]
START = np.array([1.5, 1.5, 1.0])
GOAL = np.array([18.5, 18.5, 2.0])


def shape():
    return tuple(int(round(v)) for v in SIZE / CELL)


def cell_of(point):
    return tuple(int(v) for v in np.floor(np.asarray(point) / CELL))


def centre_of(cell):
    return (np.array(cell) + 0.5) * CELL


def occupancy(safety=SAFETY):
    nx, ny, nz = shape()
    grid = np.zeros((nx, ny, nz), dtype=bool)
    xs = (np.arange(nx) + 0.5) * CELL
    ys = (np.arange(ny) + 0.5) * CELL
    zs = (np.arange(nz) + 0.5) * CELL
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    for x0, y0, z0, x1, y1, z1 in BOXES:
        grid |= ((X > x0 - safety) & (X < x1 + safety) & (Y > y0 - safety) & (Y < y1 + safety)
                 & (Z > z0 - safety) & (Z < z1 + safety))
    grid |= Z < safety
    return grid


def distance_to_obstacles(point):
    best = float(point[2])
    for x0, y0, z0, x1, y1, z1 in BOXES:
        gap = np.maximum(np.maximum(np.array([x0, y0, z0]) - point, point - np.array([x1, y1, z1])), 0.0)
        best = min(best, float(np.linalg.norm(gap)))
    return best