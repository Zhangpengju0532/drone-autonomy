import math

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import drone
import swarm

FRAME_STEP = 2
FPS = 15
ARM = 0.45
ROTOR = 0.22
TRAIL = 60
COLORS = ["#2a78d6", "#eb6834", "#1baf6c", "#8a5cd0", "#d6453d", "#3d3d3d"]


def frame_shapes(pos, angles):
    r = drone.rotation(angles)
    tips = [pos + r @ np.array([ARM * math.cos(a), ARM * math.sin(a), 0.0]) for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4)]
    arms = np.array([tips[0], tips[2], pos, tips[1], tips[3]])
    ring = np.linspace(0, 2 * math.pi, 11)
    rotors = [np.array([tip + r @ np.array([ROTOR * math.cos(a), ROTOR * math.sin(a), 0.0]) for a in ring]) for tip in tips]
    return arms, rotors


class Scene:
    def __init__(self, fig, rect, log, count, limits, title):
        self.t = log[:, 0]
        self.positions = swarm.positions_of(log, count)
        self.angles = swarm.angles_of(log, count)
        self.count = count
        self.ax = fig.add_axes(rect, projection="3d")
        (x0, x1), (y0, y1) = limits
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_zlim(0, 4)
        self.ax.set_box_aspect((x1 - x0, y1 - y0, 4), zoom=0.9)
        self.ax.set_xticks([5, 10, 15])
        self.ax.set_yticks([5, 10, 15])
        self.ax.set_zticks([0, 2, 4])
        self.ax.view_init(elev=42, azim=-60)
        self.ax.set_title(title, fontsize=10, loc="left", y=0.98)
        self.trails = [self.ax.plot([], [], [], color=COLORS[i], linewidth=1.3, alpha=0.8)[0] for i in range(count)]
        self.shadows = [self.ax.plot([], [], [], color="0.75", linewidth=1.0)[0] for i in range(count)]
        self.arms = [self.ax.plot([], [], [], color="0.1", linewidth=1.5)[0] for i in range(count)]
        self.rotors = [[self.ax.plot([], [], [], color=COLORS[i], linewidth=1.4)[0] for _ in range(4)] for i in range(count)]

    def draw(self, frame):
        k = min(frame * FRAME_STEP, len(self.t) - 1)
        a = max(0, k - TRAIL)
        for i in range(self.count):
            p = self.positions[a:k + 1, i]
            self.trails[i].set_data_3d(p[:, 0], p[:, 1], p[:, 2])
            self.shadows[i].set_data_3d(p[:, 0], p[:, 1], np.zeros(len(p)))
            arms, rotors = frame_shapes(self.positions[k, i], self.angles[k, i])
            self.arms[i].set_data_3d(arms[:, 0], arms[:, 1], arms[:, 2])
            for line, shape in zip(self.rotors[i], rotors):
                line.set_data_3d(shape[:, 0], shape[:, 1], shape[:, 2])


def main():
    data = np.load("results/swarm_log.npz")
    fig = plt.figure(figsize=(9.6, 4.2))
    left = Scene(fig, [0.0, 0.0, 0.47, 0.95], data["swap_on"], 6, ((3, 17), (3, 17)), "Six vehicles swap places, avoidance on")
    right = Scene(fig, [0.47, 0.0, 0.53, 0.95], data["formation"], 5, ((0, 20), (2, 18)), "Formation: line, wedge, ring")
    frames = max(len(data["swap_on"]), len(data["formation"])) // FRAME_STEP

    def update(frame):
        left.draw(frame)
        right.draw(frame)
        return []

    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save("results/swarm.gif", writer=PillowWriter(fps=FPS), dpi=85)
    print("animation written to results/swarm.gif,", frames, "frames")


main()