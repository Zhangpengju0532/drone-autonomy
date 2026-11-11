import math

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import drone
import ground

FRAME_STEP = 10
FPS = 15
ARM = 0.8
ROTOR = 0.36
CAR_LENGTH = 4.5
CAR_WIDTH = 1.8
CAR_HEIGHT = 1.4
BLUE = "#2a78d6"


def frame_shapes(pos, angles):
    r = drone.rotation(angles)
    tips = [pos + r @ np.array([ARM * math.cos(a), ARM * math.sin(a), 0.0]) for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4)]
    arms = np.array([tips[0], tips[2], pos, tips[1], tips[3]])
    ring = np.linspace(0, 2 * math.pi, 13)
    rotors = [np.array([tip + r @ np.array([ROTOR * math.cos(a), ROTOR * math.sin(a), 0.0]) for a in ring]) for tip in tips]
    return arms, rotors


def car_outline(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(-0.9, -CAR_WIDTH / 2), (CAR_LENGTH - 0.9, -CAR_WIDTH / 2), (CAR_LENGTH - 0.9, CAR_WIDTH / 2), (-0.9, CAR_WIDTH / 2)]
    base = [np.array([x + c * px - s * py, y + s * px + c * py]) for px, py in corners]
    order = [0, 1, 2, 3, 0]
    low = np.array([[base[k][0], base[k][1], 0.0] for k in order])
    high = np.array([[base[k][0], base[k][1], CAR_HEIGHT] for k in order])
    posts = [np.array([[base[k][0], base[k][1], 0.0], [base[k][0], base[k][1], CAR_HEIGHT]]) for k in range(4)]
    return [low, high] + posts


def main():
    log = np.load("results/follow_log.npz")["log"]
    frames = len(log) // FRAME_STEP
    path = ground.road()
    x0, y0, z0, x1, y1, z1 = ground.BRIDGE

    fig = plt.figure(figsize=(8.4, 3.9))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.93], projection="3d")
    ax.plot(path[:, 0], path[:, 1], np.zeros(len(path)), color="0.72", linewidth=6, solid_capstyle="round")
    ax.bar3d(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0, color="0.55", alpha=0.45, edgecolor="0.3", linewidth=0.5, shade=False)
    for px, py in ((x0, y0), (x0, y1 - 0.4)):
        ax.bar3d(px, py, 0.0, x1 - x0, 0.4, z0, color="0.55", alpha=0.45, edgecolor="0.3", linewidth=0.5, shade=False)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 20)
    ax.set_zlim(0, 6)
    ax.set_box_aspect((40, 20, 6), zoom=1.4)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 10, 20])
    ax.set_zticks([0, 3, 6])
    ax.view_init(elev=30, azim=-60)
    trail, = ax.plot([], [], [], color=BLUE, linewidth=1.6)
    tether, = ax.plot([], [], [], color="0.6", linewidth=0.9, linestyle=(0, (2, 2)))
    arms, = ax.plot([], [], [], color="0.1", linewidth=2.0)
    rotors = [ax.plot([], [], [], color=BLUE, linewidth=1.5)[0] for _ in range(4)]
    car_lines = [ax.plot([], [], [], color="0.1", linewidth=1.6)[0] for _ in range(6)]
    title = fig.text(0.03, 0.96, "", fontsize=10, color="0.1", va="center")

    def update(frame):
        i = frame * FRAME_STEP
        t, cx, cy, cyaw, speed = log[i, 0:5]
        pos, angles = log[i, 6:9], log[i, 9:12]
        trail.set_data_3d(log[:i + 1, 6], log[:i + 1, 7], log[:i + 1, 8])
        tether.set_data_3d([pos[0], pos[0]], [pos[1], pos[1]], [CAR_HEIGHT, pos[2]])
        arm_shape, rotor_shapes = frame_shapes(pos, angles)
        arms.set_data_3d(arm_shape[:, 0], arm_shape[:, 1], arm_shape[:, 2])
        for line, shape in zip(rotors, rotor_shapes):
            line.set_data_3d(shape[:, 0], shape[:, 1], shape[:, 2])
        for line, shape in zip(car_lines, car_outline(cx, cy, cyaw)):
            line.set_data_3d(shape[:, 0], shape[:, 1], shape[:, 2])
        title.set_text("t = %4.1f s   car speed %.1f m/s   vehicle height %.1f m   offset %.2f m" % (
            t, speed, pos[2], math.hypot(pos[0] - cx, pos[1] - cy)))
        return []

    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save("results/follow.gif", writer=PillowWriter(fps=FPS), dpi=90)
    print("animation written to results/follow.gif,", frames, "frames")


main()