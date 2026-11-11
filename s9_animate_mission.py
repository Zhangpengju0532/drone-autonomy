import math

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import drone
import world

FRAME_STEP = 10
FPS = 15
ARM = 0.7
ROTOR = 0.33
GPS_MEMORY = 3.0
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def frame_shapes(pos, angles):
    r = drone.rotation(angles)
    tips = [pos + r @ np.array([ARM * math.cos(a), ARM * math.sin(a), 0.0]) for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4)]
    arms = [np.array([tips[0], tips[2]]), np.array([tips[1], tips[3]])]
    ring = np.linspace(0, 2 * math.pi, 13)
    rotors = [np.array([tip + r @ np.array([ROTOR * math.cos(a), ROTOR * math.sin(a), 0.0]) for a in ring]) for tip in tips]
    return arms, rotors


def main():
    data = np.load("results/mission_log.npz")
    log, fixes = data["log"], data["fixes"]
    t = log[:, 0]
    true, ref, est, angles = log[:, 1:4], log[:, 4:7], log[:, 7:10], log[:, 10:13]
    frames = len(log) // FRAME_STEP

    fig = plt.figure(figsize=(7.0, 4.6))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.93], projection="3d")
    for x0, y0, z0, x1, y1, z1 in world.BOXES:
        ax.bar3d(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0, color="0.6", alpha=0.28, edgecolor="0.35", linewidth=0.4, shade=False)
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="0.3", linewidth=0.9, linestyle=(0, (4, 3)))
    ax.scatter(*world.GOAL, color="0.1", s=40, marker="*")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_zlim(0, 6)
    ax.set_box_aspect((20, 20, 6), zoom=1.12)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_zticks([0, 3, 6])
    ax.view_init(elev=38, azim=-58)
    trail, = ax.plot([], [], [], color=BLUE, linewidth=1.8)
    dots, = ax.plot([], [], [], linestyle="none", marker="o", markersize=2.5, color=ORANGE, alpha=0.8)
    arms = [ax.plot([], [], [], color="0.1", linewidth=2.0)[0] for _ in range(2)]
    rotors = [ax.plot([], [], [], color=BLUE, linewidth=1.5)[0] for _ in range(4)]
    title = fig.text(0.03, 0.96, "", fontsize=10, color="0.1", va="center")
    fig.text(0.97, 0.96, "orange: raw GPS   blue: flown   dashed: planned", fontsize=8, color="0.35", va="center", ha="right")

    def update(frame):
        i = frame * FRAME_STEP
        trail.set_data_3d(true[:i + 1, 0], true[:i + 1, 1], true[:i + 1, 2])
        recent = fixes[(fixes[:, 0] <= t[i]) & (fixes[:, 0] > t[i] - GPS_MEMORY)]
        dots.set_data_3d(recent[:, 1], recent[:, 2], np.clip(recent[:, 3], 0, 6))
        arm_shapes, rotor_shapes = frame_shapes(true[i], angles[i])
        for line, shape in zip(arms, arm_shapes):
            line.set_data_3d(shape[:, 0], shape[:, 1], shape[:, 2])
        for line, shape in zip(rotors, rotor_shapes):
            line.set_data_3d(shape[:, 0], shape[:, 1], shape[:, 2])
        title.set_text("t = %4.1f s   estimate error %.2f m   clearance %.2f m" % (
            t[i], np.linalg.norm(est[i] - true[i]), world.distance_to_obstacles(true[i])))
        return []

    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save("results/mission.gif", writer=PillowWriter(fps=FPS), dpi=90)
    print("animation written to results/mission.gif,", frames, "frames")


main()