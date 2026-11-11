import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DURATION = 30.0
DT = 0.01
GPS_EVERY = 10
GPS_NOISE = 0.7
ACCEL_NOISE = 0.2
ACCEL_BIAS = 0.05
SEED = 1


def main():
    rng = np.random.default_rng(SEED)
    t = np.arange(0.0, DURATION + DT / 2, DT)
    true_height = 2.0 + 0.5 * np.sin(0.5 * t)
    true_speed = 0.25 * np.cos(0.5 * t)
    true_accel = -0.125 * np.sin(0.5 * t)

    gps_t = t[::GPS_EVERY]
    gps = true_height[::GPS_EVERY] + rng.normal(0.0, GPS_NOISE, len(gps_t))

    accel = true_accel + ACCEL_BIAS + rng.normal(0.0, ACCEL_NOISE, len(t))
    speed = true_speed[0]
    height = true_height[0]
    integrated = []
    for a in accel:
        integrated.append(height)
        speed += a * DT
        height += speed * DT
    integrated = np.array(integrated)

    print("time (s)  true height (m)  GPS reading (m)  GPS error (m)  integrated accelerometer (m)  its error (m)")
    for moment in (1.0, 5.0, 10.0, 20.0, 30.0):
        i = int(round(moment / DT))
        j = i // GPS_EVERY
        print("%8.0f  %15.2f  %15.2f  %13.2f  %28.2f  %13.2f" % (
            moment, true_height[i], gps[j], gps[j] - true_height[i], integrated[i], integrated[i] - true_height[i]))

    os.makedirs("results", exist_ok=True)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True)
    top.plot(gps_t, gps, linestyle="none", marker="o", markersize=2.5, color="#eb6834", label="GPS readings")
    top.plot(t, true_height, color="0.1", linewidth=1.8, label="true height")
    top.set_ylabel("height (m)")
    top.set_title("GPS: noisy at every instant, but it never wanders off", fontsize=10, loc="left")
    top.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    bottom.plot(t, integrated, color="#2a78d6", linewidth=1.8, label="height from integrating the accelerometer")
    bottom.plot(t, true_height, color="0.1", linewidth=1.8, label="true height")
    bottom.set_ylabel("height (m)")
    bottom.set_xlabel("time (s)")
    bottom.set_title("Accelerometer: smooth, but the error keeps growing", fontsize=10, loc="left")
    bottom.legend(frameon=False, fontsize=9, loc="upper left")
    for ax in (top, bottom):
        ax.grid(color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/s1_sensor_noise.png", dpi=200)
    print("figure written to results/s1_sensor_noise.png")


main()