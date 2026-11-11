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
ASSUMED_ACCEL_NOISE = 0.3


def main():
    rng = np.random.default_rng(SEED)
    t = np.arange(0.0, DURATION + DT / 2, DT)
    true_height = 2.0 + 0.5 * np.sin(0.5 * t)
    true_speed = 0.25 * np.cos(0.5 * t)
    true_accel = -0.125 * np.sin(0.5 * t)
    gps = true_height[::GPS_EVERY] + rng.normal(0.0, GPS_NOISE, len(t[::GPS_EVERY]))
    accel = true_accel + ACCEL_BIAS + rng.normal(0.0, ACCEL_NOISE, len(t))

    x = np.array([gps[0], 0.0])
    P = np.diag([GPS_NOISE ** 2, 1.0])
    F = np.array([[1.0, DT], [0.0, 1.0]])
    B = np.array([0.5 * DT * DT, DT])
    Q = np.outer(B, B) * ASSUMED_ACCEL_NOISE ** 2
    H = np.array([1.0, 0.0])
    R = GPS_NOISE ** 2
    estimate = []
    spread = []
    gains = []
    for i in range(len(t)):
        if i % GPS_EVERY == 0:
            K = P @ H / (H @ P @ H + R)
            x = x + K * (gps[i // GPS_EVERY] - H @ x)
            P = P - np.outer(K, H @ P)
            gains.append(K[0])
        estimate.append(x[0])
        spread.append(np.sqrt(P[0, 0]))
        x = F @ x + B * accel[i]
        P = F @ P @ F.T + Q
    estimate = np.array(estimate)
    spread = np.array(spread)

    settled = t >= 5.0
    gps_error = gps - true_height[::GPS_EVERY]
    kalman_error = estimate - true_height
    print("after the first 5 s:")
    print("  GPS alone            rms error %.3f m, worst %.2f m" % (np.sqrt((gps_error[5 * 10:] ** 2).mean()), np.abs(gps_error[5 * 10:]).max()))
    print("  Kalman filter        rms error %.3f m, worst %.2f m" % (np.sqrt((kalman_error[settled] ** 2).mean()), np.abs(kalman_error[settled]).max()))
    print("  accelerometer alone  error at 30 s: about 22 m (lesson 1)")
    print("weight given to each GPS reading: first %.2f, after 1 s %.2f, at the end %.3f" % (gains[0], gains[10], gains[-1]))
    print("the filter's own estimate of its uncertainty at the end: %.2f m" % spread[-1])

    os.makedirs("results", exist_ok=True)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    top.plot(t[::GPS_EVERY], gps, linestyle="none", marker="o", markersize=2.5, color="#eb6834", alpha=0.6, label="GPS readings")
    top.plot(t, true_height, color="0.1", linewidth=1.8, label="true height")
    top.plot(t, estimate, color="#2a78d6", linewidth=1.8, label="Kalman filter")
    top.set_ylabel("height (m)")
    top.set_ylim(0, 5.2)
    top.legend(frameon=False, fontsize=9, loc="upper right", ncol=3)
    top.set_title("Fusing GPS and accelerometer", fontsize=10, loc="left")
    bottom.axhline(0, color="0.45", linewidth=1)
    bottom.fill_between(t, -2 * spread, 2 * spread, color="#2a78d6", alpha=0.15, linewidth=0, label="filter's own 2-sigma band")
    bottom.plot(t, kalman_error, color="#2a78d6", linewidth=1.5, label="actual error")
    bottom.set_ylabel("error (m)")
    bottom.set_xlabel("time (s)")
    bottom.set_ylim(-0.9, 1.1)
    bottom.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    for ax in (top, bottom):
        ax.grid(color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/s2_kalman_1d.png", dpi=200)
    print("figure written to results/s2_kalman_1d.png")


main()