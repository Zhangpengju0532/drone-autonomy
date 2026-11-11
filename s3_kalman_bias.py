import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DURATION = 60.0
DT = 0.01
GPS_EVERY = 10
GPS_NOISE = 0.7
ACCEL_NOISE = 0.2
ACCEL_BIAS = 0.05
SEED = 1
ASSUMED_ACCEL_NOISE = 0.3
ASSUMED_BIAS_DRIFT = 0.001
INITIAL_BIAS_DOUBT = 0.2


def run_filter(t, gps, accel, with_bias):
    x = np.array([gps[0], 0.0, 0.0])
    P = np.diag([GPS_NOISE ** 2, 1.0, INITIAL_BIAS_DOUBT ** 2 if with_bias else 0.0])
    F = np.array([[1.0, DT, -0.5 * DT * DT], [0.0, 1.0, -DT], [0.0, 0.0, 1.0]])
    B = np.array([0.5 * DT * DT, DT, 0.0])
    Q = np.outer(B, B) * ASSUMED_ACCEL_NOISE ** 2
    if with_bias:
        Q[2, 2] = ASSUMED_BIAS_DRIFT ** 2 * DT
    H = np.array([1.0, 0.0, 0.0])
    R = GPS_NOISE ** 2
    heights = []
    biases = []
    for i in range(len(t)):
        if i % GPS_EVERY == 0:
            K = P @ H / (H @ P @ H + R)
            x = x + K * (gps[i // GPS_EVERY] - H @ x)
            P = P - np.outer(K, H @ P)
        heights.append(x[0])
        biases.append(x[2])
        x = F @ x + B * accel[i]
        P = F @ P @ F.T + Q
    return np.array(heights), np.array(biases)


def main():
    rng = np.random.default_rng(SEED)
    t = np.arange(0.0, DURATION + DT / 2, DT)
    true_height = 2.0 + 0.5 * np.sin(0.5 * t)
    true_accel = -0.125 * np.sin(0.5 * t)
    gps = true_height[::GPS_EVERY] + rng.normal(0.0, GPS_NOISE, len(t[::GPS_EVERY]))
    accel = true_accel + ACCEL_BIAS + rng.normal(0.0, ACCEL_NOISE, len(t))

    plain, _ = run_filter(t, gps, accel, with_bias=False)
    aware, bias = run_filter(t, gps, accel, with_bias=True)

    late = t >= 30.0
    print("second half of the flight (30 s to 60 s):")
    for name, estimate in (("filter that ignores the bias", plain), ("filter that estimates the bias", aware)):
        error = estimate - true_height
        print("  %-31s mean error %+.3f m, rms error %.3f m" % (name, error[late].mean(), np.sqrt((error[late] ** 2).mean())))
    print("true accelerometer bias: %.3f" % ACCEL_BIAS)
    for moment in (5.0, 15.0, 30.0, 60.0):
        print("  bias estimated by the filter at %2.0f s: %.3f" % (moment, bias[int(round(moment / DT))]))

    os.makedirs("results", exist_ok=True)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    top.axhline(0, color="0.45", linewidth=1)
    top.plot(t, plain - true_height, color="#eb6834", linewidth=1.5, label="ignores the bias")
    top.plot(t, aware - true_height, color="#2a78d6", linewidth=1.5, label="estimates the bias")
    top.set_ylabel("height error (m)")
    top.set_ylim(-0.7, 0.9)
    top.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    top.set_title("Height error of two Kalman filters on the same data", fontsize=10, loc="left")
    bottom.axhline(ACCEL_BIAS, color="0.1", linewidth=1.2, linestyle=(0, (4, 3)), label="true bias")
    bottom.plot(t, bias, color="#2a78d6", linewidth=1.8, label="estimated bias")
    bottom.set_ylabel("accelerometer bias (m/s$^2$)")
    bottom.set_xlabel("time (s)")
    bottom.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
    for ax in (top, bottom):
        ax.grid(color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/s3_kalman_bias.png", dpi=200)
    print("figure written to results/s3_kalman_bias.png")


main()