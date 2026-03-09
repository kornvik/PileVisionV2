"""
Does the IMU compensator actually help?
Test across shake severity levels with and without IMU correction.
"""

import sys
from unittest.mock import MagicMock
sys.modules["depthai"] = MagicMock()

import numpy as np
import matplotlib.pyplot as plt
from hammer_tracker import BlowDetector, IMUCompensator, IMPACT_VELOCITY_THRESHOLD

CAMERA_FPS = 60
IMU_RATE = 400
NUM_BLOWS = 8
LIFT_HEIGHT = 1.5
DROP_GRAVITY = 9.81
BOUNCE_AMP = 0.07
BOUNCE_FREQ = 12.0
BOUNCE_DECAY = 5.0
INITIAL_PILE_H = 1.0
SET_PER_BLOW_MM = 30.0
SET_DECAY = 0.85

IMU_NOISE_STD = 0.03  # typical MEMS noise


def generate_trajectory():
    times, heights = [], []
    t, dt = 0.0, 1.0 / CAMERA_FPS
    pile_top = INITIAL_PILE_H
    impact_times, expected_sets = [], []
    current_set = SET_PER_BLOW_MM

    # 3s warmup
    for _ in range(int(3.0 * CAMERA_FPS)):
        times.append(t); heights.append(pile_top); t += dt

    for _ in range(NUM_BLOWS):
        start_h, peak_h = pile_top, pile_top + LIFT_HEIGHT
        for i in range(int(1.2 * CAMERA_FPS)):
            frac = i / int(1.2 * CAMERA_FPS)
            times.append(t); heights.append(start_h + (peak_h - start_h) * (1 - (1 - frac)**2)); t += dt
        for _ in range(int(0.2 * CAMERA_FPS)):
            times.append(t); heights.append(peak_h); t += dt
        drop_dur = np.sqrt(2 * (peak_h - pile_top) / DROP_GRAVITY)
        drop_frames = max(int(drop_dur * CAMERA_FPS), 2)
        for i in range(drop_frames):
            elapsed = ((i + 1) / drop_frames) * drop_dur
            times.append(t); heights.append(max(peak_h - 0.5 * DROP_GRAVITY * elapsed**2, pile_top)); t += dt
        impact_times.append(t)
        expected_sets.append(current_set)
        pile_top -= current_set / 1000.0
        current_set *= SET_DECAY
        for i in range(int(1.0 * CAMERA_FPS)):
            bt = (i + 1) * dt
            bounce = BOUNCE_AMP * np.exp(-BOUNCE_DECAY * bt) * np.sin(2 * np.pi * BOUNCE_FREQ * bt)
            times.append(t); heights.append(pile_top + abs(bounce)); t += dt
        for _ in range(int(1.5 * CAMERA_FPS)):
            times.append(t); heights.append(pile_top); t += dt

    return np.array(times), np.array(heights), impact_times, expected_sets


def shake_at(t, impact_times, shake_g, shake_freq, impact_g, impact_freq, impact_decay, calib_end=2.5):
    if t < calib_end:
        return 0.0, 0.0
    G = 9.81
    w_base = 2 * np.pi * shake_freq
    amp_base = (shake_g * G) / w_base**2 if w_base > 0 else 0

    disp = amp_base * np.sin(w_base * t)
    accel = -amp_base * w_base**2 * np.sin(w_base * t)

    w_imp = 2 * np.pi * impact_freq
    amp_imp = (impact_g * G) / w_imp**2 if w_imp > 0 else 0
    d = impact_decay
    for ti in impact_times:
        dt = t - ti
        if dt <= 0:
            continue
        e = np.exp(-d * dt)
        s, c = np.sin(w_imp * dt), np.cos(w_imp * dt)
        disp += amp_imp * e * s
        accel += amp_imp * e * ((d**2 - w_imp**2) * s - 2 * d * w_imp * c)

    return disp, accel


def make_imu_packet(ax, ay, az, ts):
    pkt = MagicMock()
    d = MagicMock()
    d.acceleroMeter.x = ax
    d.acceleroMeter.y = ay
    d.acceleroMeter.z = az
    td = MagicMock(); td.total_seconds.return_value = ts
    d.acceleroMeter.getTimestampDevice.return_value = td
    pkt.packets = [d]
    return pkt


def run_scenario(label, shake_g, shake_freq, impact_g, impact_freq, impact_decay):
    """Run with and without IMU correction, return RMS errors and blow counts."""
    times, true_h, impact_times, expected_sets = generate_trajectory()
    dt_cam = 1.0 / CAMERA_FPS
    dt_imu = 1.0 / IMU_RATE
    n_imu = int(IMU_RATE / CAMERA_FPS)

    # Compute shake displacement for all frames
    shake = np.array([shake_at(t, impact_times, shake_g, shake_freq, impact_g, impact_freq, impact_decay)[0] for t in times])
    measured_h = true_h + shake

    results = {}

    for mode in ["no_imu", "with_imu"]:
        np.random.seed(42)  # reproducible noise
        detector = BlowDetector()
        compensator = IMUCompensator()
        compensator.CALIB_COUNT = int(2.0 * IMU_RATE)

        corrected = []
        blow_count = 0

        for i in range(len(times)):
            t = times[i]
            # Feed IMU
            for j in range(n_imu):
                imu_t = t - dt_cam + (j + 1) * dt_imu
                _, sa = shake_at(imu_t, impact_times, shake_g, shake_freq, impact_g, impact_freq, impact_decay)
                pkt = make_imu_packet(
                    np.random.normal(0, IMU_NOISE_STD),
                    -9.81 + sa + np.random.normal(0, IMU_NOISE_STD),
                    np.random.normal(0, IMU_NOISE_STD),
                    imu_t
                )
                compensator.add_imu_packet(pkt)

            raw_tvec_y = -(true_h[i] + shake[i])

            if mode == "with_imu" and compensator.is_ready:
                correction = compensator.get_correction()
                height = -(raw_tvec_y - correction[1])
            else:
                compensator.get_correction()  # advance calibration
                height = -raw_tvec_y

            corrected.append(height)
            vel, blow, _, _ = detector.update(t, height)
            if blow:
                blow_count += 1
            if abs(vel) < 0.05:
                compensator.reset_drift()

        corrected = np.array(corrected)
        calib_end = int(2.5 * CAMERA_FPS)
        rms = np.sqrt(np.mean((corrected[calib_end:] - true_h[calib_end:]) ** 2)) * 1000  # mm
        results[mode] = {"rms_mm": rms, "blows": blow_count, "corrected": corrected}

    return results, times, true_h, measured_h, shake, impact_times


# ============================================================
# SCENARIOS
# ============================================================

scenarios = [
    ("Mild (tripod, far from pile)",   0.02, 5.0,  0.1,  10.0, 3.0),
    ("Moderate (tripod, near pile)",    0.05, 8.0,  0.5,  15.0, 3.0),
    ("Severe (unstable mount)",         0.15, 8.0,  2.0,  20.0, 2.5),
    ("Extreme (on vibrating structure)", 0.3, 10.0, 5.0,  25.0, 2.0),
]


def main():
    print("=" * 72)
    print("  Does the IMU Compensator Actually Help?")
    print("  Testing across 4 shake severity levels")
    print("=" * 72)
    print()
    print(f"  {'Scenario':<38} {'Raw RMS':>9} {'IMU RMS':>9} {'Δ':>8}  {'Blows':>6} {'Blows':>6}")
    print(f"  {'':38} {'(mm)':>9} {'(mm)':>9} {'':>8}  {'raw':>6} {'IMU':>6}")
    print(f"  {'-'*38} {'-'*9} {'-'*9} {'-'*8}  {'-'*6} {'-'*6}")

    all_results = []
    for label, sg, sf, ig, iff, idc in scenarios:
        res, times, true_h, meas_h, shake, impacts = run_scenario(label, sg, sf, ig, iff, idc)
        raw_rms = res["no_imu"]["rms_mm"]
        imu_rms = res["with_imu"]["rms_mm"]
        diff = imu_rms - raw_rms
        sign = "+" if diff > 0 else ""
        verdict = "WORSE" if diff > 0 else "BETTER"
        print(f"  {label:<38} {raw_rms:8.2f}  {imu_rms:8.2f}  {sign}{diff:6.2f}  {res['no_imu']['blows']:>5}  {res['with_imu']['blows']:>5}   {verdict}")
        all_results.append((label, res, times, true_h, meas_h, shake, impacts))

    print()
    print("  VERDICT:")
    print("  " + "-" * 68)

    any_better = any(r[1]["with_imu"]["rms_mm"] < r[1]["no_imu"]["rms_mm"] for r in all_results)
    if any_better:
        print("  IMU helps in severe shake scenarios.")
    else:
        print("  IMU double-integration DRIFT exceeds shake amplitude in ALL cases.")
        print("  The compensator adds more noise than it removes.")
    print()

    # ============================================================
    # PLOT: 4 scenarios side by side
    # ============================================================

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 3.2 * len(scenarios)), sharex=False)
    fig.suptitle("IMU Compensator: Does It Help?", fontsize=14, fontweight="bold", y=1.01)

    for idx, (label, res, times, true_h, meas_h, shake, impacts) in enumerate(all_results):
        ax = axes[idx]
        raw_rms = res["no_imu"]["rms_mm"]
        imu_rms = res["with_imu"]["rms_mm"]

        # Zoom into a 2-blow window for clarity
        t_start = impacts[2] - 2.0 if len(impacts) > 2 else 3.0
        t_end = t_start + 8.0
        mask = (times >= t_start) & (times <= t_end)

        ax.plot(times[mask], true_h[mask] * 1000, "k-", lw=1.5, label="True", alpha=0.9)
        ax.plot(times[mask], meas_h[mask] * 1000, "r-", lw=0.6, alpha=0.5, label=f"Raw (RMS {raw_rms:.2f}mm)")
        ax.plot(times[mask], res["with_imu"]["corrected"][mask] * 1000, "b-", lw=0.8, alpha=0.7,
                label=f"IMU corrected (RMS {imu_rms:.2f}mm)")

        for it in impacts:
            if t_start <= it <= t_end:
                ax.axvline(it, color="green", alpha=0.3, ls="--", lw=0.8)

        ax.set_ylabel("Height (mm)")
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("imu_comparison.png", dpi=150, bbox_inches="tight")
    print("  Plot saved to imu_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
