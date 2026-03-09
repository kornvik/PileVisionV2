"""
Does IMU help when the GROUND shifts under the camera during pile driving?

The camera is on a tripod. Each blow causes the surrounding ground to
settle a few mm. The camera's position changes permanently — this is NOT
vibration, it's a step change in the reference frame.

We test:
  1. No correction  — ground shift accumulates in height measurements
  2. Current IMU    — corrects within a blow cycle but resets at peak
  3. Fixed IMU      — only reset velocity, keep displacement (proposed fix)
"""

import sys
from unittest.mock import MagicMock
sys.modules["depthai"] = MagicMock()

import numpy as np
import matplotlib.pyplot as plt
from hammer_tracker import BlowDetector, IMUCompensator, IMPACT_VELOCITY_THRESHOLD

CAMERA_FPS = 60
IMU_RATE = 400
NUM_BLOWS = 10
LIFT_HEIGHT = 1.5
DROP_GRAVITY = 9.81
BOUNCE_AMP = 0.07
BOUNCE_FREQ = 12.0
BOUNCE_DECAY = 5.0
INITIAL_PILE_H = 1.0
SET_PER_BLOW_MM = 30.0
SET_DECAY = 0.85
IMU_NOISE_STD = 0.02

# Ground settlement: camera drops this much with each blow (cumulative)
SETTLE_PER_BLOW_MM = 3.0   # mm per blow — realistic near-pile settlement
SETTLE_DURATION_S = 0.3    # settlement happens over ~0.3s after impact


def generate_trajectory():
    """Generate hammer trajectory + ground settlement events."""
    times, true_heights = [], []
    t, dt = 0.0, 1.0 / CAMERA_FPS
    pile_top = INITIAL_PILE_H
    impact_times, expected_sets = [], []
    current_set = SET_PER_BLOW_MM

    # 3s warmup
    for _ in range(int(3.0 * CAMERA_FPS)):
        times.append(t)
        true_heights.append(pile_top)
        t += dt

    for _ in range(NUM_BLOWS):
        start_h, peak_h = pile_top, pile_top + LIFT_HEIGHT
        # Lift
        for i in range(int(1.2 * CAMERA_FPS)):
            frac = i / int(1.2 * CAMERA_FPS)
            times.append(t)
            true_heights.append(start_h + (peak_h - start_h) * (1 - (1 - frac)**2))
            t += dt
        # Peak
        for _ in range(int(0.2 * CAMERA_FPS)):
            times.append(t)
            true_heights.append(peak_h)
            t += dt
        # Drop
        drop_dur = np.sqrt(2 * (peak_h - pile_top) / DROP_GRAVITY)
        drop_frames = max(int(drop_dur * CAMERA_FPS), 2)
        for i in range(drop_frames):
            elapsed = ((i + 1) / drop_frames) * drop_dur
            times.append(t)
            true_heights.append(max(peak_h - 0.5 * DROP_GRAVITY * elapsed**2, pile_top))
            t += dt
        # Impact
        impact_times.append(t)
        expected_sets.append(current_set)
        pile_top -= current_set / 1000.0
        current_set *= SET_DECAY
        # Bounce
        for i in range(int(1.0 * CAMERA_FPS)):
            bt = (i + 1) * dt
            bounce = BOUNCE_AMP * np.exp(-BOUNCE_DECAY * bt) * np.sin(2 * np.pi * BOUNCE_FREQ * bt)
            times.append(t)
            true_heights.append(pile_top + abs(bounce))
            t += dt
        # Rest
        for _ in range(int(1.5 * CAMERA_FPS)):
            times.append(t)
            true_heights.append(pile_top)
            t += dt

    return np.array(times), np.array(true_heights), impact_times, expected_sets


def ground_shift_at(t, impact_times):
    """
    Cumulative ground settlement at time t.
    Each blow causes SETTLE_PER_BLOW_MM of settlement over SETTLE_DURATION_S.
    Returns displacement in meters (positive = camera dropped = heights read higher).
    """
    total = 0.0
    for ti in impact_times:
        dt_since = t - ti
        if dt_since <= 0:
            continue
        # Smooth step: sigmoid-ish settlement over SETTLE_DURATION_S
        frac = min(dt_since / SETTLE_DURATION_S, 1.0)
        # Smooth ease-in-out
        frac = frac * frac * (3 - 2 * frac)
        total += (SETTLE_PER_BLOW_MM / 1000.0) * frac
    return total


def ground_shift_accel_at(t, impact_times):
    """
    Analytical acceleration of ground settlement at time t.
    d²/dt² of the smoothstep function.
    """
    accel = 0.0
    T = SETTLE_DURATION_S
    A = SETTLE_PER_BLOW_MM / 1000.0
    for ti in impact_times:
        dt_since = t - ti
        if dt_since <= 0 or dt_since >= T:
            continue
        # smoothstep: f(x) = x²(3-2x), x = dt/T
        # f'(x) = 6x(1-x), f'(t) = f'(x)/T = 6*(dt/T)*(1-dt/T)/T
        # f''(x) = 6-12x, f''(t) = f''(x)/T² = (6-12*dt/T)/T²
        x = dt_since / T
        accel += A * (6 - 12 * x) / (T * T)
    return accel


def make_imu_packet(ax, ay, az, ts):
    pkt = MagicMock()
    d = MagicMock()
    d.acceleroMeter.x = ax
    d.acceleroMeter.y = ay
    d.acceleroMeter.z = az
    td = MagicMock()
    td.total_seconds.return_value = ts
    d.acceleroMeter.getTimestampDevice.return_value = td
    pkt.packets = [d]
    return pkt


class IMUCompensatorKeepDisplacement(IMUCompensator):
    """Modified: only reset velocity, keep accumulated displacement."""
    def reset_drift(self):
        self.velocity = np.zeros(3)
        # displacement NOT reset — preserves real camera movement


def run_mode(mode, times, true_h, impact_times):
    """Run one mode: 'none', 'current_imu', 'fixed_imu'."""
    np.random.seed(42)
    dt_cam = 1.0 / CAMERA_FPS
    n_imu = int(IMU_RATE / CAMERA_FPS)
    dt_imu = 1.0 / IMU_RATE

    detector = BlowDetector()
    if mode == "fixed_imu":
        compensator = IMUCompensatorKeepDisplacement()
    else:
        compensator = IMUCompensator()
    compensator.CALIB_COUNT = int(2.0 * IMU_RATE)

    heights_out = []
    blow_times_out = []
    detected_sets = []

    for i in range(len(times)):
        t = times[i]

        # Feed IMU
        for j in range(n_imu):
            imu_t = t - dt_cam + (j + 1) * dt_imu
            # IMU on the camera feels: gravity + ground settlement acceleration
            settle_acc = ground_shift_accel_at(imu_t, impact_times)
            # Settlement moves camera DOWN → IMU Y accel becomes more negative
            pkt = make_imu_packet(
                np.random.normal(0, IMU_NOISE_STD),
                -9.81 - settle_acc + np.random.normal(0, IMU_NOISE_STD),
                np.random.normal(0, IMU_NOISE_STD),
                imu_t
            )
            compensator.add_imu_packet(pkt)

        # Camera sees: true hammer height + ground shift effect
        # If camera drops by `shift`, the board appears `shift` higher
        shift = ground_shift_at(t, impact_times)
        camera_measured_y = -(true_h[i] + shift)  # tvec[1] in camera coords

        if mode != "none" and compensator.is_ready:
            correction = compensator.get_correction()
            height = -(camera_measured_y - correction[1])
        else:
            compensator.get_correction()
            height = -camera_measured_y

        heights_out.append(height)
        vel, blow, set_mm, _ = detector.update(t, height)

        if blow:
            blow_times_out.append(t)
            detected_sets.append(set_mm)

        if abs(vel) < 0.05:
            compensator.reset_drift()

    return np.array(heights_out), detector.blow_count, blow_times_out, detected_sets


def main():
    times, true_h, impact_times, expected_sets = generate_trajectory()

    # Ground shift array for plotting
    shifts = np.array([ground_shift_at(t, impact_times) for t in times])

    print("=" * 72)
    print("  Ground Settlement Simulation")
    print(f"  {SETTLE_PER_BLOW_MM}mm settlement per blow × {NUM_BLOWS} blows")
    print(f"  = {SETTLE_PER_BLOW_MM * NUM_BLOWS:.0f}mm total camera drift")
    print("=" * 72)

    modes = [
        ("none",        "No IMU correction"),
        ("current_imu", "Current IMU (reset vel+disp)"),
        ("fixed_imu",   "Fixed IMU (reset vel only)"),
    ]

    results = {}
    calib_end = int(2.5 * CAMERA_FPS)

    for mode_key, mode_label in modes:
        h_out, blows, blow_t, det_sets = run_mode(mode_key, times, true_h, impact_times)
        rms = np.sqrt(np.mean((h_out[calib_end:] - true_h[calib_end:])**2)) * 1000
        results[mode_key] = {
            "label": mode_label, "heights": h_out, "blows": blows,
            "rms_mm": rms, "blow_times": blow_t, "sets": det_sets
        }

    print()
    print(f"  {'Mode':<35} {'RMS err':>9} {'Blows':>6} {'Max drift':>11}")
    print(f"  {'-'*35} {'-'*9} {'-'*6} {'-'*11}")
    for mk, mv in results.items():
        max_drift = np.max(np.abs(mv["heights"][calib_end:] - true_h[calib_end:])) * 1000
        print(f"  {mv['label']:<35} {mv['rms_mm']:7.2f}mm {mv['blows']:>5}  {max_drift:8.2f}mm")

    # Set accuracy comparison
    print()
    print(f"  Set accuracy (mm):")
    print(f"  {'Blow':>4}  {'Expected':>10}  {'No IMU':>10}  {'Current':>10}  {'Fixed':>10}")
    print(f"  {'----':>4}  {'-'*10:>10}  {'-'*10:>10}  {'-'*10:>10}  {'-'*10:>10}")
    for i in range(min(len(expected_sets), 9)):
        exp = expected_sets[i]
        row = f"  {i+1:4d}  {exp:8.1f}mm"
        for mk in ["none", "current_imu", "fixed_imu"]:
            sets = results[mk]["sets"]
            if i < len(sets) and sets[i] is not None:
                row += f"  {sets[i]:8.1f}mm"
            else:
                row += f"  {'--':>10}"
        print(row)

    print()

    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Ground Settlement: {SETTLE_PER_BLOW_MM}mm/blow — Does IMU Help?",
                 fontsize=14, fontweight="bold")

    # Panel 1: Full height comparison
    ax1 = axes[0]
    ax1.plot(times, true_h, "k-", lw=1.5, label="True height", alpha=0.9)
    colors = {"none": "red", "current_imu": "blue", "fixed_imu": "green"}
    for mk in ["none", "current_imu", "fixed_imu"]:
        mv = results[mk]
        ax1.plot(times, mv["heights"], color=colors[mk], lw=0.8, alpha=0.7,
                 label=f"{mv['label']} (RMS {mv['rms_mm']:.1f}mm)")
    for it in impact_times:
        ax1.axvline(it, color="orange", alpha=0.2, ls=":", lw=0.8)
    ax1.set_ylabel("Height (m)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("Hammer Height — All Modes")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Zoomed error (measured - true)
    ax2 = axes[1]
    for mk in ["none", "current_imu", "fixed_imu"]:
        mv = results[mk]
        error = (mv["heights"] - true_h) * 1000  # mm
        ax2.plot(times, error, color=colors[mk], lw=0.8, alpha=0.7,
                 label=f"{mv['label']}")
    ax2.axhline(0, color="k", ls="-", lw=0.5)
    for it in impact_times:
        ax2.axvline(it, color="orange", alpha=0.2, ls=":", lw=0.8)
    ax2.set_ylabel("Error (mm)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title("Height Error (measured − true)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Ground settlement
    ax3 = axes[2]
    ax3.plot(times, shifts * 1000, "brown", lw=1.5, label="Ground settlement (camera drop)")
    ax3.set_ylabel("Settlement (mm)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(fontsize=9)
    ax3.set_title("Cumulative Ground Settlement Under Camera")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ground_shift_result.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved to ground_shift_result.png")
    plt.close()


if __name__ == "__main__":
    main()
