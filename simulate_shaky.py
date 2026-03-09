"""
Pile Driving Hammer Tracker — Shaky Environment Simulation

Generates synthetic hammer motion with camera shake, runs BlowDetector +
IMUCompensator, and plots how well the system performs under vibration.

Usage:
    python simulate_shaky.py
"""

import sys
from unittest.mock import MagicMock

# Mock depthai so hammer_tracker can be imported without OAK-D hardware SDK
sys.modules["depthai"] = MagicMock()

import numpy as np
import matplotlib.pyplot as plt
from hammer_tracker import BlowDetector, IMUCompensator, IMPACT_VELOCITY_THRESHOLD

# ============================================================
# SIMULATION PARAMETERS
# ============================================================

NUM_BLOWS       = 10
CAMERA_FPS      = 60
IMU_RATE        = 400       # Hz
LIFT_HEIGHT     = 1.5       # meters above pile top
LIFT_DURATION   = 1.2       # seconds to lift
DROP_GRAVITY    = 9.81      # m/s²
BOUNCE_AMP      = 0.07      # post-impact bounce amplitude (m)
BOUNCE_FREQ     = 12.0      # Hz
BOUNCE_DECAY    = 5.0       # exponential decay rate
INITIAL_PILE_H  = 1.0       # starting pile-top height in camera frame
SET_PER_BLOW_MM = 30.0      # initial penetration per blow (mm)
SET_DECAY       = 0.85      # each blow penetrates less (resistance builds)

# Camera shake parameters — defined by acceleration (realistic G-levels)
# Real pile driving ground vibration: 0.1-2G peak
BASELINE_SHAKE_G    = 0.05    # 0.05G continuous background vibration
BASELINE_SHAKE_FREQ = 8.0     # Hz (low-frequency tripod sway)
IMPACT_SHAKE_G      = 0.5     # 0.5G peak shake after impact
IMPACT_SHAKE_FREQ   = 15.0    # Hz (ground vibration frequency)
IMPACT_SHAKE_DECAY  = 3.0     # decay rate (s⁻¹)

# Derived displacement amplitudes: A_disp = A_accel / ω²
_G = 9.81
BASELINE_SHAKE_AMP  = (BASELINE_SHAKE_G * _G) / (2 * np.pi * BASELINE_SHAKE_FREQ) ** 2
IMPACT_SHAKE_AMP    = (IMPACT_SHAKE_G * _G) / (2 * np.pi * IMPACT_SHAKE_FREQ) ** 2

IMU_NOISE_STD       = 0.03    # m/s² sensor noise (typical MEMS)


# ============================================================
# GENERATE SYNTHETIC HAMMER MOTION
# ============================================================

def generate_hammer_trajectory():
    """Generate ground-truth hammer height over time for NUM_BLOWS cycles."""
    times = []
    heights = []
    impact_times = []
    expected_sets = []

    t = 0.0
    dt = 1.0 / CAMERA_FPS
    pile_top = INITIAL_PILE_H
    current_set = SET_PER_BLOW_MM

    # Warmup: 3 seconds at rest (for IMU calibration + lockout)
    warmup_frames = int(3.0 * CAMERA_FPS)
    for _ in range(warmup_frames):
        times.append(t)
        heights.append(pile_top)
        t += dt

    for blow_i in range(NUM_BLOWS):
        # --- LIFT PHASE ---
        start_h = pile_top
        peak_h = pile_top + LIFT_HEIGHT
        lift_frames = int(LIFT_DURATION * CAMERA_FPS)
        for i in range(lift_frames):
            frac = i / lift_frames
            # Smooth ease-out lift
            h = start_h + (peak_h - start_h) * (1 - (1 - frac) ** 2)
            times.append(t)
            heights.append(h)
            t += dt

        # --- BRIEF PEAK ---
        for _ in range(int(0.2 * CAMERA_FPS)):
            times.append(t)
            heights.append(peak_h)
            t += dt

        # --- DROP PHASE (free fall under gravity) ---
        drop_h = peak_h - pile_top  # distance to fall
        drop_duration = np.sqrt(2 * drop_h / DROP_GRAVITY)
        drop_frames = max(int(drop_duration * CAMERA_FPS), 2)
        for i in range(drop_frames):
            frac = (i + 1) / drop_frames
            # h = peak - 0.5*g*t²  mapped to fraction
            elapsed = frac * drop_duration
            h = peak_h - 0.5 * DROP_GRAVITY * elapsed ** 2
            h = max(h, pile_top)  # clamp at pile top
            times.append(t)
            heights.append(h)
            t += dt

        # --- IMPACT ---
        impact_t = t
        impact_times.append(impact_t)
        penetration_m = current_set / 1000.0
        expected_sets.append(current_set)
        pile_top -= penetration_m
        current_set *= SET_DECAY

        # --- POST-IMPACT BOUNCE (damped oscillation) ---
        bounce_frames = int(1.0 * CAMERA_FPS)
        for i in range(bounce_frames):
            bt = (i + 1) * dt
            bounce = BOUNCE_AMP * np.exp(-BOUNCE_DECAY * bt) * np.sin(2 * np.pi * BOUNCE_FREQ * bt)
            times.append(t)
            heights.append(pile_top + abs(bounce))
            t += dt

        # --- REST before next blow ---
        rest_frames = int(1.5 * CAMERA_FPS)
        for _ in range(rest_frames):
            times.append(t)
            heights.append(pile_top)
            t += dt

    return np.array(times), np.array(heights), impact_times, expected_sets


# ============================================================
# GENERATE CAMERA SHAKE (analytical — displacement + acceleration)
# ============================================================

def shake_at(t, impact_times, calib_end=2.5):
    """
    Compute shake displacement and acceleration analytically at time t.
    Returns (displacement_m, acceleration_m_s2).
    No shake during calibration period.
    """
    if t < calib_end:
        return 0.0, 0.0

    disp = 0.0
    accel = 0.0
    w_base = 2 * np.pi * BASELINE_SHAKE_FREQ

    # Baseline continuous vibration: A*sin(wt)
    disp += BASELINE_SHAKE_AMP * np.sin(w_base * t)
    accel += -BASELINE_SHAKE_AMP * w_base ** 2 * np.sin(w_base * t)

    # Impact bursts: A*exp(-decay*(t-ti))*sin(w*(t-ti))
    w_impact = 2 * np.pi * IMPACT_SHAKE_FREQ
    d = IMPACT_SHAKE_DECAY
    for ti in impact_times:
        dt = t - ti
        if dt <= 0:
            continue
        e = np.exp(-d * dt)
        s = np.sin(w_impact * dt)
        c = np.cos(w_impact * dt)

        disp += IMPACT_SHAKE_AMP * e * s

        # Second derivative of A*exp(-d*t)*sin(w*t):
        # A*exp(-d*t) * [(d²-w²)*sin(wt) - 2dw*cos(wt)]
        accel += IMPACT_SHAKE_AMP * e * ((d**2 - w_impact**2) * s - 2 * d * w_impact * c)

    return disp, accel


def generate_shake_arrays(times, impact_times):
    """Generate shake displacement array for all camera frames."""
    shake = np.zeros_like(times)
    for i, t in enumerate(times):
        shake[i], _ = shake_at(t, impact_times)
    return shake


# ============================================================
# MOCK IMU PACKET (matches depthai interface)
# ============================================================

def make_imu_packet(accel_x, accel_y, accel_z, timestamp_s):
    """Create a mock IMU packet compatible with IMUCompensator.add_imu_packet()."""
    packet = MagicMock()
    imu_data = MagicMock()
    imu_data.acceleroMeter.x = accel_x
    imu_data.acceleroMeter.y = accel_y
    imu_data.acceleroMeter.z = accel_z
    td = MagicMock()
    td.total_seconds.return_value = timestamp_s
    imu_data.acceleroMeter.getTimestampDevice.return_value = td
    packet.packets = [imu_data]
    return packet


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation():
    print("=" * 65)
    print("  Pile Driving Hammer Tracker — Shaky Environment Simulation")
    print("=" * 65)

    # Generate ground truth
    times, true_heights, impact_times, expected_sets = generate_hammer_trajectory()
    shake = generate_shake_arrays(times, impact_times)
    measured_heights = true_heights + shake  # what camera "sees"

    dt_cam = 1.0 / CAMERA_FPS
    dt_imu = 1.0 / IMU_RATE
    n_imu_per_frame = int(IMU_RATE / CAMERA_FPS)  # ~6-7 IMU samples per frame

    # Initialize detector + compensator
    detector = BlowDetector()
    compensator = IMUCompensator()
    compensator.CALIB_COUNT = int(2.0 * IMU_RATE)  # 2s calibration (clean, no shake)

    # Storage for results
    corrected_heights = []
    velocities = []
    blow_frames = []
    detected_sets = []

    print(f"\nSimulating {NUM_BLOWS} hammer blows at {CAMERA_FPS}fps with camera shake...")
    print(f"  Baseline shake: {BASELINE_SHAKE_G}G ({BASELINE_SHAKE_AMP*1000:.2f}mm) @ {BASELINE_SHAKE_FREQ}Hz")
    print(f"  Impact shake:   {IMPACT_SHAKE_G}G ({IMPACT_SHAKE_AMP*1000:.2f}mm peak) @ {IMPACT_SHAKE_FREQ}Hz")
    print()

    for i in range(len(times)):
        t = times[i]

        # Feed IMU data (n_imu_per_frame samples between frames)
        for j in range(n_imu_per_frame):
            imu_t = t - dt_cam + (j + 1) * dt_imu
            # Compute analytical shake acceleration at this IMU timestamp
            _, shake_acc = shake_at(imu_t, impact_times)
            # IMU measures: gravity (Y=-9.81) + shake acceleration + sensor noise
            noise = np.random.normal(0, IMU_NOISE_STD)
            accel_y = -9.81 + shake_acc + noise
            pkt = make_imu_packet(
                accel_x=np.random.normal(0, IMU_NOISE_STD),
                accel_y=accel_y,
                accel_z=np.random.normal(0, IMU_NOISE_STD),
                timestamp_s=imu_t
            )
            compensator.add_imu_packet(pkt)

        # Simulate what main loop does:
        # tvec[1] from camera = -(true_height + shake) in camera coords
        raw_tvec_y = -(true_heights[i] + shake[i])

        if compensator.is_ready:
            correction = compensator.get_correction()
            # correction[1] is the Y displacement the IMU measured
            corrected_y = raw_tvec_y - correction[1]
            height = -corrected_y
        else:
            compensator.get_correction()  # process pending to advance calibration
            height = -raw_tvec_y
            corrected_heights.append(height)
            velocities.append(0.0)
            continue

        corrected_heights.append(height)

        vel, blow, set_mm, drop = detector.update(t, height)
        velocities.append(vel)

        if blow:
            blow_frames.append(i)
            detected_sets.append(set_mm)

        # Reset drift when hammer is nearly stationary (same as main loop)
        if abs(vel) < 0.05:
            compensator.reset_drift()

    corrected_heights = np.array(corrected_heights)
    velocities = np.array(velocities)

    # ============================================================
    # RESULTS SUMMARY
    # ============================================================

    print("-" * 65)
    print(f"  {'RESULTS':^61}")
    print("-" * 65)
    print(f"  Expected blows:  {NUM_BLOWS}")
    print(f"  Detected blows:  {detector.blow_count}")
    print(f"  False positives: {max(0, detector.blow_count - NUM_BLOWS)}")
    print(f"  Missed blows:    {max(0, NUM_BLOWS - detector.blow_count)}")
    print()

    # Set accuracy
    if len(detected_sets) > 1:
        print(f"  {'Blow':>4}  {'Expected set':>14}  {'Detected set':>14}  {'Error':>8}")
        print(f"  {'----':>4}  {'-'*14:>14}  {'-'*14:>14}  {'-'*8:>8}")
        for j, (exp, det) in enumerate(zip(expected_sets[: len(detected_sets)], detected_sets)):
            if det is not None:
                err = abs(det - exp)
                print(f"  {j+1:4d}  {exp:12.1f}mm  {det:12.1f}mm  {err:6.1f}mm")
            else:
                print(f"  {j+1:4d}  {exp:12.1f}mm  {'--':>14}  {'--':>8}")

    # RMS error: corrected vs true
    # Only compare after calibration period
    calib_frames = int(2.5 * CAMERA_FPS)
    if len(corrected_heights) > calib_frames:
        rms_raw = np.sqrt(np.mean((measured_heights[calib_frames:] - true_heights[calib_frames:]) ** 2))
        rms_corrected = np.sqrt(np.mean((corrected_heights[calib_frames:] - true_heights[calib_frames:len(corrected_heights)]) ** 2))
        print()
        print(f"  Height RMS error (raw shaky):     {rms_raw*1000:.2f} mm")
        print(f"  Height RMS error (IMU corrected):  {rms_corrected*1000:.2f} mm")
        improvement = (1 - rms_corrected / rms_raw) * 100 if rms_raw > 0 else 0
        print(f"  Improvement:                       {improvement:.1f}%")

    print("-" * 65)

    # ============================================================
    # PLOT
    # ============================================================

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Hammer Tracker Simulation — Shaky Environment", fontsize=14, fontweight="bold")

    # --- Panel 1: Heights ---
    ax1 = axes[0]
    ax1.plot(times, true_heights, "k-", linewidth=1.5, label="True height", alpha=0.9)
    ax1.plot(times, measured_heights, "r-", linewidth=0.5, alpha=0.5, label="Raw (with shake)")
    if len(corrected_heights) == len(times):
        ax1.plot(times, corrected_heights, "b-", linewidth=0.8, alpha=0.8, label="IMU corrected")
    # Mark blow detections
    for bf in blow_frames:
        ax1.axvline(times[bf], color="green", alpha=0.4, linewidth=1, linestyle="--")
    # Mark expected impacts
    for it in impact_times:
        ax1.axvline(it, color="orange", alpha=0.3, linewidth=1, linestyle=":")
    ax1.set_ylabel("Height (m)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("Hammer Height — True vs Shaky vs IMU-corrected")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Velocity ---
    ax2 = axes[1]
    ax2.plot(times[:len(velocities)], velocities, "b-", linewidth=0.6, alpha=0.8)
    ax2.axhline(-IMPACT_VELOCITY_THRESHOLD, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({-IMPACT_VELOCITY_THRESHOLD} m/s)")
    for bf in blow_frames:
        ax2.axvline(times[bf], color="green", alpha=0.4, linewidth=1, linestyle="--")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_title("Velocity (positive=up)")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Shake + Set ---
    ax3 = axes[2]
    ax3.plot(times, shake * 1000, "r-", linewidth=0.5, alpha=0.6, label="Camera shake (mm)")
    # Plot detected sets as bar markers
    if detected_sets:
        blow_times_det = [times[bf] for bf in blow_frames]
        sets_plot = [s if s is not None else 0 for s in detected_sets]
        ax3b = ax3.twinx()
        ax3b.bar(blow_times_det, sets_plot, width=0.3, alpha=0.5, color="green", label="Set per blow (mm)")
        ax3b.set_ylabel("Set (mm)", color="green")
        ax3b.tick_params(axis="y", labelcolor="green")
        ax3b.legend(loc="upper right", fontsize=9)
    ax3.set_ylabel("Shake (mm)", color="red")
    ax3.tick_params(axis="y", labelcolor="red")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_title("Camera Shake & Penetration per Blow")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simulation_result.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to simulation_result.png")

    # Show interactively if not headless
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except Exception:
        pass


if __name__ == "__main__":
    run_simulation()
