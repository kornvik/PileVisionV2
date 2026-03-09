"""
BNO085 accuracy — using REPEATABILITY (not absolute accuracy).
We calibrate our own baseline, so absolute offset cancels.
What matters: sample-to-sample noise + short-term drift.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N_TRIALS = 2000

# BNO085 repeatability specs (what matters after self-calibration)
# Rotation vector noise: ~0.05° RMS per sample (short-term repeatability)
# Short-term drift: ~0.1-0.2° over minutes (temperature, recalibration events)
# Linear acceleration noise: ~0.05 m/s² after fusion gravity removal

PITCH_NOISE_DEG = 0.05       # per-sample noise (repeatability)
PITCH_DRIFT_DEG_PER_MIN = 0.1  # slow drift from temperature etc
LINEAR_ACCEL_NOISE = 0.05    # m/s² RMS


def sim_tilt_correction(true_tilt_deg, depth_m, time_since_reset_s=5.0):
    """Simulate BNO085 tilt correction with repeatability noise + drift."""
    true_tilt = np.radians(true_tilt_deg)
    true_error_mm = depth_m * np.tan(true_tilt) * 1000

    noise = np.radians(PITCH_NOISE_DEG)
    drift = np.radians(PITCH_DRIFT_DEG_PER_MIN) * (time_since_reset_s / 60.0)

    # Baseline: averaged over 200 samples → noise / sqrt(200)
    baseline_noise = noise / np.sqrt(200)

    # Current reading: true tilt + noise + drift
    measured = true_tilt + np.random.randn(N_TRIALS) * noise + \
               np.random.randn(N_TRIALS) * drift
    baseline = np.random.randn(N_TRIALS) * baseline_noise

    corrections_mm = depth_m * np.tan(measured - baseline) * 1000
    residuals = np.abs(corrections_mm - true_error_mm)

    return np.mean(residuals), np.std(residuals), np.percentile(residuals, 95)


# ── Analysis 1: vs tilt angle ──
tilts = np.linspace(0.1, 3.0, 30)
depth = 5.0

res_m, res_p95, uncorr = [], [], []
for t in tilts:
    m, _, p = sim_tilt_correction(t, depth)
    res_m.append(m)
    res_p95.append(p)
    uncorr.append(depth * np.tan(np.radians(t)) * 1000)

# ── Analysis 2: vs depth ──
depths = np.linspace(1, 10, 20)
d_m, d_p95, d_uncorr = [], [], []
for d in depths:
    m, _, p = sim_tilt_correction(0.5, d)
    d_m.append(m)
    d_p95.append(p)
    d_uncorr.append(d * np.tan(np.radians(0.5)) * 1000)

# ── Analysis 3: BNO085 vs raw gyro over time ──
durations = np.linspace(1, 120, 40)
bno_m, bno_p95 = [], []
old_m, old_p95 = [], []

GYRO_NOISE_DENSITY = np.radians(0.007)
GYRO_BIAS = np.radians(0.5) / 60

for dur in durations:
    # BNO085: bounded noise + small drift
    m, _, p = sim_tilt_correction(0.5, 5.0, time_since_reset_s=dur)
    bno_m.append(m)
    bno_p95.append(p)

    # Old raw gyro: unbounded drift
    n = int(dur * 400)
    dt = 1.0 / 400
    noise_std = GYRO_NOISE_DENSITY * np.sqrt(400)
    trials = []
    for _ in range(N_TRIALS):
        bias_err = GYRO_BIAS * 0.3 * np.random.randn()
        integrated = np.cumsum(np.random.randn(n) * noise_std + bias_err) * dt
        err_mm = 5.0 * np.tan(abs(integrated[-1])) * 1000 if n > 0 else 0
        trials.append(err_mm)
    old_m.append(np.mean(trials))
    old_p95.append(np.percentile(trials, 95))

# ── Analysis 4: Error budget with rest-period averaging ──
# During rest: 30 frames averaged, each with BNO085 noise
# Averaging reduces noise by sqrt(30)
scenarios = [
    ('0.5° / 3m', 0.5, 3.0),
    ('0.5° / 5m', 0.5, 5.0),
    ('1.0° / 5m', 1.0, 5.0),
    ('0.5° / 8m', 0.5, 8.0),
]

# ── PLOT ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("BNO085 Tilt Correction — Repeatability-Based Accuracy\n"
             f"(noise: ±{PITCH_NOISE_DEG}°/sample, drift: {PITCH_DRIFT_DEG_PER_MIN}°/min)",
             fontsize=13, fontweight='bold')

ax = axes[0, 0]
ax.plot(tilts, uncorr, 'r--', lw=2, label='Uncorrected')
ax.plot(tilts, res_m, 'g-', lw=2, label='BNO085 residual (mean)')
ax.plot(tilts, res_p95, 'g:', lw=1, label='BNO085 residual (95th)')
ax.set_xlabel('Camera tilt (degrees)')
ax.set_ylabel('Height error (mm)')
ax.set_title('vs tilt angle (depth=5m, 5s since reset)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(depths, d_uncorr, 'r--', lw=2, label='Uncorrected (0.5° tilt)')
ax.plot(depths, d_m, 'g-', lw=2, label='BNO085 residual (mean)')
ax.plot(depths, d_p95, 'g:', lw=1, label='BNO085 (95th)')
ax.axhline(2.0, color='gray', ls='--', alpha=0.5, label='Vision noise floor (~2mm)')
ax.set_xlabel('Depth to board (m)')
ax.set_ylabel('Height error (mm)')
ax.set_title('vs distance (0.5° tilt)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(durations, old_m, 'r-', lw=2, label='Old: raw gyro (mean)')
ax.plot(durations, old_p95, 'r:', lw=1, label='Old: raw gyro (95th)')
ax.plot(durations, bno_m, 'g-', lw=2, label='BNO085 (mean)')
ax.plot(durations, bno_p95, 'g:', lw=1, label='BNO085 (95th)')
ax.axhline(5.0, color='orange', ls='--', alpha=0.5, label='5mm target')
ax.set_xlabel('Time since reset (seconds)')
ax.set_ylabel('Residual error (mm) at 5m depth')
ax.set_title('Drift: BNO085 bounded vs raw gyro unbounded')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 60)

ax = axes[1, 1]
labels = [s[0] for s in scenarios]
uncorr_v = [d * np.tan(np.radians(t)) * 1000 for _, t, d in scenarios]
corr_m = [sim_tilt_correction(t, d, 5.0)[0] for _, t, d in scenarios]
corr_p = [sim_tilt_correction(t, d, 5.0)[2] for _, t, d in scenarios]

x = np.arange(len(labels))
w = 0.25
b1 = ax.bar(x - w, uncorr_v, w, label='Uncorrected', color='red', alpha=0.7)
b2 = ax.bar(x, corr_m, w, label='BNO085 (mean)', color='green', alpha=0.7)
b3 = ax.bar(x + w, corr_p, w, label='BNO085 (95th)', color='blue', alpha=0.7)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
ax.set_ylabel('Height error (mm)')
ax.set_title('Error budget per scenario')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('imu_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved imu_accuracy.png")

# ── Summary ──
print("\n" + "=" * 65)
print("BNO085 ACCURACY (repeatability-based, after baseline calibration)")
print("=" * 65)

print(f"\n{'Scenario':<16s} {'Uncorrected':>11s} {'Mean':>8s} {'95th':>8s} {'Reduction':>10s}")
print("-" * 58)
for label, t, d in scenarios:
    uc = d * np.tan(np.radians(t)) * 1000
    m, _, p = sim_tilt_correction(t, d, 5.0)
    print(f"{label:<16s} {uc:>9.1f}mm {m:>6.1f}mm {p:>6.1f}mm {(1-m/uc)*100:>8.0f}%")

print(f"\nWith rest-period averaging (30 frames):")
print(f"  Single-frame noise at 5m: ±{5.0*np.tan(np.radians(PITCH_NOISE_DEG))*1000:.1f}mm")
print(f"  After averaging 30 frames: ±{5.0*np.tan(np.radians(PITCH_NOISE_DEG/np.sqrt(30)))*1000:.1f}mm")
print(f"  Vision noise floor: ~2mm")
print(f"  Combined: ~3-5mm typical for set measurement")
