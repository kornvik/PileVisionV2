"""
Simulate old IMUCompensator (double-integration) vs new IMUHelper (tilt + gating)
on synthetic MEMS IMU data with realistic noise, then plot the comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Simulation parameters ──
DURATION = 30.0        # seconds (6 blow cycles)
IMU_HZ = 400
CAM_HZ = 60
DEPTH = 5.0            # meters to board

dt_imu = 1.0 / IMU_HZ
dt_cam = 1.0 / CAM_HZ
n_imu = int(DURATION * IMU_HZ)
n_cam = int(DURATION * CAM_HZ)

t_imu = np.arange(n_imu) * dt_imu
t_cam = np.arange(n_cam) * dt_cam

# ── Ground truth: camera tilt (tripod sinking) ──
# Ramp from 0° to 0.5° over 30s (realistic tripod drift on soft ground)
true_tilt_rad = np.linspace(0, np.radians(0.5), n_imu)
true_tilt_at_cam = np.interp(t_cam, t_imu, true_tilt_rad)

# True height error from tilt: depth * tan(tilt)
true_height_error = DEPTH * np.tan(true_tilt_at_cam)

# ── MEMS noise model ──
# Accelerometer: 0.003 m/s² noise density (typical BMI270)
accel_noise_std = 0.003 * np.sqrt(IMU_HZ)  # ~0.06 m/s² per sample
accel_bias = 0.02  # m/s² constant bias (typical after factory cal)

# Gyroscope: 0.007 °/s noise density → rad/s
gyro_noise_std = np.radians(0.007) * np.sqrt(IMU_HZ)  # ~0.0024 rad/s per sample
gyro_bias = np.radians(0.5) / 60  # 0.5°/min drift (typical)

# True angular velocity from tilt ramp
true_omega = np.gradient(true_tilt_rad, dt_imu)

# Measured IMU signals
meas_accel_y = -9.81 + accel_bias + np.random.randn(n_imu) * accel_noise_std
meas_gyro_x = true_omega + gyro_bias + np.random.randn(n_imu) * gyro_noise_std

# Add vibration bursts during impacts (6 blows at t=3,8,13,18,23,28)
blow_times = [3, 8, 13, 18, 23, 28]
for bt in blow_times:
    mask = (t_imu > bt) & (t_imu < bt + 0.5)
    meas_accel_y[mask] += np.random.randn(mask.sum()) * 2.0  # 2 m/s² vibration
    meas_gyro_x[mask] += np.random.randn(mask.sum()) * 0.3   # vibration

# ── Method 1: Old double-integration (IMUCompensator) ──
# Calibrate on first 200 samples
calib_end = 200
gravity_est = np.mean(meas_accel_y[:calib_end])

velocity = 0.0
displacement = 0.0
disp_old = np.zeros(n_imu)

for i in range(calib_end, n_imu):
    linear_accel = meas_accel_y[i] - gravity_est
    velocity += linear_accel * dt_imu
    displacement += velocity * dt_imu
    disp_old[i] = displacement

# Sample at camera times
disp_old_cam = np.interp(t_cam, t_imu, disp_old)

# ── Method 2: New tilt correction (IMUHelper) ──
gyro_bias_est = np.mean(meas_gyro_x[:calib_end])

tilt_angle = 0.0
tilt_new = np.zeros(n_imu)

for i in range(calib_end, n_imu):
    corrected_gyro = meas_gyro_x[i] - gyro_bias_est
    tilt_angle += corrected_gyro * dt_imu
    tilt_new[i] = tilt_angle

tilt_correction_cam = DEPTH * np.tan(np.interp(t_cam, t_imu, tilt_new))

# ── Method 2b: Frame gating — mark non-still frames ──
accel_mag = np.abs(meas_accel_y)
gravity_mag = np.abs(gravity_est)
accel_dev = np.abs(accel_mag - gravity_mag)
gyro_mag = np.abs(meas_gyro_x - gyro_bias_est)

still_mask_imu = (accel_dev < 0.5) & (gyro_mag < 0.05)
# Sample at camera rate (nearest neighbor)
still_mask_cam = np.array([still_mask_imu[int(t * IMU_HZ)] for t in t_cam])

# ── Simulated vision height with noise ──
base_height = 1.0  # meters
vision_noise = np.random.randn(n_cam) * 0.002  # 2mm vision noise
# Add tilt-induced error (what the camera sees without correction)
raw_height = base_height + true_height_error + vision_noise

# Add vibration-induced noise at blow times
for bt in blow_times:
    mask = (t_cam > bt) & (t_cam < bt + 0.5)
    raw_height[mask] += np.random.randn(mask.sum()) * 0.01  # 10mm vibration noise

# ── Corrected heights ──
# Old method: subtract displacement (this is wrong — displacement is noise)
height_old = raw_height - disp_old_cam

# New method: subtract tilt correction (this targets the actual error)
height_new = raw_height - tilt_correction_cam

# True height (no tilt error, no noise)
height_true = np.full(n_cam, base_height)

# ── Rest-period windows (between blows) for set measurement simulation ──
rest_windows = []
for i, bt in enumerate(blow_times):
    rest_start = bt + 1.0  # 1s after impact
    rest_end = rest_start + 2.0  # 2s window
    if rest_end < DURATION:
        rest_windows.append((rest_start, rest_end))

# Compute rest-period stats
def rest_stats(t, height, mask=None):
    stds = []
    for rs, re in rest_windows:
        idx = (t >= rs) & (t <= re)
        if mask is not None:
            idx = idx & mask
        if idx.sum() > 5:
            stds.append(np.std(height[idx]) * 1000)  # mm
    return stds

std_raw = rest_stats(t_cam, raw_height)
std_old = rest_stats(t_cam, height_old)
std_new = rest_stats(t_cam, height_new)
std_new_gated = rest_stats(t_cam, height_new, still_mask_cam)

# ── PLOT ──
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Old IMU Compensator vs New Tilt Correction + Frame Gating", fontsize=14, fontweight='bold')

# Plot 1: Height error comparison
ax = axes[0]
ax.plot(t_cam, (raw_height - base_height) * 1000, alpha=0.4, color='gray', label='No correction')
ax.plot(t_cam, (height_old - base_height) * 1000, alpha=0.6, color='red', label='Old: double-integration')
ax.plot(t_cam, (height_new - base_height) * 1000, alpha=0.8, color='green', label='New: tilt correction')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
for bt in blow_times:
    ax.axvline(bt, color='orange', alpha=0.3, linewidth=8)
ax.set_ylabel('Height error (mm)')
ax.set_title('Height measurement error (0 = perfect)')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(-100, 100)

# Plot 2: Double-integration drift vs tilt tracking
ax = axes[1]
ax.plot(t_cam, disp_old_cam * 1000, color='red', label='Old: accumulated displacement (drift)')
ax.plot(t_cam, tilt_correction_cam * 1000, color='green', label='New: tilt correction')
ax.plot(t_cam, true_height_error * 1000, color='blue', linestyle='--', label='True tilt error')
ax.set_ylabel('Correction (mm)')
ax.set_title('Correction applied: drift explosion vs targeted tilt tracking')
ax.legend(loc='upper left', fontsize=9)

# Plot 3: Frame gating
ax = axes[2]
not_still = ~still_mask_cam
ax.plot(t_cam, raw_height * 1000, '.', markersize=1, color='gray', alpha=0.3, label='All frames')
ax.plot(t_cam[still_mask_cam], raw_height[still_mask_cam] * 1000, '.', markersize=1.5, color='green', label='Still frames (gated)')
ax.plot(t_cam[not_still], raw_height[not_still] * 1000, '.', markersize=2, color='red', alpha=0.5, label='Rejected frames')
for rs, re in rest_windows:
    ax.axvspan(rs, re, alpha=0.1, color='blue')
ax.set_ylabel('Height (mm)')
ax.set_title('Frame gating: red dots rejected during rest averaging')
ax.legend(loc='upper left', fontsize=9, markerscale=5)

# Plot 4: Rest-period standard deviation comparison
ax = axes[3]
x = np.arange(len(std_raw))
w = 0.2
ax.bar(x - 1.5*w, std_raw, w, label='No correction', color='gray')
ax.bar(x - 0.5*w, std_old, w, label='Old: double-integ', color='red')
ax.bar(x + 0.5*w, std_new, w, label='New: tilt corr', color='green')
ax.bar(x + 1.5*w, std_new_gated, w, label='New: tilt + gating', color='blue')
ax.set_xlabel('Rest period (after each blow)')
ax.set_ylabel('Std dev (mm)')
ax.set_title('Rest-period height noise (lower = better set measurement)')
ax.legend(fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f'Blow {i+1}' for i in x])

plt.tight_layout()
plt.savefig('imu_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved imu_comparison.png")

# Print summary
print(f"\nRest-period std dev (mm):")
print(f"  No correction:     {np.mean(std_raw):.2f} mm avg")
print(f"  Old double-integ:  {np.mean(std_old):.2f} mm avg")
print(f"  New tilt corr:     {np.mean(std_new):.2f} mm avg")
print(f"  New tilt + gating: {np.mean(std_new_gated):.2f} mm avg")
