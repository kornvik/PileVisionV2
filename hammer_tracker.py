"""
Pile Driving Hammer Tracker
OAK-D PoE + ChArUco — Drop Hammer Vertical Tracking

Setup:
  1. Connect OAK-D PoE via ethernet (same subnet as your laptop)
  2. Attach printed ChArUco board to side of drop hammer
  3. Mount camera perpendicular to drop axis, level, at mid-drop height
  4. Run this script

Output:
  - Live video with pose overlay
  - CSV log: timestamp, hammer_height_m, velocity_m_s, event
  - Console blow count + set per blow
"""

import cv2
import numpy as np
import time
import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

# ============================================================
# CONFIGURATION — update SQUARE_LENGTH after measuring print
# ============================================================

SQUARES_X       = 5
SQUARES_Y       = 6
SQUARE_LENGTH   = 0.083     # MEASURE YOUR PRINT and update this
MARKER_LENGTH   = 0.058
ARUCO_DICT      = cv2.aruco.DICT_4X4_50

CAMERA_FPS      = 60
CAMERA_WIDTH    = 1920
CAMERA_HEIGHT   = 1080
EXPOSURE_US     = 800       # microseconds — reduce if blur, increase if dark
ISO             = 400

# OAK-D PoE IP — set to None to auto-discover USB, or set IP string
# e.g. DEVICE_IP = "169.254.1.222"
DEVICE_IP       = None

# Blow detection — tune to your hammer
# A "blow" is detected when hammer velocity crosses this threshold downward
IMPACT_VELOCITY_THRESHOLD = 0.5   # m/s downward
MIN_BLOW_INTERVAL_S       = 1.0   # ignore events closer than this

# Output CSV
OUTPUT_CSV = "hammer_log.csv"

# ============================================================
# CHARUCO SETUP
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

detector_params = cv2.aruco.DetectorParameters()
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 23
detector_params.adaptiveThreshWinSizeStep = 4
detector_params.minMarkerPerimeterRate    = 0.02
detector_params.cornerRefinementMethod   = cv2.aruco.CORNER_REFINE_SUBPIX
detector_params.cornerRefinementWinSize  = 5

charuco_params   = cv2.aruco.CharucoParameters()
charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

# ============================================================
# DEPTHAI PIPELINE
# ============================================================

def build_pipeline():
    import depthai as dai
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(CAMERA_FPS)
    cam.setInterleaved(False)

    # Manual exposure to freeze motion blur
    # cam.initialControl.setManualExposure(EXPOSURE_US, ISO)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)

    # IMU — BNO085 fused outputs (hardware sensor fusion, not raw MEMS)
    # ROTATION_VECTOR: absolute orientation quaternion (accel+gyro+mag fusion)
    # LINEAR_ACCELERATION: gravity-removed acceleration (fusion-based)
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 400)
    imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, 400)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    imu_xout = pipeline.create(dai.node.XLinkOut)
    imu_xout.setStreamName("imu")
    imu.out.link(imu_xout.input)

    return pipeline


# ============================================================
# REALSENSE PIPELINE (D435i)
# ============================================================

def build_realsense_pipeline():
    """Start RealSense D435i: color 1920x1080@30fps + accel + gyro."""
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
    cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    profile = pipe.start(cfg)
    return pipe, profile


def get_realsense_intrinsics(profile):
    """Extract camera_matrix and dist_coeffs from RealSense calibration."""
    import pyrealsense2 as rs
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream.get_intrinsics()
    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(intr.coeffs)
    return camera_matrix, dist_coeffs


# ============================================================
# POSE ESTIMATION
# ============================================================

def estimate_pose(frame, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = charuco_detector.detectBoard(gray)
    corners, ids = result[0], result[1]

    if ids is None or len(ids) < 4:
        return None, None, 0, frame

    obj_points, img_points = board.matchImagePoints(corners, ids)

    if len(obj_points) < 4:
        return None, None, 0, frame

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_points, img_points,
        camera_matrix, dist_coeffs,
        reprojectionError=2.5,
        confidence=0.99,
        iterationsCount=200
    )

    if not ret or inliers is None:
        return None, None, 0, frame

    inlier_ratio = len(inliers) / len(ids)

    # Draw overlay
    cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # Quality indicator
    color = (0, 255, 0) if inlier_ratio > 0.7 else (0, 165, 255)
    cv2.putText(frame, f"Inliers: {len(inliers)}/{len(ids)} ({inlier_ratio:.0%})",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return rvec, tvec, inlier_ratio, frame


# ============================================================
# MADGWICK AHRS FILTER (for RealSense raw IMU)
# ============================================================

class MadgwickFilter:
    """Fuses raw accel + gyro into orientation quaternion (Madgwick AHRS)."""

    def __init__(self, beta=0.1):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.beta = beta

    def update(self, gyro, accel, dt):
        if dt <= 0:
            return
        q0, q1, q2, q3 = self.q
        gx, gy, gz = gyro
        a_norm = np.linalg.norm(accel)
        if a_norm < 1e-6:
            return
        ax, ay, az = accel / a_norm
        # Gradient descent corrective step
        f = np.array([
            2 * (q1 * q3 - q0 * q2) - ax,
            2 * (q0 * q1 + q2 * q3) - ay,
            2 * (0.5 - q1 * q1 - q2 * q2) - az
        ])
        J = np.array([
            [-2 * q2,  2 * q3, -2 * q0,  2 * q1],
            [ 2 * q1,  2 * q0,  2 * q3,  2 * q2],
            [      0, -4 * q1, -4 * q2,       0]
        ])
        step = J.T @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 1e-6:
            step /= step_norm
        # Quaternion rate from gyroscope
        q_dot = 0.5 * np.array([
            -q1 * gx - q2 * gy - q3 * gz,
             q0 * gx + q2 * gz - q3 * gy,
             q0 * gy - q1 * gz + q3 * gx,
             q0 * gz + q1 * gy - q2 * gx
        ])
        q_dot -= self.beta * step
        self.q += q_dot * dt
        self.q /= np.linalg.norm(self.q)

    def get_pitch(self):
        """Extract pitch from quaternion [w, x, y, z]."""
        w, x, y, z = self.q
        sinp = 2.0 * (w * x + y * z)
        return float(np.arcsin(np.clip(sinp, -1.0, 1.0)))

    def get_gravity_vector(self):
        """Gravity vector in sensor frame (m/s^2)."""
        w, x, y, z = self.q
        return np.array([
            2 * (x * z - w * y),
            2 * (w * x + y * z),
            w * w - x * x - y * y + z * z
        ]) * 9.81

    def remove_gravity(self, accel):
        """Return linear acceleration with gravity subtracted."""
        return accel - self.get_gravity_vector()


# ============================================================
# IMU COMPENSATOR
# ============================================================

class IMUHelper:
    """
    Uses BNO085 fused outputs for pile driving corrections:

    1. TILT CORRECTION (ROTATION_VECTOR):
       BNO085 fuses accel+gyro+mag → absolute orientation quaternion.
       No manual integration, bounded drift. Extract pitch, then:
       correction = depth * tan(pitch - baseline_pitch)

    2. DISPLACEMENT TRACKING with ZUPT (LINEAR_ACCELERATION):
       Between rest periods, double-integrate linear acceleration to
       track camera displacement from ground settlement.
       At each rest period: velocity must be zero (camera is still).
       Any residual velocity = drift → correct it (ZUPT).
       Reset integrator each rest period so drift never accumulates.

    3. STILLNESS WEIGHTING (LINEAR_ACCELERATION magnitude):
       Weight for rest-period averaging. Quiet frames count more.

    Calibration: collect baseline orientation while stationary at startup.
    """

    def __init__(self):
        self.calibrated       = False
        self.calib_pitches    = []
        self.CALIB_COUNT      = 200    # ~0.5s at 400Hz

        # Baseline pitch from calibration (radians)
        self.baseline_pitch   = 0.0

        # Latest state from fused IMU
        self.pitch            = 0.0    # current pitch (radians)
        self.linear_accel_mag = 0.0    # magnitude of linear acceleration (m/s²)

        # Stillness thresholds
        self.ACCEL_STILL_THRESHOLD = 0.3  # m/s² — tighter since gravity already removed

        # ZUPT displacement tracking
        # We only care about vertical (Y) displacement for height correction.
        # Accumulate accel samples during impact, then ZUPT-correct at next rest.
        self._zupt_accel_y    = []     # buffered accel_y samples during impact
        self._zupt_dt         = []     # dt for each sample
        self._zupt_active     = False  # True = currently integrating (impact phase)
        self._displacement_y  = 0.0    # accumulated camera displacement (meters)
        self._prev_time       = None

        # Buffer of unprocessed IMU packets
        self.pending = []

    @staticmethod
    def _quat_to_pitch(i, j, k, real):
        """Extract pitch (rotation around X axis) from quaternion."""
        sinp = 2.0 * (real * i + j * k)
        sinp = np.clip(sinp, -1.0, 1.0)
        return float(np.arcsin(sinp))

    def add_imu_packet(self, packet):
        """Called for every OAK-D IMU packet received from the device."""
        self.pending.append(packet)

    def process_sample(self, pitch, linear_accel_mag, accel_y, timestamp=None):
        """Process a single IMU sample (works for both OAK-D and RealSense)."""
        self.pitch = pitch
        self.linear_accel_mag = linear_accel_mag

        if not self.calibrated:
            self.calib_pitches.append(pitch)
            if len(self.calib_pitches) >= self.CALIB_COUNT:
                self.baseline_pitch = float(np.mean(self.calib_pitches))
                self.calibrated = True
                print(f"IMU calibrated. Baseline pitch: "
                      f"{np.degrees(self.baseline_pitch):.2f}°")
            return

        # ZUPT: buffer accel samples when active
        if self._zupt_active and timestamp is not None:
            if self._prev_time is not None:
                dt = timestamp - self._prev_time
                if 0 < dt < 0.1:
                    self._zupt_accel_y.append(accel_y)
                    self._zupt_dt.append(dt)
            self._prev_time = timestamp

    def _process_pending(self):
        """Process all pending OAK-D IMU packets."""
        for packet in self.pending:
            for imu_data in packet.packets:
                rv = imu_data.rotationVector
                pitch = self._quat_to_pitch(rv.i, rv.j, rv.k, rv.real)

                la = imu_data.acceleroMeter
                accel_mag = float(np.sqrt(la.x**2 + la.y**2 + la.z**2))

                timestamp = None
                if self._zupt_active:
                    timestamp = la.getTimestampDevice().total_seconds()

                self.process_sample(pitch, accel_mag, la.y, timestamp)

        self.pending.clear()

    def start_zupt(self):
        """Start recording accel for displacement tracking (call at blow)."""
        self._zupt_active = True
        self._zupt_accel_y.clear()
        self._zupt_dt.clear()
        self._prev_time = None

    def finish_zupt(self):
        """
        End displacement tracking (call when camera settles).
        Apply ZUPT correction: final velocity must be zero.
        Returns the vertical displacement in meters.
        """
        self._process_pending()

        if not self._zupt_active or len(self._zupt_accel_y) < 2:
            self._zupt_active = False
            return 0.0

        self._zupt_active = False

        accel = np.array(self._zupt_accel_y)
        dt_arr = np.array(self._zupt_dt)

        # Raw double integration
        velocity = np.cumsum(accel * dt_arr)
        # ZUPT: final velocity should be zero → subtract linear ramp
        # This removes constant bias drift
        drift_rate = velocity[-1] / np.sum(dt_arr)
        t_cumul = np.cumsum(dt_arr)
        velocity_corrected = velocity - drift_rate * t_cumul

        # Integrate corrected velocity → displacement
        displacement = np.sum(velocity_corrected * dt_arr)

        self._displacement_y += displacement
        self._zupt_accel_y.clear()
        self._zupt_dt.clear()

        return displacement

    def get_displacement_y(self):
        """Total accumulated vertical camera displacement (meters)."""
        return self._displacement_y

    def get_tilt_correction(self, tvec):
        """
        Returns height correction in meters based on pitch change
        from baseline. Uses tvec[2] (depth) to scale:
          correction = depth * tan(pitch - baseline)
        """
        self._process_pending()
        if not self.calibrated:
            return 0.0
        depth = float(np.asarray(tvec).flatten()[2])
        delta_pitch = self.pitch - self.baseline_pitch
        return depth * np.tan(delta_pitch)

    def is_frame_still(self):
        """Returns True if linear acceleration is below threshold."""
        self._process_pending()
        if not self.calibrated:
            return True
        return bool(self.linear_accel_mag < self.ACCEL_STILL_THRESHOLD)

    def get_stillness_weight(self):
        """
        Returns weight 0..1 based on linear acceleration magnitude.
        1.0 = still, approaches 0 = heavy vibration.
        """
        self._process_pending()
        if not self.calibrated:
            return 1.0
        return float(np.exp(-self.linear_accel_mag / self.ACCEL_STILL_THRESHOLD))

    def reset_tilt(self):
        """Re-baseline pitch to current orientation."""
        self.baseline_pitch = self.pitch

    @property
    def is_ready(self):
        return self.calibrated




class BlowDetector:
    """
    Detects hammer blows robustly despite post-impact vibration/bounce.

    Three conditions must ALL be true to count a blow:

    1. SPEED CHECK — hammer must be moving down fast enough
       velocity < -IMPACT_VELOCITY_THRESHOLD
       Filters out slow drift, camera noise, gentle swinging.

    2. DROP HEIGHT CHECK — hammer must have risen at least MIN_DROP_HEIGHT
       since the last blow before it can trigger again.
       This is the key fix: after impact the hammer bounces 5-10cm.
       Those bounces are fast but small. By requiring a real lift
       (e.g. 30cm) before the next blow counts, all post-impact
       vibration is ignored automatically — no matter how shaky.

    3. TIME LOCKOUT — MIN_BLOW_INTERVAL_S seconds must have passed
       since last blow. Backup safety net. For drop hammers the
       full cycle (lift + drop) is typically 3-5 seconds anyway.

    Set measurement uses rest-period averaging: after each blow, once the
    hammer settles (velocity near zero for SETTLE_FRAMES), the height is
    averaged over REST_AVG_FRAMES to get a precise rest height. Set is the
    difference between consecutive rest heights.
    """

    def __init__(self):
        self.history                  = deque(maxlen=10)
        self.blow_count               = 0
        self.last_blow_time           = 0
        self.last_set_mm              = None   # set on the most recent blow
        self.set_history              = []     # all set measurements (mm)
        self.prev_height              = None
        self.prev_time                = None
        self.peak_height_since_blow   = None   # tracks highest point since last blow

        # Tunable — adjust on site
        self.MIN_DROP_HEIGHT   = 0.30   # hammer must rise 30cm before next blow counts
        self.LOCKOUT_SECONDS   = 2.0    # hard time lockout after each blow

        # Rest-period averaging for accurate set measurement
        self.SETTLE_VELOCITY   = 0.10   # m/s — hammer considered "at rest" below this
        self.SETTLE_FRAMES     = 10     # consecutive slow frames before averaging starts
        self.REST_AVG_FRAMES   = 30     # frames to average (~0.5s at 60fps)

        self._settle_count     = 0      # consecutive frames below SETTLE_VELOCITY
        self._rest_samples     = []     # (height, weight) tuples during rest averaging
        self._awaiting_rest    = False  # True after blow, waiting to collect rest height
        self._last_rest_height = None   # averaged rest height from previous blow
        self._rest_height      = None   # averaged rest height from current blow (for display)

    def update(self, timestamp, height, stillness_weight=1.0):
        velocity = 0.0

        if self.prev_height is not None and self.prev_time is not None:
            dt = timestamp - self.prev_time
            if dt > 0:
                # positive = moving up, negative = moving down (dropping)
                velocity = (height - self.prev_height) / dt

        self.history.append((timestamp, height, velocity))
        self.prev_height = height
        self.prev_time   = timestamp

        # Track highest point since last blow (for drop height check)
        if self.peak_height_since_blow is None:
            self.peak_height_since_blow = height
        else:
            self.peak_height_since_blow = max(self.peak_height_since_blow, height)

        blow_detected = False
        set_per_blow  = None

        # --- Rest-period averaging state machine ---
        if self._awaiting_rest:
            if abs(velocity) < self.SETTLE_VELOCITY:
                self._settle_count += 1
            else:
                self._settle_count = 0
                self._rest_samples.clear()

            # Weighted averaging: still frames contribute more, noisy less
            if self._settle_count >= self.SETTLE_FRAMES and stillness_weight > 0:
                self._rest_samples.append((height, stillness_weight))

            if len(self._rest_samples) >= self.REST_AVG_FRAMES:
                heights = np.array([h for h, w in self._rest_samples])
                weights = np.array([w for h, w in self._rest_samples])
                rest_h = np.average(heights, weights=weights)
                self._rest_height = rest_h

                if self._last_rest_height is not None:
                    set_per_blow = (self._last_rest_height - rest_h) * 1000  # mm
                    self.last_set_mm = set_per_blow
                    self.set_history.append(set_per_blow)
                    set_str = f"set: {set_per_blow:.1f}mm"
                    print(f"  SET #{self.blow_count:4d} | "
                          f"rest height: {rest_h:.4f}m | "
                          f"{set_str} (avg of {self.REST_AVG_FRAMES} frames)")

                self._last_rest_height = rest_h
                self._awaiting_rest = False
                self._rest_samples.clear()
                self._settle_count = 0

        # How far has hammer risen since last blow?
        drop_available = (
            self.peak_height_since_blow - height
            if self.peak_height_since_blow is not None else 0
        )

        time_since_last = timestamp - self.last_blow_time

        # All conditions must pass (speed check disabled for testing)
        # speed_ok   = velocity < -IMPACT_VELOCITY_THRESHOLD
        drop_ok    = drop_available > self.MIN_DROP_HEIGHT
        lockout_ok = time_since_last > self.LOCKOUT_SECONDS

        if drop_ok and lockout_ok:
            self.blow_count         += 1
            self.last_blow_time      = timestamp
            blow_detected            = True
            self.peak_height_since_blow = None  # reset — start tracking lift for next blow

            # Start rest-period averaging for set measurement
            self._awaiting_rest = True
            self._settle_count  = 0
            self._rest_samples.clear()

            print(f"BLOW #{self.blow_count:4d} | "
                  f"velocity: {velocity:.2f}m/s | "
                  f"drop: {drop_available:.2f}m")

        return velocity, blow_detected, set_per_blow, drop_available


# ============================================================
# UI — START/STOP BUTTON
# ============================================================

# Button rect (x1, y1, x2, y2) — computed once, used for drawing + hit-test
_BTN_W, _BTN_H = 160, 50
_BTN_RECT = (
    CAMERA_WIDTH // 2 - _BTN_W // 2,
    CAMERA_HEIGHT - 70,
    CAMERA_WIDTH // 2 + _BTN_W // 2,
    CAMERA_HEIGHT - 70 + _BTN_H,
)

tracking_active = False


def draw_button(frame, active):
    """Draw START/STOP button on frame."""
    x1, y1, x2, y2 = _BTN_RECT
    color = (0, 0, 200) if active else (0, 180, 0)
    label = "STOP" if active else "START"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
    cv2.putText(frame, label, (x1 + (_BTN_W - tw) // 2, y2 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def build_summary_lines(blow_detector):
    """Build summary text lines from a completed session's blow detector."""
    lines = []
    if not blow_detector:
        return lines
    lines.append(f"Total blows: {blow_detector.blow_count}")
    if blow_detector.set_history:
        for i, s in enumerate(blow_detector.set_history, 1):
            lines.append(f"  #{i}: {s:.1f}mm")
        avg = sum(blow_detector.set_history) / len(blow_detector.set_history)
        lines.append(f"  Avg set: {avg:.1f}mm")
    return lines


def draw_summary(frame, lines, y_start=110):
    """Draw summary lines on frame."""
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, y_start + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def save_set_plot(blow_detector, session_num):
    """Save a bar chart of set per blow to PNG."""
    if not blow_detector or not blow_detector.set_history:
        return None
    sets = blow_detector.set_history
    blows = list(range(1, len(sets) + 1))
    avg = sum(sets) / len(sets)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(blows, sets, color='steelblue', edgecolor='white')
    ax.axhline(avg, color='red', linestyle='--', linewidth=1.5, label=f'Avg: {avg:.1f}mm')
    ax.set_xlabel('Blow #')
    ax.set_ylabel('Set (mm)')
    ax.set_title(f'Session {session_num} — Set per Blow')
    ax.legend()
    ax.set_xticks(blows)
    fig.tight_layout()

    filename = f"set_plot_session_{session_num}.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {filename}")
    return filename


def mouse_callback(event, x, y, flags, param):
    """Toggle tracking_active when user clicks the button."""
    global tracking_active
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    bx1, by1, bx2, by2 = _BTN_RECT
    if bx1 <= x <= bx2 and by1 <= y <= by2:
        tracking_active = not tracking_active


# ============================================================
# FRAME PROCESSING (shared by OAK-D and RealSense)
# ============================================================

def process_frame(frame, timestamp, camera_matrix, dist_coeffs,
                  imu_helper, blow_detector, writer, csv_file,
                  frame_count, start_time):
    """Process one frame: pose estimation, blow detection, overlays, CSV."""
    rvec, tvec, inlier_ratio, frame = estimate_pose(
        frame, camera_matrix, dist_coeffs)

    blow = False
    velocity = 0.0

    if tvec is not None:
        tilt_corr = imu_helper.get_tilt_correction(tvec)
        disp_corr = imu_helper.get_displacement_y()
        height = -float(tvec.flatten()[1]) + tilt_corr - disp_corr

        weight = imu_helper.get_stillness_weight()
        velocity, blow, set_per_blow, drop_available = blow_detector.update(
            timestamp, height, stillness_weight=weight)

        if blow:
            imu_helper.start_zupt()
        if blow_detector._awaiting_rest and \
           blow_detector._settle_count == blow_detector.SETTLE_FRAMES:
            imu_helper.finish_zupt()

        if inlier_ratio > 0.7 and abs(velocity) < 0.10:
            imu_helper.reset_tilt()

        # Overlay
        cv2.putText(frame, f"Height:   {height:.3f}m",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Velocity: {velocity:+.2f}m/s",
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        drop_color = (0, 255, 0) if drop_available > blow_detector.MIN_DROP_HEIGHT else (100, 100, 100)
        cv2.putText(frame, f"Drop avail: {drop_available:.2f}m",
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, drop_color, 2)
        cv2.putText(frame, f"Blows: {blow_detector.blow_count}",
                    (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        last_set_str = f"{blow_detector.last_set_mm:.1f}mm" if blow_detector.last_set_mm is not None else "--"
        cv2.putText(frame, f"Last set: {last_set_str}",
                    (10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        writer.writerow([f"{timestamp:.4f}", f"{height:.4f}",
                         f"{velocity:.4f}", int(blow),
                         f"{set_per_blow:.1f}" if set_per_blow is not None else "",
                         f"{inlier_ratio:.2f}"])
        csv_file.flush()

    else:
        cv2.putText(frame, "NO DETECTION", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # IMU status (always visible)
    imu_status = "IMU OK" if imu_helper.is_ready else "IMU calibrating..."
    imu_color = (0, 255, 0) if imu_helper.is_ready else (0, 165, 255)
    cv2.putText(frame, imu_status,
                (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, imu_color, 2)
    if imu_helper.is_ready:
        pitch_deg = np.degrees(imu_helper.pitch)
        accel_mag = imu_helper.linear_accel_mag
        cv2.putText(frame, f"Pitch: {pitch_deg:.2f} deg",
                    (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"Accel: {accel_mag:.3f} m/s2",
                    (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # FPS counter
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if blow:
        cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT),
                      (0, 255, 255), 8)

    return frame, blow


# ============================================================
# OAK-D MAIN LOOP
# ============================================================

def run_oakd():
    global tracking_active
    import depthai as dai

    pipeline = build_pipeline()
    if DEVICE_IP:
        device_info = dai.DeviceInfo(DEVICE_IP)
        device_ctx = dai.Device(pipeline, device_info)
    else:
        device_ctx = dai.Device(pipeline)

    print("Connecting to OAK-D...")

    with device_ctx as device:
        print(f"Connected: {device.getMxId()}")

        calib = device.readCalibration()
        camera_matrix = np.array(calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, CAMERA_WIDTH, CAMERA_HEIGHT))
        dist_coeffs = np.array(calib.getDistortionCoefficients(
            dai.CameraBoardSocket.CAM_A))

        print(f"Camera matrix loaded from device calibration")
        print(f"Starting at {CAMERA_FPS}fps, exposure {EXPOSURE_US}µs")
        print(f"Click START to begin tracking, Q to quit\n")

        q = device.getOutputQueue("video", maxSize=4, blocking=False)
        imu_q = device.getOutputQueue("imu", maxSize=50, blocking=False)

        cv2.namedWindow("Hammer Tracker", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Hammer Tracker", mouse_callback)

        tracking_active = False
        was_active = False
        blow_detector = None
        imu_helper = None
        csv_file = None
        writer = None
        session_num = 0
        frame_count = 0
        start_time = time.time()
        summary_lines = []

        try:
            while True:
                pkt = q.tryGet()
                if pkt is None:
                    continue

                frame = pkt.getCvFrame()
                timestamp = time.time() - start_time
                frame_count += 1

                # --- Transition: start tracking ---
                if tracking_active and not was_active:
                    session_num += 1
                    session_tag = time.strftime("%Y%m%d_%H%M%S")
                    imu_helper = IMUHelper()
                    blow_detector = BlowDetector()
                    csv_name = f"hammer_log_{session_tag}.csv"
                    csv_file = open(csv_name, "w", newline="")
                    writer = csv.writer(csv_file)
                    writer.writerow(["timestamp", "height_m", "velocity_ms_pos_up",
                                     "blow", "set_mm", "inlier_ratio"])
                    summary_lines = []
                    print(f"\n--- Session {session_num} started → {csv_name} ---")

                # --- Transition: stop tracking ---
                if not tracking_active and was_active:
                    summary_lines = build_summary_lines(blow_detector)
                    save_set_plot(blow_detector, session_tag)
                    blows = blow_detector.blow_count if blow_detector else 0
                    if csv_file:
                        csv_file.close()
                        csv_file = None
                    print(f"--- Session {session_num} stopped. {blows} blows → {csv_name} ---\n")
                    blow_detector = None
                    imu_helper = None

                was_active = tracking_active

                if tracking_active:
                    # Drain IMU packets
                    while True:
                        imu_pkt = imu_q.tryGet()
                        if imu_pkt is None:
                            break
                        imu_helper.add_imu_packet(imu_pkt)

                    frame, blow = process_frame(
                        frame, timestamp, camera_matrix, dist_coeffs,
                        imu_helper, blow_detector, writer, csv_file,
                        frame_count, start_time)
                else:
                    # Preview only — show ChArUco overlay but no tracking
                    _, _, _, frame = estimate_pose(frame, camera_matrix, dist_coeffs)
                    # Drain and discard IMU packets to avoid queue buildup
                    while imu_q.tryGet() is not None:
                        pass
                    if summary_lines:
                        draw_summary(frame, summary_lines)

                draw_button(frame, tracking_active)
                cv2.imshow("Hammer Tracker", frame)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    snap = f"snapshot_{int(timestamp)}.png"
                    cv2.imwrite(snap, frame)
                    print(f"Saved {snap}")

        finally:
            if csv_file:
                csv_file.close()
            cv2.destroyAllWindows()
            if blow_detector:
                print(f"\nDone. {blow_detector.blow_count} blows logged")


# ============================================================
# REALSENSE MAIN LOOP
# ============================================================

def run_realsense():
    global tracking_active
    import pyrealsense2 as rs

    print("Connecting to RealSense D435i...")
    pipe, profile = build_realsense_pipeline()
    camera_matrix, dist_coeffs = get_realsense_intrinsics(profile)

    print(f"Camera matrix loaded from RealSense calibration")
    print(f"Click START to begin tracking, Q to quit\n")

    cv2.namedWindow("Hammer Tracker", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Hammer Tracker", mouse_callback)

    tracking_active = False
    was_active = False
    madgwick = None
    imu_helper = None
    blow_detector = None
    csv_file = None
    writer = None
    session_num = 0
    frame_count = 0
    start_time = time.time()
    last_gyro_ts = None
    summary_lines = []

    try:
        while True:
            frames = pipe.wait_for_frames()

            # --- Transition: start tracking ---
            if tracking_active and not was_active:
                session_num += 1
                session_tag = time.strftime("%Y%m%d_%H%M%S")
                madgwick = MadgwickFilter(beta=0.1)
                imu_helper = IMUHelper()
                imu_helper.CALIB_COUNT = 100
                blow_detector = BlowDetector()
                csv_name = f"hammer_log_{session_tag}.csv"
                csv_file = open(csv_name, "w", newline="")
                writer = csv.writer(csv_file)
                writer.writerow(["timestamp", "height_m", "velocity_ms_pos_up",
                                 "blow", "set_mm", "inlier_ratio"])
                last_gyro_ts = None
                summary_lines = []
                print(f"\n--- Session {session_num} started → {csv_name} ---")

            # --- Transition: stop tracking ---
            if not tracking_active and was_active:
                summary_lines = build_summary_lines(blow_detector)
                save_set_plot(blow_detector, session_tag)
                blows = blow_detector.blow_count if blow_detector else 0
                if csv_file:
                    csv_file.close()
                    csv_file = None
                print(f"--- Session {session_num} stopped. {blows} blows → {csv_name} ---\n")
                blow_detector = None
                imu_helper = None
                madgwick = None

            was_active = tracking_active

            # Process IMU when tracking
            if tracking_active:
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)

                if accel_frame and gyro_frame:
                    ad = accel_frame.as_motion_frame().get_motion_data()
                    gd = gyro_frame.as_motion_frame().get_motion_data()

                    accel = np.array([ad.x, ad.y, ad.z])
                    gyro = np.array([gd.x, gd.y, gd.z])

                    t_now = gyro_frame.get_timestamp() / 1000.0
                    dt = (t_now - last_gyro_ts) if last_gyro_ts is not None else 0.005
                    last_gyro_ts = t_now

                    if 0 < dt < 0.1:
                        madgwick.update(gyro, accel, dt)

                    pitch = madgwick.get_pitch()
                    lin_accel = madgwick.remove_gravity(accel)
                    lin_mag = float(np.linalg.norm(lin_accel))
                    imu_helper.process_sample(pitch, lin_mag, lin_accel[1], t_now)

            # Get color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            timestamp = time.time() - start_time
            frame_count += 1

            if tracking_active:
                frame, blow = process_frame(
                    frame, timestamp, camera_matrix, dist_coeffs,
                    imu_helper, blow_detector, writer, csv_file,
                    frame_count, start_time)
            else:
                _, _, _, frame = estimate_pose(frame, camera_matrix, dist_coeffs)
                if summary_lines:
                    draw_summary(frame, summary_lines)

            draw_button(frame, tracking_active)
            cv2.imshow("Hammer Tracker", frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord('s'):
                snap = f"snapshot_{int(timestamp)}.png"
                cv2.imwrite(snap, frame)
                print(f"Saved {snap}")

    finally:
        pipe.stop()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()
        if blow_detector:
            print(f"\nDone. {blow_detector.blow_count} blows logged")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Pile Driving Hammer Tracker")
    parser.add_argument("--intel", action="store_true",
                        help="Use RealSense D435i instead of OAK-D")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.intel:
        run_realsense()
    else:
        run_oakd()


if __name__ == "__main__":
    main()