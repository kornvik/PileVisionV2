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

import depthai as dai
import cv2
import numpy as np
import time
import csv
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
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(CAMERA_FPS)
    cam.setInterleaved(False)

    # Manual exposure to freeze motion blur
    cam.initialControl.setManualExposure(EXPOSURE_US, ISO)

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
# POSE ESTIMATION
# ============================================================

def estimate_pose(frame, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = charuco_detector.detectBoard(gray)

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
        self.calib_quats      = []
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
        """Called for every IMU packet received from the device."""
        self.pending.append(packet)

    def _process_pending(self):
        """Process all pending IMU packets."""
        for packet in self.pending:
            for imu_data in packet.packets:
                # Rotation vector → quaternion (i, j, k, real)
                rv = imu_data.rotationVector
                self.pitch = self._quat_to_pitch(rv.i, rv.j, rv.k, rv.real)

                # Linear acceleration (gravity already removed by BNO085)
                la = imu_data.linearAcceleration
                self.linear_accel_mag = float(np.sqrt(
                    la.x**2 + la.y**2 + la.z**2))

                # Calibration: collect baseline orientation
                if not self.calibrated:
                    self.calib_quats.append(self.pitch)
                    if len(self.calib_quats) >= self.CALIB_COUNT:
                        self.baseline_pitch = float(np.mean(self.calib_quats))
                        self.calibrated = True
                        print(f"IMU calibrated. Baseline pitch: "
                              f"{np.degrees(self.baseline_pitch):.2f}°")
                    continue

                # ZUPT: buffer accel samples when active
                if self._zupt_active:
                    t = imu_data.linearAcceleration.getTimestampDevice() \
                        .total_seconds()
                    if self._prev_time is not None:
                        dt = t - self._prev_time
                        if 0 < dt < 0.1:
                            self._zupt_accel_y.append(la.y)
                            self._zupt_dt.append(dt)
                    self._prev_time = t

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

        # All three conditions must pass
        speed_ok   = velocity < -IMPACT_VELOCITY_THRESHOLD
        drop_ok    = drop_available > self.MIN_DROP_HEIGHT
        lockout_ok = time_since_last > self.LOCKOUT_SECONDS

        if speed_ok and drop_ok and lockout_ok:
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
# MAIN
# ============================================================

def main():
    pipeline = build_pipeline()

    # Connect to device
    if DEVICE_IP:
        device_info = dai.DeviceInfo(DEVICE_IP)
        device_ctx  = dai.Device(pipeline, device_info)
    else:
        device_ctx  = dai.Device(pipeline)

    print("Connecting to OAK-D...")

    with device_ctx as device:
        print(f"Connected: {device.getMxId()}")

        # Get factory calibration — no manual calibration needed
        calib         = device.readCalibration()
        camera_matrix = np.array(calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, CAMERA_WIDTH, CAMERA_HEIGHT))
        dist_coeffs   = np.array(calib.getDistortionCoefficients(
            dai.CameraBoardSocket.CAM_A))

        print(f"Camera matrix loaded from device calibration")
        print(f"Starting at {CAMERA_FPS}fps, exposure {EXPOSURE_US}µs")
        print(f"Press Q to quit, S to save snapshot\n")

        q            = device.getOutputQueue("video", maxSize=4, blocking=False)
        imu_q        = device.getOutputQueue("imu",   maxSize=50, blocking=False)
        blow_detector  = BlowDetector()
        imu_helper     = IMUHelper()

        # CSV logging — use a reference so we can close in finally
        csv_file = open(OUTPUT_CSV, "w", newline="")
        writer   = csv.writer(csv_file)
        writer.writerow(["timestamp", "height_m", "velocity_ms_pos_up",
                         "blow", "set_mm", "inlier_ratio"])

        frame_count  = 0
        start_time   = time.time()

        try:
            while True:
                pkt = q.tryGet()
                if pkt is None:
                    continue

                frame     = pkt.getCvFrame()
                timestamp = time.time() - start_time
                frame_count += 1

                # Drain all IMU packets that arrived since last frame
                while True:
                    imu_pkt = imu_q.tryGet()
                    if imu_pkt is None:
                        break
                    imu_helper.add_imu_packet(imu_pkt)

                rvec, tvec, inlier_ratio, frame = estimate_pose(
                    frame, camera_matrix, dist_coeffs)

                blow          = False
                set_per_blow  = None
                velocity      = 0.0
                height        = None

                if tvec is not None:
                    # tvec[1] = vertical axis, negate so up = positive
                    # Apply tilt correction + accumulated displacement
                    tilt_corr = imu_helper.get_tilt_correction(tvec)
                    disp_corr = imu_helper.get_displacement_y()
                    height = -float(tvec[1]) + tilt_corr - disp_corr

                    # Weighted averaging: stillness weight from IMU
                    weight = imu_helper.get_stillness_weight()
                    velocity, blow, set_per_blow, drop_available = blow_detector.update(
                        timestamp, height, stillness_weight=weight)

                    # ZUPT: start tracking on blow, finish when settled
                    if blow:
                        imu_helper.start_zupt()
                    if blow_detector._awaiting_rest and \
                       blow_detector._settle_count == blow_detector.SETTLE_FRAMES:
                        imu_helper.finish_zupt()

                    # Reset tilt baseline when vision is confident during rest
                    if inlier_ratio > 0.7 and abs(velocity) < 0.10:
                        imu_helper.reset_tilt()

                    # IMU status
                    imu_status = "IMU OK" if imu_helper.is_ready else "IMU calibrating..."
                    imu_color  = (0, 255, 0) if imu_helper.is_ready else (0, 165, 255)
                    cv2.putText(frame, imu_status,
                                (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, imu_color, 2)

                    # Overlay
                    cv2.putText(frame, f"Height:   {height:.3f}m",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Velocity: {velocity:+.2f}m/s",
                                (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    # Drop available — goes green when enough height for a valid blow
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

                # FPS counter
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                # Flash on blow
                if blow:
                    cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT),
                                  (0, 255, 255), 8)

                cv2.imshow("Hammer Tracker", frame)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    snap = f"snapshot_{int(timestamp)}.png"
                    cv2.imwrite(snap, frame)
                    print(f"Saved {snap}")

        finally:
            csv_file.close()
            cv2.destroyAllWindows()
            print(f"\nDone. {blow_detector.blow_count} blows logged to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()