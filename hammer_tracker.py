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
# BLOW DETECTOR
# ============================================================

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
    """

    def __init__(self):
        self.history                  = deque(maxlen=10)
        self.blow_count               = 0
        self.last_blow_time           = 0
        self.last_blow_height         = None   # hammer height at last impact
        self.prev_height              = None
        self.prev_time                = None
        self.peak_height_since_blow   = None   # tracks highest point since last blow
        self.in_lockout               = False  # True during post-impact shake window

        # Tunable — adjust on site
        self.MIN_DROP_HEIGHT   = 0.30   # hammer must rise 30cm before next blow counts
        self.LOCKOUT_SECONDS   = 2.0    # hard time lockout after each blow

    def update(self, timestamp, height):
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

            # Set per blow = how much lower than last impact
            if self.last_blow_height is not None:
                set_per_blow = (self.last_blow_height - height) * 1000  # mm, positive = penetrated
            self.last_blow_height = height

            set_str = f"set: {set_per_blow:.1f}mm" if set_per_blow is not None else "set: --"
            print(f"BLOW #{self.blow_count:4d} | "
                  f"height: {height:.3f}m | "
                  f"velocity: {velocity:.2f}m/s | "
                  f"drop: {drop_available:.2f}m | "
                  f"{set_str}")

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
        blow_detector = BlowDetector()

        # CSV logging
        csv_file = open(OUTPUT_CSV, "w", newline="")
        writer   = csv.writer(csv_file)
        writer.writerow(["timestamp", "height_m", "velocity_ms_pos_up",
                         "blow", "set_mm", "inlier_ratio"])

        frame_count  = 0
        start_time   = time.time()

        while True:
            pkt = q.tryGet()
            if pkt is None:
                continue

            frame     = pkt.getCvFrame()
            timestamp = time.time() - start_time
            frame_count += 1

            rvec, tvec, inlier_ratio, frame = estimate_pose(
                frame, camera_matrix, dist_coeffs)

            blow          = False
            set_per_blow  = None
            velocity      = 0.0
            height        = None

            if tvec is not None:
                # tvec[1] = vertical axis (camera looking sideways at hammer)
                # Negate so "up" is positive
                height = -float(tvec[1])

                velocity, blow, set_per_blow, drop_available = blow_detector.update(timestamp, height)

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

                writer.writerow([f"{timestamp:.4f}", f"{height:.4f}",
                                 f"{velocity:.4f}", int(blow),
                                 f"{set_per_blow:.1f}" if set_per_blow is not None else "",
                                 f"{inlier_ratio:.2f}"])
                csv_file.flush()

            else:
                cv2.putText(frame, "NO DETECTION", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # FPS counter
            fps = frame_count / (time.time() - start_time)
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

        csv_file.close()
        cv2.destroyAllWindows()
        print(f"\nDone. {blow_detector.blow_count} blows logged to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()