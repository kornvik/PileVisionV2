"""Tests for hammer_tracker.py — BlowDetector, IMUCompensator, and estimate_pose."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from collections import namedtuple

from hammer_tracker import BlowDetector, IMUCompensator, estimate_pose, IMPACT_VELOCITY_THRESHOLD


# ============================================================
# BlowDetector Tests
# ============================================================

class TestBlowDetector:

    def make_detector(self):
        d = BlowDetector()
        d.LOCKOUT_SECONDS = 2.0
        d.MIN_DROP_HEIGHT = 0.30
        return d

    def simulate_cycle(self, detector, t_start, rest_height, lift_height, impact_height, dt=0.05):
        """Simulate one full hammer cycle: rest → lift → drop → impact → settle.
        Returns list of (velocity, blow, set_mm, drop_avail) tuples."""
        results = []
        t = t_start

        # Lift phase: rise from rest_height to lift_height over ~0.5s
        lift_steps = 10
        for i in range(lift_steps):
            h = rest_height + (lift_height - rest_height) * (i / lift_steps)
            results.append(detector.update(t, h))
            t += dt

        # Peak (briefly stationary)
        results.append(detector.update(t, lift_height))
        t += dt

        # Drop phase: fall from lift_height to impact_height over ~0.3s
        drop_steps = 6
        for i in range(drop_steps):
            h = lift_height - (lift_height - impact_height) * (i / drop_steps)
            results.append(detector.update(t, h))
            t += dt

        # Impact frame
        results.append(detector.update(t, impact_height))
        t += dt

        # Rest phase: hammer sits at impact_height for rest-period averaging
        # Need SETTLE_FRAMES (10) + REST_AVG_FRAMES (30) = 40 frames at low velocity
        for _ in range(50):
            results.append(detector.update(t, impact_height))
            t += dt

        return results, t

    def test_no_blow_on_first_frame(self):
        d = self.make_detector()
        vel, blow, set_mm, drop = d.update(0.0, 1.0)
        assert blow is False
        assert vel == 0.0
        assert set_mm is None

    def test_no_blow_from_small_movement(self):
        """Small oscillations (bounces) should NOT trigger blows."""
        d = self.make_detector()
        # Feed small up-down movements (5cm amplitude) — simulates post-impact bounce
        t = 0.0
        dt = 0.02
        for _ in range(50):
            d.update(t, 1.0 + 0.05 * np.sin(t * 20))
            t += dt

        assert d.blow_count == 0

    def test_no_blow_without_enough_drop(self):
        """Hammer hasn't risen enough → no blow even with fast velocity and past lockout."""
        d = self.make_detector()

        # First valid blow (after lockout period)
        d.update(0.0, 1.0)
        d.update(2.5, 2.0)  # rise (past 2s lockout)
        d.update(3.0, 2.0)  # peak
        vel, blow, _, _ = d.update(3.05, 0.5)  # fast drop → blow #1
        assert blow is True

        # After blow, peak resets. Rise only 15cm (need 30cm)
        d.update(6.0, 0.55)
        d.update(6.05, 0.60)
        d.update(6.10, 0.65)  # peak at 0.65
        vel, blow, _, _ = d.update(6.12, 0.40)  # drop = 0.65-0.40 = 0.25m < 0.30
        assert blow is False

    def test_single_blow_detection(self):
        """A full lift-then-drop cycle should detect exactly one blow."""
        d = self.make_detector()

        # Note: last_blow_time starts at 0, so lockout requires t > 2.0s
        d.update(0.0, 1.0)
        d.update(1.0, 1.5)
        d.update(2.0, 2.0)  # peak
        d.update(2.5, 2.0)  # still at peak (past lockout)

        # Fast drop — velocity = (1.0 - 2.0) / 0.05 = -20 m/s
        vel, blow, set_mm, drop = d.update(2.55, 1.0)

        assert d.blow_count == 1
        assert blow is True

    def test_set_per_blow_via_rest_averaging(self):
        """Set is measured from rest-period height averaging between blows."""
        d = self.make_detector()
        dt = 1.0 / 60  # 60fps

        # First blow (past lockout)
        d.update(0.0, 1.0)
        d.update(2.5, 2.0)
        d.update(3.0, 2.0)
        _, blow, _, _ = d.update(3.05, 0.5)  # fast drop → blow
        assert blow is True

        # Rest at 1.0m — settle + averaging (10 + 30 = 40 frames)
        t = 3.1
        for _ in range(50):
            d.update(t, 1.0)
            t += dt

        # Second blow — pile sinks to 0.95m (50mm penetration)
        d.update(t + 2.5, 2.0)  # rise (past lockout)
        d.update(t + 3.0, 2.0)  # peak
        _, blow, _, _ = d.update(t + 3.05, 0.5)  # fast drop → blow
        assert blow is True

        # Rest at 0.95m — this triggers set calculation
        t2 = t + 3.1
        set_mm = None
        for _ in range(50):
            _, _, s, _ = d.update(t2, 0.95)
            if s is not None:
                set_mm = s
            t2 += dt

        assert set_mm is not None
        assert abs(set_mm - 50.0) < 1.0  # ~50mm penetration

    def test_lockout_prevents_rapid_blows(self):
        """Time lockout prevents blows detected closer than LOCKOUT_SECONDS."""
        d = self.make_detector()
        d.LOCKOUT_SECONDS = 2.0

        # First blow (past initial lockout)
        d.update(0.0, 1.0)
        d.update(2.5, 2.0)   # rise
        d.update(3.0, 2.0)   # peak
        d.update(3.05, 0.5)  # fast drop → blow #1 at t=3.05
        assert d.blow_count == 1

        # Try second blow 1.5s later at t=4.55 (inside lockout: 4.55-3.05 = 1.5 < 2.0)
        d.update(4.0, 2.0)    # lift — enough drop
        d.update(4.5, 2.0)    # peak
        d.update(4.55, 0.3)   # fast drop, but lockout
        assert d.blow_count == 1  # should NOT count

    def test_velocity_direction(self):
        """Positive velocity = moving up, negative = moving down."""
        d = self.make_detector()

        d.update(0.0, 1.0)
        vel_up, _, _, _ = d.update(0.1, 1.5)  # 0.5m in 0.1s = +5 m/s
        assert vel_up > 0

        vel_down, _, _, _ = d.update(0.2, 0.5)  # -1.0m in 0.1s = -10 m/s
        assert vel_down < 0

    def test_drop_available_tracking(self):
        """drop_available should track height from peak since last blow."""
        d = self.make_detector()

        d.update(0.0, 1.0)
        _, _, _, drop = d.update(0.1, 1.5)
        assert drop == pytest.approx(0.0, abs=0.01)  # at peak, no drop yet

        _, _, _, drop = d.update(0.2, 1.0)
        assert drop == pytest.approx(0.5, abs=0.01)  # dropped 0.5 from peak of 1.5

    def test_peak_resets_after_blow(self):
        """After a blow, peak tracking resets so small bounces don't re-trigger."""
        d = self.make_detector()

        # First blow (past lockout)
        d.update(0.0, 1.0)
        d.update(2.5, 2.0)
        d.update(3.0, 2.0)
        d.update(3.05, 0.5)
        assert d.blow_count == 1
        assert d.peak_height_since_blow is None  # reset after blow

    def test_multiple_blow_cycle(self):
        """Simulate 5 full hammer cycles, each should detect exactly one blow."""
        d = self.make_detector()
        # Start at t=3.0 to be past the initial lockout (last_blow_time=0, lockout=2s)
        t = 3.0

        for i in range(5):
            rest_h = 1.0 - i * 0.05  # pile sinks each blow
            lift_h = rest_h + 1.0
            impact_h = rest_h - 0.05

            results, t = self.simulate_cycle(d, t, rest_h, lift_h, impact_h)

            # Lockout gap between cycles
            t += 2.5

        assert d.blow_count == 5

    def test_history_maxlen(self):
        """History deque should not grow beyond maxlen."""
        d = self.make_detector()
        for i in range(100):
            d.update(float(i), 1.0 + 0.01 * i)
        assert len(d.history) == 10  # maxlen=10


# ============================================================
# IMUCompensator Tests
# ============================================================

class TestIMUCompensator:

    def make_imu_packet(self, accel_samples):
        """Create a mock IMU packet with given accel samples.
        Each sample is (x, y, z, timestamp_seconds).
        """
        packet = MagicMock()
        imu_datas = []
        for ax, ay, az, ts in accel_samples:
            imu_data = MagicMock()
            imu_data.acceleroMeter.x = ax
            imu_data.acceleroMeter.y = ay
            imu_data.acceleroMeter.z = az
            td = MagicMock()
            td.total_seconds.return_value = ts
            imu_data.acceleroMeter.getTimestampDevice.return_value = td
            imu_datas.append(imu_data)
        packet.packets = imu_datas
        return packet

    def test_initial_state(self):
        comp = IMUCompensator()
        assert comp.is_ready is False
        assert np.allclose(comp.velocity, [0, 0, 0])
        assert np.allclose(comp.displacement, [0, 0, 0])

    def test_calibration_phase(self):
        """Compensator should not be ready until CALIB_COUNT samples collected."""
        comp = IMUCompensator()
        comp.CALIB_COUNT = 10  # reduce for test speed

        # Feed 9 stationary samples (gravity = 0, -9.81, 0 for example)
        samples = [(0, -9.81, 0, 0.001 * i) for i in range(9)]
        comp.add_imu_packet(self.make_imu_packet(samples))
        comp.get_correction()
        assert comp.is_ready is False

        # Feed 1 more → should calibrate
        comp.add_imu_packet(self.make_imu_packet([(0, -9.81, 0, 0.01)]))
        comp.get_correction()
        assert comp.is_ready is True

    def test_gravity_estimation(self):
        """During calibration, gravity vector should be mean of stationary samples."""
        comp = IMUCompensator()
        comp.CALIB_COUNT = 5

        # Gravity along Y-axis = -9.81
        samples = [(0.0, -9.81, 0.0, 0.001 * i) for i in range(5)]
        comp.add_imu_packet(self.make_imu_packet(samples))
        comp.get_correction()

        assert comp.is_ready is True
        assert np.allclose(comp.gravity_vec, [0, -9.81, 0], atol=0.01)

    def test_zero_correction_when_stationary(self):
        """If camera isn't shaking (accel = gravity), correction should be ~zero."""
        comp = IMUCompensator()
        comp.CALIB_COUNT = 5

        # Calibration phase
        calib_samples = [(0, -9.81, 0, 0.001 * i) for i in range(5)]
        comp.add_imu_packet(self.make_imu_packet(calib_samples))
        comp.get_correction()

        # Post-calibration: feed more stationary samples
        post_samples = [(0, -9.81, 0, 0.01 + 0.0025 * i) for i in range(20)]
        comp.add_imu_packet(self.make_imu_packet(post_samples))
        correction = comp.get_correction()

        assert np.allclose(correction, [0, 0, 0], atol=0.001)

    def test_shake_produces_nonzero_correction(self):
        """Camera shake (accel != gravity) should produce nonzero correction."""
        comp = IMUCompensator()
        comp.CALIB_COUNT = 5

        # Calibration
        calib = [(0, -9.81, 0, 0.001 * i) for i in range(5)]
        comp.add_imu_packet(self.make_imu_packet(calib))
        comp.get_correction()

        # Shake: add 2 m/s² in X direction
        shake = [(2.0, -9.81, 0, 0.01 + 0.0025 * i) for i in range(20)]
        comp.add_imu_packet(self.make_imu_packet(shake))
        correction = comp.get_correction()

        # X displacement should be positive (accelerated in +X)
        assert correction[0] > 0
        # Y and Z should be ~0
        assert abs(correction[1]) < 0.001
        assert abs(correction[2]) < 0.001

    def test_reset_drift(self):
        """reset_drift should zero velocity but keep displacement (ground shift)."""
        comp = IMUCompensator()
        comp.CALIB_COUNT = 5

        calib = [(0, -9.81, 0, 0.001 * i) for i in range(5)]
        comp.add_imu_packet(self.make_imu_packet(calib))
        comp.get_correction()

        # Add some shake to build up velocity + displacement
        shake = [(5.0, -9.81, 0, 0.01 + 0.0025 * i) for i in range(10)]
        comp.add_imu_packet(self.make_imu_packet(shake))
        comp.get_correction()

        disp_before = comp.displacement.copy()
        comp.reset_drift()
        assert np.allclose(comp.velocity, [0, 0, 0])
        # Displacement preserved — tracks real camera movement
        assert np.allclose(comp.displacement, disp_before)

    def test_bad_timestamps_skipped(self):
        """Samples with dt <= 0 or dt > 0.1 should be skipped."""
        comp = IMUCompensator()
        comp.CALIB_COUNT = 3

        # Calibration
        calib = [(0, -9.81, 0, 0.001 * i) for i in range(3)]
        comp.add_imu_packet(self.make_imu_packet(calib))
        comp.get_correction()

        # Bad timestamp: backwards jump
        bad = [
            (5.0, -9.81, 0, 0.01),   # first post-calib (sets prev_time)
            (5.0, -9.81, 0, 0.005),   # dt = -0.005 → skipped
            (5.0, -9.81, 0, 0.50),    # dt = 0.49 > 0.1 → skipped
        ]
        comp.add_imu_packet(self.make_imu_packet(bad))
        correction = comp.get_correction()

        # Only the first sample set prev_time, subsequent were skipped
        # so displacement should be zero
        assert np.allclose(correction, [0, 0, 0], atol=0.001)

    def test_correction_returns_copy(self):
        """get_correction should return a copy, not a reference to internal state."""
        comp = IMUCompensator()
        comp.CALIB_COUNT = 3

        calib = [(0, -9.81, 0, 0.001 * i) for i in range(3)]
        comp.add_imu_packet(self.make_imu_packet(calib))

        c1 = comp.get_correction()
        c1[0] = 999.0
        c2 = comp.get_correction()
        assert c2[0] != 999.0


# ============================================================
# estimate_pose Tests
# ============================================================

class TestEstimatePose:

    def test_returns_none_when_no_markers(self):
        """No markers detected → all None returns."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam_matrix = np.eye(3)
        dist = np.zeros(5)

        with patch('hammer_tracker.charuco_detector') as mock_det:
            mock_det.detectBoard.return_value = (None, None, None)
            rvec, tvec, ratio, out_frame = estimate_pose(frame, cam_matrix, dist)

        assert rvec is None
        assert tvec is None
        assert ratio == 0

    def test_returns_none_when_too_few_ids(self):
        """< 4 detected IDs → returns None."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam_matrix = np.eye(3)
        dist = np.zeros(5)

        corners = np.array([[[100, 100]], [[200, 200]], [[300, 300]]], dtype=np.float32)
        ids = np.array([[0], [1], [2]])

        with patch('hammer_tracker.charuco_detector') as mock_det:
            mock_det.detectBoard.return_value = (corners, ids, None)
            rvec, tvec, ratio, out_frame = estimate_pose(frame, cam_matrix, dist)

        assert rvec is None
        assert tvec is None

    def test_returns_none_when_solvepnp_fails(self):
        """solvePnPRansac returns ret=False → returns None."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam_matrix = np.eye(3) * 500
        cam_matrix[2, 2] = 1
        dist = np.zeros(5)

        corners = np.array([[[100, 100]], [[200, 100]], [[300, 100]], [[400, 100]]], dtype=np.float32)
        ids = np.array([[0], [1], [2], [3]])

        with patch('hammer_tracker.charuco_detector') as mock_det, \
             patch('hammer_tracker.board') as mock_board, \
             patch('cv2.solvePnPRansac') as mock_solve:
            mock_det.detectBoard.return_value = (corners, ids, None)
            mock_board.matchImagePoints.return_value = (
                np.zeros((4, 3), dtype=np.float32),
                corners.reshape(4, 1, 2)
            )
            mock_solve.return_value = (False, None, None, None)

            rvec, tvec, ratio, out_frame = estimate_pose(frame, cam_matrix, dist)

        assert rvec is None
        assert tvec is None

    def test_successful_pose_returns_values(self):
        """Successful detection → returns rvec, tvec, and positive inlier_ratio."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam_matrix = np.eye(3) * 500
        cam_matrix[2, 2] = 1
        dist = np.zeros(5)

        n_corners = 6
        corners = np.random.rand(n_corners, 1, 2).astype(np.float32) * 400
        ids = np.arange(n_corners).reshape(-1, 1)

        mock_rvec = np.array([[0.1], [0.2], [0.3]])
        mock_tvec = np.array([[0.0], [-1.5], [3.0]])
        inliers = np.arange(5).reshape(-1, 1)  # 5 of 6 = 83%

        with patch('hammer_tracker.charuco_detector') as mock_det, \
             patch('hammer_tracker.board') as mock_board, \
             patch('cv2.solvePnPRansac') as mock_solve, \
             patch('cv2.aruco.drawDetectedCornersCharuco'), \
             patch('cv2.drawFrameAxes'), \
             patch('cv2.putText'):
            mock_det.detectBoard.return_value = (corners, ids, None)
            mock_board.matchImagePoints.return_value = (
                np.zeros((n_corners, 3), dtype=np.float32),
                corners
            )
            mock_solve.return_value = (True, mock_rvec, mock_tvec, inliers)

            rvec, tvec, ratio, out_frame = estimate_pose(frame, cam_matrix, dist)

        assert rvec is not None
        assert tvec is not None
        assert ratio == pytest.approx(5 / 6, abs=0.01)

    def test_inlier_ratio_calculation(self):
        """inlier_ratio = len(inliers) / len(ids)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam_matrix = np.eye(3) * 500
        cam_matrix[2, 2] = 1
        dist = np.zeros(5)

        n_corners = 10
        corners = np.random.rand(n_corners, 1, 2).astype(np.float32) * 400
        ids = np.arange(n_corners).reshape(-1, 1)
        inliers = np.arange(7).reshape(-1, 1)  # 7 of 10 = 70%

        with patch('hammer_tracker.charuco_detector') as mock_det, \
             patch('hammer_tracker.board') as mock_board, \
             patch('cv2.solvePnPRansac') as mock_solve, \
             patch('cv2.aruco.drawDetectedCornersCharuco'), \
             patch('cv2.drawFrameAxes'), \
             patch('cv2.putText'):
            mock_det.detectBoard.return_value = (corners, ids, None)
            mock_board.matchImagePoints.return_value = (
                np.zeros((n_corners, 3), dtype=np.float32),
                corners
            )
            mock_solve.return_value = (True, np.zeros((3, 1)), np.zeros((3, 1)), inliers)

            _, _, ratio, _ = estimate_pose(frame, cam_matrix, dist)

        assert ratio == pytest.approx(0.7, abs=0.01)
