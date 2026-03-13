"""Tests for hammer_tracker.py — BlowDetector, IMUCompensator, and estimate_pose."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from collections import namedtuple

from hammer_tracker import BlowDetector, IMUHelper, MadgwickFilter, estimate_pose, IMPACT_VELOCITY_THRESHOLD


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

    def test_weighted_averaging_weights_noisy_frames_less(self):
        """Noisy frames (low weight) should contribute less to rest average."""
        d = self.make_detector()
        dt = 1.0 / 60

        # First blow
        d.update(0.0, 1.0)
        d.update(2.5, 2.0)
        d.update(3.0, 2.0)
        _, blow, _, _ = d.update(3.05, 0.5)
        assert blow is True

        # Settle phase
        t = 3.1
        for _ in range(15):
            d.update(t, 1.0)
            t += dt

        # Averaging phase: alternate 1.000m (weight=1.0) and 1.001m (weight=0.01)
        # Small height diff keeps velocity < SETTLE_VELOCITY
        # Unweighted mean would be ~1.0005, weighted should be ~1.0000
        for i in range(d.REST_AVG_FRAMES):
            if i % 2 == 0:
                d.update(t, 1.001, stillness_weight=0.01)
            else:
                d.update(t, 1.000, stillness_weight=1.0)
            t += dt

        assert d._rest_height is not None
        # Weighted avg pulled toward 1.000 (high weight), not 1.0005 (unweighted)
        assert d._rest_height < 1.0003

    def test_zero_weight_frames_excluded(self):
        """Frames with zero weight should not be collected."""
        d = self.make_detector()
        dt = 1.0 / 60

        # First blow
        d.update(0.0, 1.0)
        d.update(2.5, 2.0)
        d.update(3.0, 2.0)
        d.update(3.05, 0.5)

        # Settle
        t = 3.1
        for _ in range(15):
            d.update(t, 1.0)
            t += dt

        # All zero weight → nothing collected
        samples_before = len(d._rest_samples)
        for _ in range(10):
            d.update(t, 1.0, stillness_weight=0.0)
            t += dt

        assert len(d._rest_samples) == samples_before

    def test_default_weight_is_one(self):
        """Default stillness_weight=1.0 for backward compatibility."""
        d = self.make_detector()
        dt = 1.0 / 60

        # First blow
        d.update(0.0, 1.0)
        d.update(2.5, 2.0)
        d.update(3.0, 2.0)
        d.update(3.05, 0.5)

        # Rest — no weight arg → defaults to 1.0, all frames collected
        t = 3.1
        for _ in range(50):
            d.update(t, 1.0)
            t += dt

        assert d._awaiting_rest is False


# ============================================================
# IMUCompensator Tests
# ============================================================

class TestIMUHelper:

    @staticmethod
    def pitch_to_quat(pitch_rad):
        """Convert a pitch angle (rotation around X) to quaternion (i,j,k,real)."""
        half = pitch_rad / 2.0
        return np.sin(half), 0.0, 0.0, np.cos(half)

    def make_imu_packet(self, samples):
        """Create a mock BNO085 fused IMU packet.
        Each sample is (pitch_rad, lin_accel_x, lin_accel_y, lin_accel_z).
        """
        packet = MagicMock()
        imu_datas = []
        for pitch_rad, lax, lay, laz in samples:
            imu_data = MagicMock()
            qi, qj, qk, qr = self.pitch_to_quat(pitch_rad)
            imu_data.rotationVector.i = qi
            imu_data.rotationVector.j = qj
            imu_data.rotationVector.k = qk
            imu_data.rotationVector.real = qr
            imu_data.linearAcceleration.x = lax
            imu_data.linearAcceleration.y = lay
            imu_data.linearAcceleration.z = laz
            imu_datas.append(imu_data)
        packet.packets = imu_datas
        return packet

    def test_initial_state(self):
        h = IMUHelper()
        assert h.is_ready is False
        assert h.pitch == 0.0
        assert h.baseline_pitch == 0.0

    def test_calibration_phase(self):
        """Helper should not be ready until CALIB_COUNT samples collected."""
        h = IMUHelper()
        h.CALIB_COUNT = 10

        # 9 samples — not enough
        samples = [(0.0, 0, 0, 0)] * 9
        h.add_imu_packet(self.make_imu_packet(samples))
        h.get_tilt_correction(np.array([0, 0, 5.0]))
        assert h.is_ready is False

        # 1 more → calibrated
        h.add_imu_packet(self.make_imu_packet([(0.0, 0, 0, 0)]))
        h.get_tilt_correction(np.array([0, 0, 5.0]))
        assert h.is_ready is True

    def test_baseline_pitch_estimation(self):
        """Calibration should average pitch as baseline."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        # Stationary at 1° pitch
        pitch = np.radians(1.0)
        samples = [(pitch, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(samples))
        h.get_tilt_correction(np.array([0, 0, 5.0]))

        assert h.is_ready is True
        assert abs(h.baseline_pitch - pitch) < 0.001

    def test_tilt_correction_at_known_angle(self):
        """0.5° tilt from baseline at 5m depth → ~44mm correction."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        # Calibrate at 0° pitch
        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h.get_tilt_correction(np.array([0, 0, 5.0]))

        # Now tilted 0.5°
        target_rad = np.radians(0.5)
        tilted = [(target_rad, 0, 0, 0)] * 3
        h.add_imu_packet(self.make_imu_packet(tilted))

        tvec = np.array([[0.0], [-1.0], [5.0]])
        correction = h.get_tilt_correction(tvec)

        # Expected: 5.0 * tan(0.5°) ≈ 0.0436m ≈ 43.6mm
        expected = 5.0 * np.tan(target_rad)
        assert abs(correction - expected) < 0.001

    def test_zero_tilt_when_stationary(self):
        """Same pitch as baseline → zero correction."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        samples = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(samples))
        h.get_tilt_correction(np.array([0, 0, 5.0]))

        # Still at same pitch
        post = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(post))

        correction = h.get_tilt_correction(np.array([[0], [-1], [5.0]]))
        assert abs(correction) < 0.001

    def test_frame_still_when_no_linear_accel(self):
        """Zero linear acceleration → frame is still."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h.is_frame_still()

        post = [(0.0, 0, 0, 0)] * 3
        h.add_imu_packet(self.make_imu_packet(post))

        assert h.is_frame_still() is True

    def test_frame_not_still_high_linear_accel(self):
        """High linear acceleration → frame is not still."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h.is_frame_still()

        # 2 m/s² linear accel (gravity already removed by BNO085)
        noisy = [(0.0, 2.0, 0, 0)] * 3
        h.add_imu_packet(self.make_imu_packet(noisy))

        assert h.is_frame_still() is False

    def test_reset_tilt_rebaselines(self):
        """reset_tilt should set current pitch as new baseline."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        # Calibrate at 0°
        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h.get_tilt_correction(np.array([0, 0, 5.0]))

        # Tilt to 1°
        tilted = [(np.radians(1.0), 0, 0, 0)] * 3
        h.add_imu_packet(self.make_imu_packet(tilted))
        corr_before = h.get_tilt_correction(np.array([0, 0, 5.0]))
        assert abs(corr_before) > 0.01  # nonzero correction

        # Reset → current pitch becomes baseline
        h.reset_tilt()
        corr_after = h.get_tilt_correction(np.array([0, 0, 5.0]))
        assert abs(corr_after) < 0.001  # zero after reset

    def test_stillness_weight_stationary(self):
        """Zero linear accel → weight close to 1.0."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h.get_stillness_weight()

        post = [(0.0, 0, 0, 0)] * 3
        h.add_imu_packet(self.make_imu_packet(post))

        w = h.get_stillness_weight()
        assert w > 0.9

    def test_stillness_weight_shaking(self):
        """High linear accel → weight close to 0."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h.get_stillness_weight()

        # Heavy vibration: 3 m/s² in multiple axes
        noisy = [(0.0, 3.0, 2.0, 1.0)] * 3
        h.add_imu_packet(self.make_imu_packet(noisy))

        w = h.get_stillness_weight()
        assert w < 0.1

    def test_tilt_scales_with_depth(self):
        """Same tilt at different depths → proportional correction."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h.get_tilt_correction(np.array([0, 0, 5.0]))

        tilted = [(np.radians(0.5), 0, 0, 0)] * 3
        h.add_imu_packet(self.make_imu_packet(tilted))

        corr_5m = h.get_tilt_correction(np.array([0, 0, 5.0]))
        corr_10m = h.get_tilt_correction(np.array([0, 0, 10.0]))

        assert abs(corr_10m / corr_5m - 2.0) < 0.01

    def make_zupt_packet(self, samples):
        """Create mock packet for ZUPT testing.
        Each sample is (pitch_rad, lax, lay, laz, timestamp_s).
        Includes timestamp on linearAcceleration for ZUPT integration.
        """
        packet = MagicMock()
        imu_datas = []
        for pitch_rad, lax, lay, laz, ts in samples:
            imu_data = MagicMock()
            qi, qj, qk, qr = self.pitch_to_quat(pitch_rad)
            imu_data.rotationVector.i = qi
            imu_data.rotationVector.j = qj
            imu_data.rotationVector.k = qk
            imu_data.rotationVector.real = qr
            imu_data.linearAcceleration.x = lax
            imu_data.linearAcceleration.y = lay
            imu_data.linearAcceleration.z = laz
            td = MagicMock()
            td.total_seconds.return_value = ts
            imu_data.linearAcceleration.getTimestampDevice.return_value = td
            imu_datas.append(imu_data)
        packet.packets = imu_datas
        return packet

    def test_zupt_no_movement(self):
        """Zero accel throughout → zero displacement."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h._process_pending()

        h.start_zupt()
        # Feed zero accel with timestamps
        samples = [(0.0, 0, 0, 0, 0.01 + 0.0025 * i) for i in range(40)]
        h.add_imu_packet(self.make_zupt_packet(samples))

        disp = h.finish_zupt()
        assert abs(disp) < 0.0001  # essentially zero

    def test_zupt_known_displacement(self):
        """Impulse that should produce known displacement with ZUPT correction."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h._process_pending()

        h.start_zupt()
        # Simulate: accelerate down then decelerate (symmetric impulse)
        # Camera drops and stops. Net displacement but final v=0.
        # Accel: +2 m/s² for 0.1s, then -2 m/s² for 0.1s
        # Expected displacement: 2 * 0.5 * 2 * 0.1^2 = 0.02m = 20mm
        dt = 0.0025
        samples = []
        t = 0.01
        for i in range(40):  # 0.1s at 2 m/s²
            samples.append((0.0, 0, 2.0, 0, t))
            t += dt
        for i in range(40):  # 0.1s at -2 m/s²
            samples.append((0.0, 0, -2.0, 0, t))
            t += dt
        h.add_imu_packet(self.make_zupt_packet(samples))

        disp = h.finish_zupt()
        # Symmetric impulse → displacement = a * T^2 / 2 where T=0.1
        # = 2 * 0.1^2 / 2 = 0.01m per half, but symmetric so net ≈ 0.02m
        # ZUPT corrects drift; the displacement from symmetric accel/decel
        # is 0.5 * a * t^2 for accel phase = 0.5*2*0.01 = 0.01m
        assert abs(disp) > 0.005  # nonzero displacement
        assert abs(disp) < 0.05   # reasonable magnitude

    def test_zupt_removes_constant_bias(self):
        """Constant bias should be removed by ZUPT correction."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h._process_pending()

        h.start_zupt()
        # Pure bias: constant 0.05 m/s² (no real movement)
        # Without ZUPT: would accumulate displacement
        # With ZUPT: final v must be 0, so bias is removed → displacement ≈ 0
        dt = 0.0025
        samples = [(0.0, 0, 0.05, 0, 0.01 + dt * i) for i in range(80)]
        h.add_imu_packet(self.make_zupt_packet(samples))

        disp = h.finish_zupt()
        assert abs(disp) < 0.001  # bias removed, near zero

    def test_zupt_accumulates_across_blows(self):
        """Displacement accumulates across multiple start/finish cycles."""
        h = IMUHelper()
        h.CALIB_COUNT = 5

        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h._process_pending()

        # Two blows, each with symmetric impulse
        for blow_i in range(2):
            h.start_zupt()
            dt = 0.0025
            t_base = 0.01 + blow_i * 0.5
            samples = []
            t = t_base
            for i in range(40):
                samples.append((0.0, 0, 2.0, 0, t))
                t += dt
            for i in range(40):
                samples.append((0.0, 0, -2.0, 0, t))
                t += dt
            h.add_imu_packet(self.make_zupt_packet(samples))
            h.finish_zupt()

        # Should have accumulated displacement from both blows
        total = h.get_displacement_y()
        assert abs(total) > 0.01  # nonzero accumulated

    def test_zupt_inactive_returns_zero(self):
        """finish_zupt without start returns zero."""
        h = IMUHelper()
        h.CALIB_COUNT = 5
        calib = [(0.0, 0, 0, 0)] * 5
        h.add_imu_packet(self.make_imu_packet(calib))
        h._process_pending()

        assert h.finish_zupt() == 0.0
        assert h.get_displacement_y() == 0.0

    def test_uncalibrated_returns_permissive(self):
        """Before calibration, tilt correction = 0 and is_frame_still = True."""
        h = IMUHelper()
        assert h.get_tilt_correction(np.array([0, 0, 5.0])) == 0.0
        assert h.is_frame_still() is True
        assert h.get_stillness_weight() == 1.0


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


# ============================================================
# MadgwickFilter Tests
# ============================================================

class TestMadgwickFilter:

    def test_initial_quaternion_is_identity(self):
        m = MadgwickFilter()
        np.testing.assert_array_almost_equal(m.q, [1, 0, 0, 0])

    def test_converges_to_gravity_when_stationary(self):
        """With zero gyro and accel = [0, 0, 9.81], pitch should stay ~0."""
        m = MadgwickFilter(beta=0.1)
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        for _ in range(500):
            m.update(gyro, accel, 0.005)
        assert abs(m.get_pitch()) < 0.02  # near zero pitch

    def test_pitch_from_tilted_accel(self):
        """Accel tilted ~30deg around X → pitch converges near 30deg."""
        m = MadgwickFilter(beta=0.5)  # high beta for fast convergence
        pitch_rad = np.radians(30)
        # Gravity rotated by pitch around X axis
        accel = np.array([0.0, 9.81 * np.sin(pitch_rad), 9.81 * np.cos(pitch_rad)])
        gyro = np.array([0.0, 0.0, 0.0])
        for _ in range(1000):
            m.update(gyro, accel, 0.005)
        assert abs(m.get_pitch() - pitch_rad) < 0.05

    def test_remove_gravity_when_stationary(self):
        """Stationary sensor: remove_gravity should yield ~zero."""
        m = MadgwickFilter(beta=0.5)
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        for _ in range(500):
            m.update(gyro, accel, 0.005)
        lin = m.remove_gravity(accel)
        assert np.linalg.norm(lin) < 0.1  # near zero

    def test_remove_gravity_with_motion(self):
        """Stationary filter + extra 2m/s^2 on X → linear accel ~2 on X."""
        m = MadgwickFilter(beta=0.5)
        accel_still = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        for _ in range(500):
            m.update(gyro, accel_still, 0.005)
        accel_moving = np.array([2.0, 0.0, 9.81])
        lin = m.remove_gravity(accel_moving)
        assert abs(lin[0] - 2.0) < 0.2
        assert abs(lin[2]) < 0.2

    def test_gravity_vector_magnitude(self):
        """Gravity vector should have magnitude ~9.81."""
        m = MadgwickFilter()
        g = m.get_gravity_vector()
        assert abs(np.linalg.norm(g) - 9.81) < 0.01

    def test_zero_dt_is_noop(self):
        """dt=0 should not change state."""
        m = MadgwickFilter()
        q_before = m.q.copy()
        m.update(np.zeros(3), np.array([0, 0, 9.81]), 0.0)
        np.testing.assert_array_equal(m.q, q_before)

    def test_process_sample_integration(self):
        """MadgwickFilter output feeds IMUHelper.process_sample correctly."""
        m = MadgwickFilter(beta=0.5)
        h = IMUHelper()
        h.CALIB_COUNT = 5

        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])

        for i in range(10):
            m.update(gyro, accel, 0.005)
            pitch = m.get_pitch()
            lin = m.remove_gravity(accel)
            lin_mag = float(np.linalg.norm(lin))
            h.process_sample(pitch, lin_mag, lin[1], 0.01 + i * 0.005)

        assert h.is_ready is True
        assert abs(h.baseline_pitch) < 0.05
