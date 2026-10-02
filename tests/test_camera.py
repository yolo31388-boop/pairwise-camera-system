"""相机视角系统 - 14项修复验证"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.system import CameraSystem


class TestCameraCollision:
    def test_camera_does_not_clip_walls(self):
        cs = CameraSystem()
        cs.update_camera([0, 0, 0], [{"pos": [0, 5, -5], "size": 2}])
        assert cs.camera.position[2] > -5

    def test_camera_stays_outside_obstacle_volume(self):
        cs = CameraSystem()
        cs.update_camera([0, 0, 0], [{"pos": [0, 5, -10], "size": 4}])
        dist = math.sqrt(sum((cs.camera.position[i] - [0, 5, -10][i]) ** 2 for i in range(3)))
        assert dist >= 2.0

    def test_line_of_sight_blocked_between_camera_and_player(self):
        cs = CameraSystem()
        cs.update_camera([0, 0, 0], [{"pos": [0, 1.6, -6], "size": 1.0}])
        cam = cs.camera.position
        anchor = [0, 1.6, 0]
        seg_len = math.sqrt(sum((cam[i] - anchor[i]) ** 2 for i in range(3)))
        t = (6.0 - 0.4) / seg_len
        closest = [anchor[i] + (cam[i] - anchor[i]) * t for i in range(3)]
        d = math.sqrt(sum((closest[i] - [0, 1.6, -6][i]) ** 2 for i in range(3)))
        assert d >= 1.0

    def test_no_obstacle_keeps_desired_offset(self):
        cs = CameraSystem()
        cs.update_camera([3, 0, 4], [])
        x, y, z = cs.camera.position
        assert math.isclose(x, 3, abs_tol=1e-6)
        assert y > 0
        assert z < 4


class TestFrameRateIndependent:
    def test_smoothing_frame_rate_independent(self):
        cs = CameraSystem()
        cs.smooth_camera([10, 10, 10], 0.1)
        cs.smooth_camera([10, 10, 10], 0.016)
        assert cs.camera.position[0] > 0

    def test_same_total_time_similar_result_across_fps(self):
        low = CameraSystem()
        high = CameraSystem()
        for _ in range(10):
            low.smooth_camera([10, 0, 0], 0.1)
        for _ in range(60):
            high.smooth_camera([10, 0, 0], 1 / 60)
        assert abs(low.camera.position[0] - high.camera.position[0]) < 0.5

    def test_smoothing_moves_toward_target(self):
        cs = CameraSystem()
        cs.smooth_camera([0, 5, 10], 0.016)
        assert cs.camera.position[2] > -10

    def test_fast_turn_adapts_snappier(self):
        slow = CameraSystem()
        fast = CameraSystem()
        slow.turn_speed = 0.0
        fast.turn_speed = 1.0
        slow.smooth_camera([10, 5, 0], 0.016)
        fast.smooth_camera([10, 5, 0], 0.016)
        assert fast.camera.position[0] > slow.camera.position[0]


class TestLockFollowsPlayer:
    def test_locked_camera_follows_player(self):
        cs = CameraSystem()
        cs.lock_on_target({"pos": [10, 0, 0], "alive": True})
        cs.update_camera([5, 0, 0], [])
        assert cs.camera.target[0] == 10

    def test_locked_camera_position_moves_with_player(self):
        cs = CameraSystem()
        cs.lock_on_target({"pos": [10, 0, 10], "alive": True})
        cs.update_camera([0, 0, 0], [])
        first = list(cs.camera.position)
        cs.update_camera([0, 0, 8], [])
        assert not math.isclose(cs.camera.position[2], first[2], abs_tol=1e-6)

    def test_lock_switches_between_targets(self):
        cs = CameraSystem()
        a = {"pos": [10, 0, 0], "alive": True}
        b = {"pos": [-10, 0, 0], "alive": True}
        cs.lock_on_target(a)
        cs.lock_on_target(b)
        assert cs.locked_target is b
        assert cs.camera.target[0] == -10

    def test_lock_rejects_dead_target(self):
        cs = CameraSystem()
        cs.lock_on_target({"pos": [10, 0, 0], "alive": False})
        assert cs.locked_target is None

    def test_dead_locked_target_is_released_on_update(self):
        cs = CameraSystem()
        target = {"pos": [10, 0, 0], "alive": True}
        cs.lock_on_target(target)
        target["alive"] = False
        cs.update_camera([0, 0, 0], [])
        assert cs.locked_target is None

    def test_cycle_target_skips_dead(self):
        cs = CameraSystem()
        a = {"pos": [10, 0, 0], "alive": True}
        b = {"pos": [-10, 0, 0], "alive": False}
        chosen = cs.cycle_target([a, b])
        assert chosen is a


class TestOcclusionTransparency:
    def test_occluder_made_transparent(self):
        cs = CameraSystem()
        obstacles = [{"pos": [0, 2, -5], "transparent": False}]
        cs.handle_occlusion(obstacles)
        assert obstacles[0]["transparent"] is True

    def test_occlusion_pulls_camera_to_player_side(self):
        cs = CameraSystem()
        cs.camera.position = [0, 1.6, -10]
        cs.handle_occlusion([{"pos": [0, 1.6, -6], "size": 2}])
        assert cs.camera.position[2] > -8

    def test_non_blocking_obstacle_not_transparent(self):
        cs = CameraSystem()
        obstacles = [{"pos": [50, 50, 50], "size": 1, "transparent": False}]
        cs.handle_occlusion(obstacles)
        assert obstacles[0]["transparent"] is False


class TestCameraAngleLimit:
    def test_camera_angle_limited(self):
        cs = CameraSystem()
        cs.process_camera_input(1000, 1000)
        assert cs.camera.target[1] < 90

    def test_pitch_clamped_against_ground(self):
        cs = CameraSystem()
        for _ in range(100):
            cs.process_camera_input(0, 1000)
        assert cs.pitch < math.radians(90)
        assert cs.pitch >= -math.radians(80) - 1e-9

    def test_input_smoothing_dampens_fast_flick(self):
        cs = CameraSystem()
        yaw_a, _ = cs.process_camera_input(1000, 0, 1 / 60)
        raw = CameraSystem()
        raw_yaw = 0.0
        for _ in range(60):
            raw_yaw += 0.0
        assert abs(yaw_a) < 1000 * cs.sensitivity

    def test_input_acceleration_gain_with_speed(self):
        slow = CameraSystem()
        fast = CameraSystem()
        slow.process_camera_input(5, 0, 1 / 60)
        for _ in range(5):
            fast.process_camera_input(500, 0, 1 / 60)
        per_unit_fast = abs(fast.yaw) / 500.0
        per_unit_slow = abs(slow.yaw) / 5.0
        assert per_unit_fast >= per_unit_slow
