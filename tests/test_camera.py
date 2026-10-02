"""相机视角系统 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.system import CameraSystem

class TestCameraCollision:
    def test_camera_does_not_clip_walls(self):
        cs = CameraSystem()
        cs.update_camera([0,0,0], [{"pos":[0,5,-5],"size":2}])
        assert cs.camera.position[2] > -5

class TestFrameRateIndependent:
    def test_smoothing_frame_rate_independent(self):
        cs = CameraSystem()
        cs.smooth_camera([10, 10, 10], 0.1)
        cs.smooth_camera([10, 10, 10], 0.016)
        assert cs.camera.position[0] > 0

class TestLockFollowsPlayer:
    def test_locked_camera_follows_player(self):
        cs = CameraSystem()
        cs.lock_on_target({"pos": [10,0,0], "alive": True})
        cs.update_camera([5,0,0], [])
        assert cs.camera.target[0] == 10

class TestOcclusionTransparency:
    def test_occluder_made_transparent(self):
        cs = CameraSystem()
        obstacles = [{"pos":[0,2,-5],"transparent":False}]
        cs.handle_occlusion(obstacles)
        assert obstacles[0]["transparent"] == True

class TestCameraAngleLimit:
    def test_camera_angle_limited(self):
        cs = CameraSystem()
        cs.process_camera_input(1000, 1000)
        assert cs.camera.target[1] < 90
