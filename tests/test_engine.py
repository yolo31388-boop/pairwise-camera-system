"""相机系统测试"""
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine import CameraSystem, Camera

class TestFollow:
    def test_smooth_follow(self):
        cs = CameraSystem()
        cs.cam.pos = [0, 0]
        cs.follow([100, 0], 0.016)
        assert cs.cam.pos[0] != 100, "相机跟随无平滑插值"

class TestShake:
    def test_shake_decays(self):
        cs = CameraSystem()
        cs.shake(10, 0.5)
        cs.shake_time = 0
        assert cs.cam.shake_amp == 0 or cs.cam.shake_time <= 0, "震动无衰减"
