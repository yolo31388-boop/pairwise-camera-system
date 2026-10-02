import pytest
from camera.system import CameraSystem


class TestCameraSystem:

    def test_follow_target(self):
        """镜头跟随只用简单插值，不做预测"""
        obj = CameraSystem()
        result = obj.follow_target()
        self.assertIsNotNone(result)

    def test_resolve_camera_collision(self):
        """镜头碰撞不做检测，相机穿墙看到地图外面"""
        obj = CameraSystem()
        result = obj.resolve_camera_collision()
        self.assertIsNotNone(result)

    def test_shake_camera(self):
        """镜头震动只用随机数，不做波形和衰减"""
        obj = CameraSystem()
        result = obj.shake_camera()
        self.assertIsNotNone(result)

    def test_switch_camera(self):
        """镜头切换只做位置插值，不做焦点和FOV过渡"""
        obj = CameraSystem()
        result = obj.switch_camera()
        self.assertIsNotNone(result)

    def test_apply_constraints(self):
        """镜头约束不做角度限制，相机可以钻到地下"""
        obj = CameraSystem()
        result = obj.apply_constraints()
        self.assertIsNotNone(result)

