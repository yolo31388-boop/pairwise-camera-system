"""
pairwise-camera-system - CameraSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class CameraSystem:
    def __init__(self):
        self.state = {}

    def follow_target(self, *args, **kwargs):
        """BUG: 镜头跟随只用简单插值，不做预测"""
        return None

    def resolve_camera_collision(self, *args, **kwargs):
        """BUG: 镜头碰撞不做检测，相机穿墙看到地图外面"""
        return None

    def shake_camera(self, *args, **kwargs):
        """BUG: 镜头震动只用随机数，不做波形和衰减"""
        return None

    def switch_camera(self, *args, **kwargs):
        """BUG: 镜头切换只做位置插值，不做焦点和FOV过渡"""
        return None

    def apply_constraints(self, *args, **kwargs):
        """BUG: 镜头约束不做角度限制，相机可以钻到地下"""
        return None

