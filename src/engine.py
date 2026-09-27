"""相机系统入口。

实现已迁移到 camera.system，此处保持向后兼容的再导出。
"""
from camera.system import Camera, CameraSystem, EffectResult

__all__ = ["Camera", "CameraSystem", "EffectResult"]
