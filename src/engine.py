"""相机系统 - 带7个bug"""
from dataclasses import dataclass

@dataclass
class Camera:
    pos: list
    target: list
    fov: float = 60.0
    shake_amp: float = 0.0
    shake_time: float = 0.0

class CameraSystem:
    def __init__(self):
        self.cam = Camera([0,0], [0,0])
        self.effects = []

    def follow(self, target_pos, dt):
        # bug1: 直接设置位置
        self.cam.pos = list(target_pos)

    def shake(self, amplitude, duration):
        # bug2: 无衰减
        self.cam.shake_amp = amplitude
        self.cam.shake_time = duration

    def clamp_angle(self, angle, min_a, max_a):
        # bug3: 硬钳制
        return max(min_a, min(max_a, angle))

    def lerp_path(self, points, t):
        # bug4: 线性插值
        if t <= 0: return points[0]
        if t >= 1: return points[-1]
        idx = int(t * (len(points)-1))
        return points[idx]

    def collide(self, player_pos, obstacles):
        # bug5: 只检查相机位置
        for obs in obstacles:
            if abs(self.cam.pos[0]-obs[0]) < obs[2] and abs(self.cam.pos[1]-obs[1]) < obs[2]:
                return True
        return False

    def apply_effects(self):
        # bug7: 效果互相覆盖
        for eff in self.effects:
            self.cam.pos = eff(self.cam.pos)
