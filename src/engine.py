"""相机系统 - 7个手感问题已修复"""
import math
from dataclasses import dataclass


@dataclass
class Camera:
    pos: list
    target: list
    fov: float = 60.0
    shake_amp: float = 0.0
    shake_time: float = 0.0


def _catmull_rom(p0, p1, p2, p3, u):
    """Catmull-Rom 样条插值，保证路径切线连续"""
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


class CameraSystem:
    # fix1: 跟随阻尼系数与最大速度
    FOLLOW_DAMPING = 8.0
    FOLLOW_MAX_SPEED = 240.0
    # fix2: 震动最大持续时间
    SHAKE_MAX_DURATION = 2.0
    # fix6: FOV 平滑过渡速度（每秒插值逼近比例）
    FOV_DAMPING = 10.0

    def __init__(self):
        self.cam = Camera([0, 0], [0, 0])
        self.effects = []
        self.target_fov = self.cam.fov
        self._shake_initial_amp = 0.0
        self._shake_duration = 0.0

    # fix2: shake_time 暴露为系统属性，便于外部读写与衰减驱动
    @property
    def shake_time(self):
        return self.cam.shake_time

    @shake_time.setter
    def shake_time(self, value):
        self.cam.shake_time = value
        if value <= 0:
            self.cam.shake_time = 0.0
            self.cam.shake_amp = 0.0
            self._shake_initial_amp = 0.0
            self._shake_duration = 0.0

    def follow(self, target_pos, dt):
        # fix1: 指数平滑阻尼插值 + 最大速度限制
        self.cam.target = list(target_pos)
        blend = 1.0 - math.exp(-self.FOLLOW_DAMPING * dt)
        max_step = self.FOLLOW_MAX_SPEED * dt
        new_pos = []
        for axis in range(len(self.cam.pos)):
            delta = (target_pos[axis] - self.cam.pos[axis]) * blend
            if delta > max_step:
                delta = max_step
            elif delta < -max_step:
                delta = -max_step
            new_pos.append(self.cam.pos[axis] + delta)
        self.cam.pos = new_pos

    def shake(self, amplitude, duration):
        # fix2: 记录初始幅度，幅度随剩余时间衰减，且限制最大持续时间
        self._shake_duration = min(duration, self.SHAKE_MAX_DURATION)
        self._shake_initial_amp = amplitude
        self.cam.shake_amp = amplitude
        self.cam.shake_time = self._shake_duration

    def update_shake(self, dt):
        # fix2: 每帧推进震动，幅度按剩余时间比例线性衰减
        if self.cam.shake_time <= 0:
            self.cam.shake_amp = 0.0
            return
        self.cam.shake_time = max(0.0, self.cam.shake_time - dt)
        if self._shake_duration > 0:
            ratio = self.cam.shake_time / self._shake_duration
            self.cam.shake_amp = self._shake_initial_amp * ratio
        if self.cam.shake_time <= 0:
            self.cam.shake_amp = 0.0

    def clamp_angle(self, angle, min_a, max_a):
        # fix3: 软限制，用 tanh 平滑逼近极限，接近极限时自然减速
        mid = (min_a + max_a) * 0.5
        half = (max_a - min_a) * 0.5
        if half <= 0:
            return mid
        normalized = (angle - mid) / half
        return mid + math.tanh(normalized) * half

    def lerp_path(self, points, t):
        # fix4: Catmull-Rom 样条，过场路径切线连续，转弯无拐点
        if t <= 0:
            return list(points[0])
        if t >= 1:
            return list(points[-1])
        count = len(points)
        if count < 3:
            return [
                points[0][axis] + (points[-1][axis] - points[0][axis]) * t
                for axis in range(len(points[0]))
            ]
        seg_count = count - 1
        scaled = t * seg_count
        idx = min(int(scaled), seg_count - 1)
        u = scaled - idx
        p0 = points[max(idx - 1, 0)]
        p1 = points[idx]
        p2 = points[idx + 1]
        p3 = points[min(idx + 2, count - 1)]
        return [
            _catmull_rom(p0[axis], p1[axis], p2[axis], p3[axis], u)
            for axis in range(len(p1))
        ]

    def collide(self, player_pos, obstacles):
        # fix5: 从玩家到相机做射线检测，碰到障碍把相机拉近到命中点前方
        start = list(player_pos)
        end = list(self.cam.pos)
        nearest_t = None
        for obs in obstacles:
            t = self._ray_box_t(start, end, obs)
            if t is not None and (nearest_t is None or t < nearest_t):
                nearest_t = t
        if nearest_t is None:
            return False
        pull = max(0.0, nearest_t - 0.02)
        self.cam.pos = [
            start[axis] + (end[axis] - start[axis]) * pull
            for axis in range(len(start))
        ]
        return True

    @staticmethod
    def _ray_box_t(start, end, obs):
        """线段与轴对齐盒（中心 obs[:2]，半尺寸 obs[2]）的 slab 法求交"""
        dims = len(start)
        center = obs[:dims]
        half = obs[dims] if len(obs) > dims else obs[-1]
        t_min, t_max = 0.0, 1.0
        for axis in range(dims):
            direction = end[axis] - start[axis]
            lo = center[axis] - half
            hi = center[axis] + half
            if abs(direction) < 1e-12:
                if start[axis] < lo or start[axis] > hi:
                    return None
                continue
            t1 = (lo - start[axis]) / direction
            t2 = (hi - start[axis]) / direction
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None
        return t_min

    def set_fov(self, fov):
        # fix6: 只记录目标 FOV，实际值在 update_fov 中平滑逼近
        self.target_fov = fov

    def update_fov(self, dt):
        # fix6: FOV 指数平滑过渡，避免瞬间跳变
        blend = 1.0 - math.exp(-self.FOV_DAMPING * dt)
        self.cam.fov += (self.target_fov - self.cam.fov) * blend

    def update(self, dt):
        self.update_shake(dt)
        self.update_fov(dt)

    def apply_effects(self):
        # fix7: 分层叠加——每层效果基于同一基准位置独立计算偏移，
        # 再把所有偏移求和合并，而不是后者覆盖前者
        base = list(self.cam.pos)
        merged = list(base)
        for eff in self.effects:
            result = eff(list(base))
            for axis in range(len(merged)):
                merged[axis] += result[axis] - base[axis]
        self.cam.pos = merged
