"""相机系统核心。

七个手感问题的修复:

1. follow       —— 阻尼插值（指数平滑），并限制最大跟随速度。
2. shake        —— 幅度随时间线性衰减，且持续时间不允许超过最大值。
3. clamp_angle  —— 软限制：进入边缘软区后逐步减速、平滑贴到极限，
                   而不是到达极限瞬间硬卡住。
4. catmull_rom  —— Catmull-Rom 样条插值，控制点处切线连续（C1）。
5. collide      —— 从玩家到相机做线段相交检测，命中障碍时把相机
                   沿连线拉近到障碍表面。
6. zoom / set_fov —— FOV 以指数平滑向目标值过渡，不再瞬间跳变。
7. apply_effects —— 分层叠加：每层效果只产生自己的偏移/增量，
                   以基础姿态为输入独立计算，最后合并（相加），
                   不再互相覆盖。

纯标准库实现。
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

Vector = List[float]


def _vadd(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def _vsub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x - y for x, y in zip(a, b)]


def _vscale(a: Sequence[float], s: float) -> Vector:
    return [x * s for x in a]


def _vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Camera:
    pos: list
    target: list
    fov: float = 60.0
    target_fov: float = 60.0
    shake_amp: float = 0.0
    shake_base: float = 0.0
    shake_time: float = 0.0
    shake_total: float = 0.0
    shake_seed: int = 0


@dataclass
class EffectResult:
    """单层效果的独立计算结果，全部为增量，与应用顺序无关。"""

    pos_offset: list = field(default_factory=lambda: [0.0, 0.0])
    target_offset: list = field(default_factory=lambda: [0.0, 0.0])
    fov_delta: float = 0.0

    def merge(self, other: "EffectResult") -> "EffectResult":
        return EffectResult(
            pos_offset=_vadd(self.pos_offset, other.pos_offset),
            target_offset=_vadd(self.target_offset, other.target_offset),
            fov_delta=self.fov_delta + other.fov_delta,
        )


# 障碍表示为 (cx, cy, radius)
Obstacle = Tuple[float, float, float]


class CameraSystem:
    def __init__(
        self,
        damping: float = 10.0,
        max_follow_speed: float = 800.0,
        max_shake_duration: float = 1.0,
        soft_zone: float = 0.1,
        fov_lerp_speed: float = 8.0,
        fov_min: float = 30.0,
        fov_max: float = 90.0,
        rng: random.Random = None,
    ):
        self.cam = Camera([0.0, 0.0], [0.0, 0.0])
        # 效果按层注册，每层独立计算后叠加
        self.effects: List[Callable[[Camera], EffectResult]] = []
        self.damping = damping
        self.max_follow_speed = max_follow_speed
        self.max_shake_duration = max_shake_duration
        self.soft_zone = soft_zone
        self.fov_lerp_speed = fov_lerp_speed
        self.fov_min = fov_min
        self.fov_max = fov_max
        self.rng = rng or random.Random(0)

    # ------------------------------------------------------------------
    # 1. 平滑跟随：指数阻尼 + 最大速度限制
    # ------------------------------------------------------------------
    def follow(self, target_pos, dt: float) -> list:
        target_pos = [float(v) for v in target_pos]
        self.cam.target = list(target_pos)
        if dt <= 0:
            return list(self.cam.pos)
        # 指数平滑：alpha 越小越“黏”
        alpha = 1.0 - math.exp(-self.damping * dt)
        delta = _vsub(target_pos, self.cam.pos)
        step = _vscale(delta, alpha)
        max_step = self.max_follow_speed * dt
        dist = math.hypot(*step)
        if dist > max_step and dist > 0.0:
            step = _vscale(step, max_step / dist)
        self.cam.pos = _vadd(self.cam.pos, step)
        return list(self.cam.pos)

    # ------------------------------------------------------------------
    # 2. 震动：幅度随时间衰减 + 最大持续时间
    # ------------------------------------------------------------------
    def shake(self, amplitude: float, duration: float):
        duration = max(0.0, min(duration, self.max_shake_duration))
        self.cam.shake_base = max(0.0, amplitude)
        self.cam.shake_total = duration
        self.cam.shake_time = duration
        if amplitude > 0.0:
            self.cam.shake_seed = self.rng.randrange(1 << 30)

    def update_shake(self, dt: float) -> float:
        if self.cam.shake_time <= 0.0 or self.cam.shake_total <= 0.0:
            self.cam.shake_amp = 0.0
            self.cam.shake_time = 0.0
            self.cam.shake_base = 0.0
            return 0.0
        self.cam.shake_time = max(0.0, self.cam.shake_time - dt)
        ratio = self.cam.shake_time / self.cam.shake_total
        self.cam.shake_amp = self.cam.shake_base * ratio
        if self.cam.shake_time <= 0.0:
            self.cam.shake_amp = 0.0
            self.cam.shake_base = 0.0
        return self.cam.shake_amp

    def shake_offset(self) -> list:
        amp = self.cam.shake_amp
        if amp <= 0.0:
            return [0.0, 0.0]
        phase = self.cam.shake_total - self.cam.shake_time
        ox = amp * math.sin(phase * 62.0 + self.cam.shake_seed)
        oy = amp * math.cos(phase * 77.0 + self.cam.shake_seed)
        return [ox, oy]

    # ------------------------------------------------------------------
    # 3. 视角软限制：软区内用缓出函数减速平滑到极限
    # ------------------------------------------------------------------
    def clamp_angle(self, angle: float, min_a: float, max_a: float) -> float:
        if angle < min_a:
            return min_a
        if angle > max_a:
            return max_a
        zone = max(0.0, self.soft_zone)
        if zone <= 0.0:
            return max(min_a, min(max_a, angle))
        # 软区内使用三次 Hermite 曲线 g(x)=x+x^2-x^3：
        # g(0)=0,g'(0)=1（软区入口处与硬区斜率一致），
        # g(1)=1,g'(1)=0（极限处速度平滑降为 0，不硬卡）。
        if angle > max_a - zone:
            local = (angle - (max_a - zone)) / zone  # 0 -> 1
            eased = local + local * local - local ** 3
            return (max_a - zone) + zone * eased
        if angle < min_a + zone:
            local = (angle - min_a) / zone
            eased = local + local * local - local ** 3
            return min_a + zone * eased
        return angle

    # ------------------------------------------------------------------
    # 4. 过场路径：Catmull-Rom 样条（切线连续）
    # ------------------------------------------------------------------
    def catmull_rom(self, points: Sequence[Sequence[float]], t: float) -> list:
        pts = [list(p) for p in points]
        if not pts:
            return [0.0, 0.0]
        if len(pts) == 1 or t <= 0.0:
            return list(pts[0])
        if t >= 1.0:
            return list(pts[-1])
        if len(pts) == 2:
            # 两点退化为直线，仍然连续平滑
            return _vadd(pts[0], _vscale(_vsub(pts[1], pts[0]), t))

        # 首尾补幽灵点，使端点切线也有定义
        extended = [_vsub(_vscale(pts[0], 2.0), pts[1])]
        extended.extend(pts)
        extended.append(_vsub(_vscale(pts[-1], 2.0), pts[-2]))

        seg_count = len(pts) - 1
        scaled = t * seg_count
        i = int(math.floor(scaled))
        if i >= seg_count:
            i = seg_count - 1
        u = scaled - i

        p0, p1, p2, p3 = extended[i], extended[i + 1], extended[i + 2], extended[i + 3]
        u2, u3 = u * u, u * u * u
        result = []
        for axis in range(len(p1)):
            v = 0.5 * (
                (2.0 * p1[axis])
                + (-p0[axis] + p2[axis]) * u
                + (2.0 * p0[axis] - 5.0 * p1[axis] + 4.0 * p2[axis] - p3[axis]) * u2
                + (-p0[axis] + 3.0 * p1[axis] - 3.0 * p2[axis] + p3[axis]) * u3
            )
            result.append(v)
        return result

    # 兼容旧名字：过场路径统一走曲线
    def lerp_path(self, points, t):
        return self.catmull_rom(points, t)

    # ------------------------------------------------------------------
    # 5. 碰撞：玩家 -> 相机 射线检测，命中则把相机拉近
    # ------------------------------------------------------------------
    def collide(self, player_pos, obstacles, cam_pos=None):
        player_pos = [float(v) for v in player_pos]
        if cam_pos is None:
            cam_pos = list(self.cam.pos)
        else:
            cam_pos = [float(v) for v in cam_pos]

        ray = _vsub(cam_pos, player_pos)
        ray_len = math.hypot(*ray)
        if ray_len <= 1e-12:
            return list(cam_pos)
        direction = _vscale(ray, 1.0 / ray_len)

        nearest = ray_len  # 最近命中点沿射线的距离
        for (cx, cy, radius) in obstacles:
            # 线段起点 player_pos、方向 direction，与圆求交
            ox = player_pos[0] - cx
            oy = player_pos[1] - cy
            b = ox * direction[0] + oy * direction[1]
            c = ox * ox + oy * oy - radius * radius
            disc = b * b - c
            if disc < 0.0:
                continue
            root = math.sqrt(disc)
            entry = -b - root  # 近端交点距离
            if entry < 0.0:
                entry = -b + root  # 起点在圆内时取远端
            if 0.0 <= entry <= ray_len and entry < nearest:
                nearest = entry

        if nearest >= ray_len:
            return list(cam_pos)
        # 停在障碍表面，留极小间距避免相机贴进碰撞体
        pull = max(0.0, nearest - 1e-4)
        return _vadd(player_pos, _vscale(direction, pull))

    # ------------------------------------------------------------------
    # 6. FOV 缩放：指数平滑过渡
    # ------------------------------------------------------------------
    def set_fov(self, fov: float):
        self.cam.target_fov = max(self.fov_min, min(self.fov_max, fov))

    def zoom(self, fov: float):
        self.set_fov(fov)

    def update_fov(self, dt: float) -> float:
        alpha = 1.0 - math.exp(-self.fov_lerp_speed * max(0.0, dt))
        self.cam.fov += (self.cam.target_fov - self.cam.fov) * alpha
        if abs(self.cam.fov - self.cam.target_fov) < 1e-4:
            self.cam.fov = self.cam.target_fov
        return self.cam.fov

    # ------------------------------------------------------------------
    # 7. 分层效果叠加：每层独立计算，结果合并而非覆盖
    # ------------------------------------------------------------------
    def add_effect(self, effect: Callable[[Camera], EffectResult]):
        self.effects.append(effect)

    def apply_effects(self, cam: Camera = None) -> Camera:
        cam = cam or self.cam
        base_pos = list(cam.pos)
        base_target = list(cam.target)
        base_fov = cam.fov
        combined = EffectResult()
        # 每层都以相同的基础姿态作为输入
        for effect in self.effects:
            snapshot = Camera(
                pos=list(base_pos),
                target=list(base_target),
                fov=base_fov,
                target_fov=cam.target_fov,
                shake_amp=cam.shake_amp,
                shake_base=cam.shake_base,
                shake_time=cam.shake_time,
                shake_total=cam.shake_total,
                shake_seed=cam.shake_seed,
            )
            combined = combined.merge(effect(snapshot))
        cam.pos = _vadd(base_pos, combined.pos_offset)
        cam.target = _vadd(base_target, combined.target_offset)
        cam.fov = base_fov + combined.fov_delta
        return cam

    # ------------------------------------------------------------------
    # 统一帧更新：按层依次更新各系统，再叠加效果
    # ------------------------------------------------------------------
    def update(self, target_pos, obstacles=(), dt: float = 1 / 60) -> Camera:
        self.follow(target_pos, dt)
        self.update_shake(dt)
        self.update_fov(dt)
        pulled = self.collide(target_pos, obstacles)
        self.cam.pos = pulled
        self.apply_effects()
        return self.cam
