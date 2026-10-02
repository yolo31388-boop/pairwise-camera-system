"""相机视角系统。

修复内容：
- update_camera：相机位置做球体碰撞检测 + 玩家到相机的视线检测（逐障碍取最近安全点）
- smooth_camera：帧率无关的指数插值，快速转身/大位移时自适应提高跟随速度
- lock_on_target：锁定时相机以玩家为中心环绕、跟随玩家移动，支持目标切换与死亡清理
- handle_occlusion：遮挡时沿视线拉近相机，并将遮挡物透明化（按障碍尺寸拉近）
- process_camera_input：输入做带速度的加速度平滑，限制俯仰角，防止转到地下
"""
from dataclasses import dataclass
import math


@dataclass
class Camera:
    position: list
    target: list
    fov: float = 60.0


def _vsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vadd(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _vscale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def _vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vlen(a):
    return math.sqrt(_vdot(a, a))


def _obstacle_center(obs):
    return list(obs.get("center", obs.get("pos", [0.0, 0.0, 0.0])))


def _obstacle_radius(obs):
    size = obs.get("size", 2.0)
    if isinstance(size, (list, tuple)):
        return max(size) * 0.5
    return float(size)


def _ray_sphere_exit(origin, direction_sq_len, oc, b, c):
    """返回障碍球沿射线方向的出射点参数 t（相机端边界），无交叠返回 None。"""
    disc = b * b - direction_sq_len * c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    t1 = (-b - root) / direction_sq_len
    t2 = (-b + root) / direction_sq_len
    if t2 < 0 or t1 > 1:
        return None
    return t2


class CameraSystem:
    def __init__(self):
        self.camera = Camera([0, 5, -10], [0, 0, 0])
        self.locked_target = None

        self.distance = 10.0
        self.height = 5.0
        self.collision_radius = 0.4
        self.pitch = math.radians(20.0)
        self.yaw = 0.0
        self.player_pos = [0.0, 0.0, 0.0]

        self.sensitivity = 0.12
        self.max_pitch = math.radians(80.0)
        self.input_velocity = [0.0, 0.0]
        self.turn_speed = 0.0

        self.smooth_half_life = 0.08
        self.fast_half_life = 0.02
        self.transparent_obstacles = []

    # ------------------------------------------------------------------
    # 碰撞辅助
    # ------------------------------------------------------------------
    def _safe_camera_position(self, anchor, desired, obstacles):
        """沿 anchor -> desired 方向做视线检测，相机本身做球体碰撞检测。"""
        segment = _vsub(desired, anchor)
        seg_len_sq = _vdot(segment, segment)
        if seg_len_sq <= 1e-8:
            return list(desired)
        seg_len = math.sqrt(seg_len_sq)

        min_t = 1.0
        for obs in obstacles:
            center = _obstacle_center(obs)
            radius = _obstacle_radius(obs)
            oc = _vsub(anchor, center)
            b = _vdot(oc, segment)
            c = _vdot(oc, oc) - radius * radius

            # 视线：障碍球与线段相交则停在入射点之前
            disc = b * b - seg_len_sq * c
            if disc >= 0:
                root = math.sqrt(disc)
                t_enter = (-b - root) / seg_len_sq
                t_exit = (-b + root) / seg_len_sq
                if t_exit >= 0 and t_enter <= 1:
                    hit_t = max(0.0, min(1.0, t_enter))
                    min_t = min(min_t, hit_t)

            # 相机球体本身：相机中心不能进入膨胀后的障碍
            cam_r = radius + self.collision_radius
            c_cam = _vdot(oc, oc) - cam_r * cam_r
            disc_cam = b * b - seg_len_sq * c_cam
            if disc_cam >= 0:
                root = math.sqrt(disc_cam)
                t_enter = (-b - root) / seg_len_sq
                t_exit = (-b + root) / seg_len_sq
                if t_exit >= 0 and t_enter <= 1:
                    hit_t = max(0.0, min(1.0, t_enter))
                    min_t = min(min_t, hit_t)

        if min_t >= 1.0:
            safe = list(desired)
        else:
            safe_t = max(0.0, min_t - 1e-3)
            safe = _vadd(anchor, _vscale(segment, safe_t))
        return safe

    def _desired_position(self, player_pos):
        x, _, z = player_pos
        horizontal = self.distance * math.cos(self.pitch)
        x += horizontal * math.sin(self.yaw)
        z -= horizontal * math.cos(self.yaw)
        y = player_pos[1] + self.height + self.distance * math.sin(self.pitch)
        return [x, y, z]

    def _eye_anchor(self, player_pos):
        return [player_pos[0], player_pos[1] + 1.6, player_pos[2]]

    # ------------------------------------------------------------------
    # 每帧更新（碰撞 + 视线）
    # ------------------------------------------------------------------
    def update_camera(self, player_pos, obstacles):
        self.player_pos = list(player_pos)

        if self.locked_target is not None:
            target_pos = self.locked_target.get("pos", self.player_pos)
            if not self.locked_target.get("alive", True):
                self.locked_target = None
            else:
                direction = _vsub(target_pos, player_pos)
                dist = _vlen(direction)
                if dist > 1e-4:
                    self.yaw = math.atan2(direction[0], -direction[2])

        desired = self._desired_position(player_pos)
        anchor = self._eye_anchor(player_pos)
        safe = self._safe_camera_position(anchor, desired, obstacles)
        self.camera.position = safe

        if self.locked_target is not None:
            self.camera.target = list(self.locked_target["pos"])
        else:
            self.camera.target = list(player_pos)
        return safe

    # ------------------------------------------------------------------
    # 帧率无关的平滑（快速转身自适应）
    # ------------------------------------------------------------------
    def smooth_camera(self, target_pos, delta):
        if delta <= 0:
            return list(self.camera.position)

        gap = max(abs(target_pos[i] - self.camera.position[i]) for i in range(3))
        gap_speed = min(1.0, gap / 20.0)
        adaptation = max(self.turn_speed, gap_speed)
        half_life = self.smooth_half_life + (self.fast_half_life - self.smooth_half_life) * adaptation
        alpha = 1.0 - math.pow(0.5, delta / half_life)

        for i in range(3):
            self.camera.position[i] += (target_pos[i] - self.camera.position[i]) * alpha
        self.turn_speed = max(0.0, self.turn_speed - delta * 3.0)
        return list(self.camera.position)

    # ------------------------------------------------------------------
    # 视角锁定
    # ------------------------------------------------------------------
    def lock_on_target(self, target):
        if target is None:
            self.locked_target = None
            self.camera.target = list(self.player_pos)
            return
        if not target.get("alive", True):
            return
        if target is self.locked_target or target == self.locked_target:
            return
        self.locked_target = target
        self.camera.target = list(target.get("pos", [0, 0, 0]))

    def cycle_target(self, candidates):
        """在多个候选目标中切换：跳过死亡目标，优先当前锁定之后的下一个。"""
        alive = [t for t in candidates if t.get("alive", True)]
        if not alive:
            self.locked_target = None
            self.camera.target = list(self.player_pos)
            return None
        if self.locked_target in alive:
            index = (alive.index(self.locked_target) + 1) % len(alive)
        else:
            index = 0
        self.locked_target = alive[index]
        self.camera.target = list(self.locked_target.get("pos", [0, 0, 0]))
        return self.locked_target

    # ------------------------------------------------------------------
    # 遮挡处理（拉近 + 透明化，按障碍尺寸）
    # ------------------------------------------------------------------
    def _ray_blocks(self, origin, end, obstacles):
        segment = _vsub(end, origin)
        seg_len_sq = _vdot(segment, segment)
        blockers = []
        if seg_len_sq <= 1e-8:
            return blockers
        for obs in obstacles:
            oc = _vsub(origin, _obstacle_center(obs))
            radius = _obstacle_radius(obs)
            b = _vdot(oc, segment)
            c = _vdot(oc, oc) - radius * radius
            disc = b * b - seg_len_sq * c
            if disc < 0:
                continue
            root = math.sqrt(disc)
            t1 = (-b - root) / seg_len_sq
            t2 = (-b + root) / seg_len_sq
            if t2 >= 0 and t1 <= 1:
                blockers.append((obs, max(0.0, t1), t2))
        return blockers

    def handle_occlusion(self, obstacles, anchor=None):
        if anchor is None:
            anchor = self._eye_anchor(self.player_pos)
        else:
            anchor = list(anchor)

        blocking_ids = set()
        blockers = self._ray_blocks(anchor, self.camera.position, obstacles)
        if blockers:
            nearest = min(blockers, key=lambda item: item[1])
            obs, t_enter, t_exit = nearest
            radius = _obstacle_radius(obs)
            pull = (radius + self.collision_radius + 0.15) / max(_vlen(_vsub(self.camera.position, anchor)), 1e-6)
            new_t = max(0.0, t_enter - pull)
            segment = _vsub(self.camera.position, anchor)
            self.camera.position = _vadd(anchor, _vscale(segment, new_t))
            blocking_ids.add(id(obs))

        for obs in obstacles:
            if isinstance(obs, dict):
                if id(obs) in blocking_ids:
                    obs["transparent"] = True
                elif obs.get("transparent") and id(obs) in {id(o) for o in self.transparent_obstacles}:
                    obs["transparent"] = False
        self.transparent_obstacles = [obs for obs in obstacles if id(obs) in blocking_ids]
        return blockers

    # ------------------------------------------------------------------
    # 输入处理（加速度平滑 + 角度限制）
    # ------------------------------------------------------------------
    def process_camera_input(self, dx, dy, delta=1.0 / 60.0):
        if delta <= 0:
            delta = 1.0 / 60.0

        target_velocity = [dx / delta, dy / delta]
        response = 1.0 - math.pow(0.05, delta)
        for i in range(2):
            self.input_velocity[i] += (target_velocity[i] - self.input_velocity[i]) * response

        smoothed_dx = self.input_velocity[0] * delta
        smoothed_dy = self.input_velocity[1] * delta

        # 输入加速度：移动越快，单位输入的转向增益越大（轻微非线性）
        input_speed = math.hypot(smoothed_dx, smoothed_dy)
        accel_gain = 1.0 + min(1.0, input_speed / 500.0) * 0.5
        gain = self.sensitivity * accel_gain

        self.yaw += smoothed_dx * gain
        self.pitch -= smoothed_dy * gain
        self.pitch = max(-self.max_pitch, min(self.max_pitch, self.pitch))

        self.turn_speed = min(1.0, input_speed / 800.0)
        return self.yaw, self.pitch
