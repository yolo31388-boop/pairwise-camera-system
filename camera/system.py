"""相机视角系统"""
import math
from dataclasses import dataclass


@dataclass
class Camera:
    position: list
    target: list
    fov: float = 60.0


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _length(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def _obstacle_radius(obs):
    if "radius" in obs:
        return float(obs["radius"])
    if "size" in obs:
        return float(obs["size"]) / 2.0
    return 1.0


def _first_hit_t(p0, p1, center, radius):
    """线段 p0->p1 首次进入球体的参数 t，未命中返回 None。"""
    d = _sub(p1, p0)
    f = _sub(p0, center)
    a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    if a == 0:
        return 0.0 if _length(f) < radius else None
    b = 2.0 * (f[0] * d[0] + f[1] * d[1] + f[2] * d[2])
    c = f[0] * f[0] + f[1] * f[1] + f[2] * f[2] - radius * radius
    if c < 0:
        return 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2.0 * a)
    if 0.0 <= t <= 1.0:
        return t
    return None


class CameraSystem:
    def __init__(self):
        self.camera = Camera([0, 5, -10], [0, 0, 0])
        self.locked_target = None
        self.known_targets = []
        self.yaw = 0.0
        self.pitch = 0.0
        self.yaw_speed = 0.0
        self.pitch_speed = 0.0
        self.eye_height = 5.0
        self.camera_distance = 10.0
        self.min_distance = 1.0
        self.collision_pad = 0.2
        self.smooth_time = 0.15
        self.max_speed = 30.0
        self.adaptive_dist = 1.0
        self.input_sensitivity = 0.1
        self.input_accel = 0.5
        self.max_pitch = 89.0
        self.last_player_pos = [0, 0, 0]

    def _resolve_collision(self, eye, desired, obstacles):
        """检查 eye->desired 视线，把相机拉到第一个遮挡物之前。"""
        pos = list(desired)
        seg_len = _length(_sub(desired, eye))
        if seg_len == 0:
            return pos
        best_t = 1.0
        for obs in obstacles:
            if "pos" not in obs:
                continue
            r = _obstacle_radius(obs) + self.collision_pad
            t = _first_hit_t(eye, pos, obs["pos"], r)
            if t is not None:
                best_t = min(best_t, max(self.min_distance / seg_len,
                                         t - r / seg_len))
        return _lerp(eye, desired, best_t)

    def update_camera(self, player_pos: list, obstacles: list) -> None:
        self.last_player_pos = list(player_pos)
        if self.locked_target is not None and not self.locked_target.get("alive", True):
            self._switch_to_next_target()
        eye = [player_pos[0], player_pos[1] + self.eye_height, player_pos[2]]
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        offset = [
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            -math.cos(yaw) * math.cos(pitch),
        ]
        desired = [eye[i] + offset[i] * self.camera_distance for i in range(3)]
        self.camera.position = self._resolve_collision(eye, desired, obstacles)
        if self.locked_target is not None:
            self.camera.target = list(self.locked_target.get("pos", eye))
        else:
            self.camera.target = eye

    def smooth_camera(self, target_pos: list, delta: float) -> None:
        """帧率无关的指数平滑，快速转身时按最大速度自适应。"""
        delta = max(delta, 0.0)
        factor = 1.0 - math.exp(-delta / self.smooth_time)
        disp_all = _sub(target_pos, self.camera.position)
        dist_all = _length(disp_all)
        for i in range(3):
            disp = disp_all[i] * factor
            max_step = self.max_speed * delta
            if dist_all > self.adaptive_dist and abs(disp) > max_step:
                disp = math.copysign(max_step, disp)
            self.camera.position[i] += disp

    def lock_on_target(self, target: dict) -> None:
        if target is None:
            self.locked_target = None
            return
        if target.get("alive", True):
            self.locked_target = target
            if target not in self.known_targets:
                self.known_targets.append(target)
            self.camera.target = list(target.get("pos", [0, 0, 0]))
        else:
            self._switch_to_next_target()

    def _switch_to_next_target(self) -> None:
        self.known_targets = [t for t in self.known_targets if t.get("alive", True)]
        if self.known_targets:
            self.locked_target = self.known_targets[0]
            self.camera.target = list(self.locked_target.get("pos", [0, 0, 0]))
        else:
            self.locked_target = None

    def handle_occlusion(self, obstacles: list) -> None:
        """遮挡物透明化，按其大小把相机拉到前方，必要时持续拉近。"""
        anchor = list(self.camera.target)
        for obs in obstacles:
            if "pos" not in obs:
                continue
            r = _obstacle_radius(obs) + self.collision_pad
            t = _first_hit_t(anchor, self.camera.position, obs["pos"], r)
            if t is None:
                continue
            obs["transparent"] = True
            seg_len = _length(_sub(self.camera.position, anchor))
            if seg_len == 0:
                continue
            t_new = max(self.min_distance / seg_len, t - r / seg_len)
            self.camera.position = _lerp(anchor, self.camera.position, t_new)
        for _ in range(16):
            blocker = None
            for obs in obstacles:
                if "pos" not in obs:
                    continue
                r = _obstacle_radius(obs) + self.collision_pad
                if _first_hit_t(anchor, self.camera.position, obs["pos"], r) is not None:
                    blocker = obs
                    obs["transparent"] = True
            if blocker is None:
                break
            seg_len = _length(_sub(self.camera.position, anchor))
            if seg_len <= self.min_distance:
                break
            t_new = max(self.min_distance / seg_len,
                        1.0 - _obstacle_radius(blocker) / seg_len - 0.1)
            self.camera.position = _lerp(anchor, self.camera.position, t_new)

    def process_camera_input(self, dx: float, dy: float) -> None:
        """输入平滑（加速度）并限制俯仰角。"""
        raw_yaw = dx * self.input_sensitivity
        raw_pitch = dy * self.input_sensitivity
        self.yaw_speed += (raw_yaw - self.yaw_speed) * self.input_accel
        self.pitch_speed += (raw_pitch - self.pitch_speed) * self.input_accel
        self.yaw = (self.yaw + self.yaw_speed + 180.0) % 360.0 - 180.0
        self.pitch = max(-self.max_pitch,
                         min(self.max_pitch, self.pitch + self.pitch_speed))
        if self.locked_target is None:
            self.camera.target[0] = self.yaw
            self.camera.target[1] = self.pitch
