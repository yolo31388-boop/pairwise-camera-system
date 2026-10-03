"""pairwise-camera-system - CameraSystem.

A small, dependency-free 3D game camera system.

- follow_target: predictive following with look-ahead framing offset and a
  dead zone (no trailing, no screen-centre glue, no micro-jitter).
- resolve_camera_collision: target->camera raycast against sphere/AABB
  colliders, asymmetric smoothing (no pops) and occlusion reporting so
  blocking geometry can be faded out.
- shake_camera: deterministic layered-sine waveforms with decay envelopes,
  priority arbitration and directional spatialisation.
- switch_camera: eased transitions blending position, focus (look_at) and
  FOV, with action-lock timing gates.
- apply_constraints: pitch/yaw and distance limits implemented as soft
  (spring-damped) limits with a hard safety clamp.

All vectors are ``(x, y, z)`` sequences of floats; angles are in degrees.
"""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]


def _vec3(value, default=(0.0, 0.0, 0.0)) -> Vec3:
    if value is None:
        return (float(default[0]), float(default[1]), float(default[2]))
    if isinstance(value, dict):
        return (
            float(value.get("x", 0.0)),
            float(value.get("y", 0.0)),
            float(value.get("z", 0.0)),
        )
    if isinstance(value, (int, float)):
        return (float(value),) * 3
    seq = list(value)
    while len(seq) < 3:
        seq.append(0.0)
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _scalar(value, default=0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return float(value)


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _length(a)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _lerp(a, b, t):
    return a + (b - a) * t


def _lerp_vec(a, b, t):
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t), _lerp(a[2], b[2], t))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _damp_alpha(speed, dt):
    """Frame-rate independent exponential damping factor in [0, 1]."""
    speed = max(0.0, float(speed))
    dt = max(0.0, float(dt))
    if speed <= 0.0:
        return 1.0
    return 1.0 - math.exp(-speed * dt)


def _ease(name, t):
    t = _clamp(float(t), 0.0, 1.0)
    if name in (None, "linear"):
        return t
    if name in ("smoothstep", "smooth"):
        return t * t * (3.0 - 2.0 * t)
    if name in ("ease_in", "ease-in", "easein"):
        return t * t
    if name in ("ease_out", "ease-out", "easeout"):
        return 1.0 - (1.0 - t) ** 2
    if name in ("ease_in_out", "ease-in-out", "easeinout", "cubic"):
        return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0
    raise ValueError(f"unknown easing curve: {name!r}")


def _ray_aabb(origin, direction, box_min, box_max, expand, max_t):
    """Slab intersection. Returns entry t in (0, max_t] or None."""
    t_min, t_max = 0.0, max_t
    for i in range(3):
        lo = box_min[i] - expand
        hi = box_max[i] + expand
        d = direction[i]
        if abs(d) < 1e-12:
            if origin[i] < lo or origin[i] > hi:
                return None
            continue
        t1 = (lo - origin[i]) / d
        t2 = (hi - origin[i]) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    return t_min if 1e-7 < t_min <= max_t else None


def _ray_sphere(origin, direction, center, radius, max_t):
    oc = _sub(origin, center)
    b = _dot(oc, direction)
    c = _dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return None
    t = -b - math.sqrt(disc)
    return t if 1e-7 < t <= max_t else None


def _collider_hit(origin, direction, collider, expand, max_t):
    if not isinstance(collider, dict):
        return None
    ctype = str(collider.get("type", "aabb")).lower()
    if ctype == "sphere":
        center = _vec3(collider.get("center"))
        radius = _scalar(collider.get("radius"), 0.5) + expand
        return _ray_sphere(origin, direction, center, radius, max_t)
    if "min" in collider and "max" in collider:
        box_min = _vec3(collider["min"])
        box_max = _vec3(collider["max"])
    else:
        center = _vec3(collider.get("center"))
        size = _vec3(collider.get("size", (1.0, 1.0, 1.0)))
        half = _scale(size, 0.5)
        box_min, box_max = _sub(center, half), _add(center, half)
    return _ray_aabb(origin, direction, box_min, box_max, expand, max_t)


def _perpendicular(direction):
    up = (0.0, 1.0, 0.0) if abs(direction[1]) < 0.9 else (1.0, 0.0, 0.0)
    return _normalize(_cross(direction, up))


class CameraSystem:
    """Stateful camera controller. Instances are reusable per camera rig."""

    DEFAULT_FOV = 60.0
    DEFAULT_CAMERA = {"position": (0.0, 2.0, -5.0), "look_at": (0.0, 0.0, 0.0), "fov": 60.0}

    def __init__(self):
        self.state = {}
        self._follow_position = None
        self._follow_anchor = None
        self._collision_t = 1.0
        self._shakes = {}
        self._next_shake_id = 1
        self._transition = None
        self._pending_switch = None
        self._settled_camera = None
        self._constraints = None

    # ------------------------------------------------------------------
    # 跟随：预测 + 偏移 + 死区
    # ------------------------------------------------------------------
    def follow_target(
        self,
        target_pos=None,
        camera_pos=None,
        target_velocity=None,
        dt=1.0 / 60.0,
        smooth_speed=6.0,
        look_ahead=0.5,
        max_look_ahead=3.0,
        dead_zone=0.25,
        offset=(0.0, 0.8, 0.0),
        **_ignored,
    ):
        """Smoothly follow ``target_pos``.

        - prediction: the framing point leads the target by
          ``target_velocity * look_ahead`` (clamped to ``max_look_ahead``),
          so sprinting no longer drags the camera behind.
        - offset: a constant framing offset keeps the target off the exact
          screen centre.
        - dead zone: per-axis movement smaller than ``dead_zone`` (world
          units) is absorbed, removing micro-jitter while the target idles.
        """
        dt = max(_scalar(dt, 1.0 / 60.0), 1e-6)
        dead_zone = max(0.0, _scalar(dead_zone, 0.25))
        look_ahead = max(0.0, _scalar(look_ahead, 0.5))
        max_lead = max(0.0, _scalar(max_look_ahead, 3.0))
        offset_v = _vec3(offset, (0.0, 0.8, 0.0))
        target = _vec3(target_pos)
        velocity = _vec3(target_velocity)

        lead = _scale(velocity, look_ahead)
        lead_len = _length(lead)
        if lead_len > max_lead > 0.0:
            lead = _scale(lead, max_lead / lead_len)
        predicted = _add(target, lead)

        if self._follow_anchor is not None and dead_zone > 0.0:
            clamped = []
            inside = True
            for i in range(3):
                delta = predicted[i] - self._follow_anchor[i]
                if abs(delta) <= dead_zone:
                    clamped.append(self._follow_anchor[i])
                else:
                    inside = False
                    clamped.append(predicted[i] - math.copysign(dead_zone, delta))
            anchor = tuple(clamped)
            in_dead_zone = inside
        else:
            anchor = predicted
            in_dead_zone = False

        desired = _add(anchor, offset_v)
        if camera_pos is not None:
            current = _vec3(camera_pos)
        elif self._follow_position is not None:
            current = self._follow_position
        else:
            current = desired

        alpha = _damp_alpha(_scalar(smooth_speed, 6.0), dt)
        position = _lerp_vec(current, desired, alpha)

        self._follow_position = position
        self._follow_anchor = anchor
        return {
            "position": position,
            "desired": desired,
            "predicted_target": predicted,
            "look_at": desired,
            "offset": offset_v,
            "dead_zone": dead_zone,
            "in_dead_zone": in_dead_zone,
        }

    # ------------------------------------------------------------------
    # 碰撞：检测 + 平滑 + 遮挡处理
    # ------------------------------------------------------------------
    def resolve_camera_collision(
        self,
        target_pos=None,
        camera_pos=None,
        colliders=None,
        dt=1.0 / 60.0,
        radius=0.3,
        min_distance=0.2,
        smooth_in=14.0,
        smooth_out=4.0,
        **_ignored,
    ):
        """Pull the camera in before it clips through level geometry.

        - detection: the target->desired-camera segment is raycast against
          sphere/AABB colliders expanded by the camera ``radius``; the
          nearest hit wins.
        - smoothing: the resolved distance springs in fast (``smooth_in``)
          and relaxes out slowly (``smooth_out``); the smoothed point is
          never allowed past a detected wall, so there are no pops.
        - occlusion: anything still blocking the resolved line of sight is
          returned in ``occluders`` with ``fade_occluders`` set, so the
          renderer can fade it out instead of hiding the target.
        """
        dt = max(_scalar(dt, 1.0 / 60.0), 1e-6)
        radius = max(0.0, _scalar(radius, 0.3))
        min_distance = max(0.0, _scalar(min_distance, 0.2))
        target = _vec3(target_pos)
        desired = _vec3(camera_pos, (0.0, 2.0, -5.0))
        colliders = list(colliders or [])

        segment = _sub(desired, target)
        seg_len = _length(segment)
        if seg_len < 1e-9:
            direction = (0.0, 0.0, -1.0)
            seg_len = 1.0
        else:
            direction = _scale(segment, 1.0 / seg_len)

        hit_t, hit_collider = None, None
        for collider in colliders:
            t_world = _collider_hit(target, direction, collider, radius, seg_len)
            t = t_world / seg_len if t_world is not None else None
            if t is not None and (hit_t is None or t < hit_t):
                hit_t, hit_collider = t, collider

        floor_t = min_distance / seg_len
        if hit_t is not None:
            wall_t = max(hit_t - radius / seg_len, floor_t)
        else:
            wall_t = 1.0
        wanted_t = _clamp(wall_t, floor_t, 1.0)

        speed = _scalar(smooth_in if wanted_t < self._collision_t else smooth_out, 8.0)
        alpha = _damp_alpha(speed, dt)
        smoothed_t = _lerp(self._collision_t, wanted_t, alpha)
        if hit_t is not None:
            smoothed_t = min(smoothed_t, wall_t)
        smoothed_t = max(smoothed_t, floor_t)
        self._collision_t = smoothed_t

        position = _add(target, _scale(direction, smoothed_t * seg_len))

        occluders = []
        sight = _sub(position, target)
        sight_len = _length(sight)
        if sight_len > 1e-9:
            sight_dir = _scale(sight, 1.0 / sight_len)
            for collider in colliders:
                t = _collider_hit(target, sight_dir, collider, 0.0, sight_len)
                if t is not None:
                    occluders.append(collider)

        return {
            "position": position,
            "distance": smoothed_t * seg_len,
            "collided": hit_t is not None,
            "hit": hit_collider,
            "occluded": bool(occluders),
            "occluders": occluders,
            "fade_occluders": bool(occluders),
        }

    # ------------------------------------------------------------------
    # 震动：波形 + 衰减 + 优先级 + 空间化
    # ------------------------------------------------------------------
    @staticmethod
    def _decay_envelope(kind, progress):
        remaining = 1.0 - _clamp(progress, 0.0, 1.0)
        kind = (kind or "cubic").lower()
        if kind == "linear":
            return remaining
        if kind in ("exponential", "exp"):
            return (math.exp(-4.0 * progress) - math.exp(-4.0)) / (1.0 - math.exp(-4.0))
        if kind in ("none", "constant"):
            return 1.0 if progress < 1.0 else 0.0
        return remaining ** 3  # cubic / smooth

    def _strongest_shake(self):
        best = None
        for shake in self._shakes.values():
            if best is None or (shake["priority"], shake["intensity"]) > (
                best["priority"],
                best["intensity"],
            ):
                best = shake
        return best

    def shake_camera(
        self,
        intensity=None,
        duration=None,
        frequency=12.0,
        priority=0,
        direction=None,
        position=None,
        source=None,
        decay="cubic",
        dt=1.0 / 60.0,
        **_ignored,
    ):
        """Trigger/advance camera shake.

        - waveform/decay: three decorrelated sine layers (deterministic
          seeded phases, no per-frame randomness) multiplied by an envelope
          that eases out and lands on exactly zero, so shakes taper instead
          of cutting off.
        - priority: a new shake at lower priority is ignored while a
          stronger one is active; equal or higher priority refreshes it.
        - spatialisation: ``direction`` aims the shake; when only a
          ``source`` position and camera ``position`` are given the
          direction is derived from them, giving 2D shakes a sense of
          direction instead of isotropic noise.
        """
        dt = max(_scalar(dt, 1.0 / 60.0), 0.0)

        if intensity is not None:
            intensity = _scalar(intensity)
            duration = max(_scalar(duration, 0.5), 1e-6)
            active = self._strongest_shake()
            if active is None or int(priority) >= int(active["priority"]):
                if direction is not None:
                    shake_dir = _vec3(direction)
                elif source is not None:
                    shake_dir = _sub(_vec3(position), _vec3(source))
                else:
                    shake_dir = (0.0, 1.0, 0.0)
                shake_dir = _normalize(shake_dir)
                if _length(shake_dir) < 1e-9:
                    shake_dir = (0.0, 1.0, 0.0)
                shake_id = self._next_shake_id
                self._next_shake_id += 1
                self._shakes[shake_id] = {
                    "id": shake_id,
                    "intensity": intensity,
                    "duration": duration,
                    "elapsed": 0.0,
                    "frequency": max(0.0, _scalar(frequency, 12.0)),
                    "priority": int(priority),
                    "direction": shake_dir,
                    "decay": decay,
                    "seed": shake_id * 12.9898,
                }

        offset = [0.0, 0.0, 0.0]
        strongest = None
        finished = []
        for shake_id, shake in self._shakes.items():
            shake["elapsed"] += dt
            progress = shake["elapsed"] / shake["duration"]
            if progress >= 1.0:
                finished.append(shake_id)
                continue
            envelope = self._decay_envelope(shake["decay"], progress)
            amplitude = shake["intensity"] * envelope
            omega = 2.0 * math.pi * shake["frequency"]
            elapsed = shake["elapsed"]
            primary = math.sin(omega * elapsed + shake["seed"])
            secondary = math.sin(omega * 1.37 * elapsed + shake["seed"] * 2.0)
            tertiary = math.sin(omega * 0.71 * elapsed + shake["seed"] * 3.0)
            direction_v = shake["direction"]
            sideways = _perpendicular(direction_v)
            vertical = _normalize(_cross(direction_v, sideways))
            contribution = _add(
                _scale(direction_v, 0.7 * primary),
                _add(
                    _scale(sideways, 0.2 * secondary),
                    _scale(vertical, 0.1 * tertiary),
                ),
            )
            contribution = _scale(contribution, amplitude)
            offset = _add(offset, contribution)
            if strongest is None or (shake["priority"], amplitude) > (
                strongest["priority"],
                strongest["amplitude"],
            ):
                strongest = {
                    "id": shake["id"],
                    "priority": shake["priority"],
                    "amplitude": amplitude,
                }

        for shake_id in finished:
            del self._shakes[shake_id]

        return {
            "offset": tuple(offset),
            "active": len(self._shakes) > 0,
            "active_count": len(self._shakes),
            "strongest_priority": strongest["priority"] if strongest else None,
            "shakes": [
                {
                    "id": s["id"],
                    "intensity": s["intensity"],
                    "priority": s["priority"],
                    "elapsed": s["elapsed"],
                    "duration": s["duration"],
                    "direction": s["direction"],
                }
                for s in self._shakes.values()
            ],
        }

    # ------------------------------------------------------------------
    # 切换：位置 + 焦点 + FOV 过渡，时机门控，缓动曲线
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_camera(camera):
        if isinstance(camera, dict):
            look = camera.get("look_at", camera.get("focus", camera.get("target")))
            return {
                "position": _vec3(camera.get("position")),
                "look_at": _vec3(look),
                "fov": _scalar(camera.get("fov"), CameraSystem.DEFAULT_FOV),
            }
        return {
            "position": _vec3(camera),
            "look_at": (0.0, 0.0, 0.0),
            "fov": CameraSystem.DEFAULT_FOV,
        }

    def switch_camera(
        self,
        from_camera=None,
        to_camera=None,
        dt=1.0 / 60.0,
        duration=1.0,
        easing="smoothstep",
        action_locked=False,
        **_ignored,
    ):
        """Blend between two camera states.

        - focus/FOV: ``position``, ``look_at`` (focus) and ``fov`` all
          interpolate together, so framing never pops mid-cut.
        - timing: while ``action_locked`` is true the request is queued and
          only starts on the first unlocked frame, so cuts never land in
          the middle of an action.
        - curve: eased interpolation (default smoothstep; also linear,
          ease_in, ease_out, ease_in_out) instead of a robotic linear cut.
        """
        dt = max(_scalar(dt, 1.0 / 60.0), 0.0)

        if to_camera is not None:
            if action_locked:
                self._pending_switch = {
                    "from_camera": from_camera,
                    "to_camera": to_camera,
                    "duration": duration,
                    "easing": easing,
                }
            else:
                start = from_camera
                if start is None and self._transition is not None:
                    start = self._transition["current"]
                if start is None:
                    start = self.DEFAULT_CAMERA
                self._transition = {
                    "from": self._normalize_camera(start),
                    "to": self._normalize_camera(to_camera),
                    "elapsed": 0.0,
                    "duration": max(_scalar(duration, 1.0), 1e-6),
                    "easing": easing,
                    "current": self._normalize_camera(start),
                }
                self._pending_switch = None

        if self._transition is None and self._pending_switch is not None and not action_locked:
            pending = self._pending_switch
            self._pending_switch = None
            return self.switch_camera(
                from_camera=pending["from_camera"],
                to_camera=pending["to_camera"],
                dt=dt,
                duration=pending["duration"],
                easing=pending["easing"],
                action_locked=False,
            )

        transition = self._transition
        if transition is None:
            idle = self._normalize_camera(
                self._settled_camera if self._settled_camera is not None else self.DEFAULT_CAMERA
            )
            return {
                "position": idle["position"],
                "look_at": idle["look_at"],
                "fov": idle["fov"],
                "active": False,
                "progress": 1.0,
                "done": True,
                "queued": self._pending_switch is not None,
            }

        transition["elapsed"] += dt
        progress = _clamp(transition["elapsed"] / transition["duration"], 0.0, 1.0)
        eased = _ease(transition["easing"], progress)
        start, end = transition["from"], transition["to"]
        current = {
            "position": _lerp_vec(start["position"], end["position"], eased),
            "look_at": _lerp_vec(start["look_at"], end["look_at"], eased),
            "fov": _lerp(start["fov"], end["fov"], eased),
        }
        transition["current"] = current
        done = progress >= 1.0
        if done:
            self._settled_camera = dict(current)
            self._transition = None
        return {
            "position": current["position"],
            "look_at": current["look_at"],
            "fov": current["fov"],
            "active": not done,
            "progress": progress,
            "done": done,
            "queued": self._pending_switch is not None,
        }

    # ------------------------------------------------------------------
    # 约束：角度限制 + 距离限制 + 软限制
    # ------------------------------------------------------------------
    @staticmethod
    def _soft_clamp(value, lo, hi, margin):
        """Soft limiting: excess beyond the bound is progressively resisted.

        Returns the resisted value (it can dip into the soft zone) and
        whether the limit was engaged.
        """
        if lo is not None and value < lo:
            overshoot = lo - value
            resisted = lo - overshoot * margin / (margin + overshoot)
            return resisted, True
        if hi is not None and value > hi:
            overshoot = value - hi
            resisted = hi + overshoot * margin / (margin + overshoot)
            return resisted, True
        return value, False

    def apply_constraints(
        self,
        target_pos=None,
        yaw=0.0,
        pitch=20.0,
        distance=5.0,
        offset=(0.0, 0.8, 0.0),
        dt=1.0 / 60.0,
        min_pitch=-30.0,
        max_pitch=70.0,
        min_yaw=None,
        max_yaw=None,
        min_distance=2.0,
        max_distance=10.0,
        soft_margin=0.15,
        stiffness=10.0,
        **_ignored,
    ):
        """Constrain an orbit camera.

        - angle limits: pitch is kept in [min_pitch, max_pitch] (so the
          camera can never dive underground); optional yaw bounds work the
          same way.
        - distance limits: orbit distance stays in [min_distance,
          max_distance], so zoom can neither run away nor collapse.
        - soft limits: excess past a bound is resisted into a soft zone
          (up to ``soft_margin`` past the bound) and the current value
          springs toward the resisted value with ``stiffness``; a final
          hard clamp at the edge of the soft zone caps the overshoot,
          replacing the old instantaneous stop.
        """
        dt = max(_scalar(dt, 1.0 / 60.0), 1e-6)
        margin = max(0.0, _scalar(soft_margin, 0.15))
        target = _vec3(target_pos)
        anchor = _add(target, _vec3(offset, (0.0, 0.8, 0.0)))

        yaw_in = _scalar(yaw, 0.0)
        pitch_in = _scalar(pitch, 20.0)
        distance_in = _scalar(distance, 5.0)
        min_pitch_v = None if min_pitch is None else _scalar(min_pitch)
        max_pitch_v = None if max_pitch is None else _scalar(max_pitch)
        min_yaw_v = None if min_yaw is None else _scalar(min_yaw)
        max_yaw_v = None if max_yaw is None else _scalar(max_yaw)
        min_distance_v = None if min_distance is None else _scalar(min_distance)
        max_distance_v = None if max_distance is None else _scalar(max_distance)

        soft_pitch, pitch_engaged = self._soft_clamp(pitch_in, min_pitch_v, max_pitch_v, margin)
        soft_distance, distance_engaged = self._soft_clamp(
            distance_in, min_distance_v, max_distance_v, margin
        )
        if min_yaw_v is not None or max_yaw_v is not None:
            soft_yaw, yaw_engaged = self._soft_clamp(yaw_in, min_yaw_v, max_yaw_v, margin)
        else:
            soft_yaw, yaw_engaged = ((yaw_in + 180.0) % 360.0) - 180.0, False

        alpha = _damp_alpha(_scalar(stiffness, 10.0), dt)
        if self._constraints is None:
            cur_pitch, cur_yaw, cur_distance = soft_pitch, soft_yaw, soft_distance
        else:
            prev = self._constraints
            cur_pitch = _lerp(prev["pitch"], soft_pitch, alpha)
            cur_yaw = _lerp(prev["yaw"], soft_yaw, alpha)
            cur_distance = _lerp(prev["distance"], soft_distance, alpha)

        final_pitch = _clamp(
            cur_pitch,
            min_pitch_v - margin if min_pitch_v is not None else -math.inf,
            max_pitch_v + margin if max_pitch_v is not None else math.inf,
        )
        final_distance = _clamp(
            cur_distance,
            min_distance_v - margin if min_distance_v is not None else -math.inf,
            max_distance_v + margin if max_distance_v is not None else math.inf,
        )
        if min_yaw_v is not None or max_yaw_v is not None:
            final_yaw = _clamp(
                cur_yaw,
                min_yaw_v - margin if min_yaw_v is not None else -math.inf,
                max_yaw_v + margin if max_yaw_v is not None else math.inf,
            )
        else:
            final_yaw = ((cur_yaw + 180.0) % 360.0) - 180.0

        pitch_rad = math.radians(final_pitch)
        yaw_rad = math.radians(final_yaw)
        horizontal = math.cos(pitch_rad)
        back = (
            math.sin(yaw_rad) * horizontal,
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * horizontal,
        )
        position = _add(anchor, _scale(back, final_distance))

        self._constraints = {"yaw": cur_yaw, "pitch": cur_pitch, "distance": cur_distance}
        return {
            "position": position,
            "look_at": anchor,
            "yaw": final_yaw,
            "pitch": final_pitch,
            "distance": final_distance,
            "soft": {"pitch": soft_pitch, "yaw": soft_yaw, "distance": soft_distance},
            "engaged": {"pitch": pitch_engaged, "yaw": yaw_engaged, "distance": distance_engaged},
            "at_limit": {
                "pitch_min": min_pitch_v is not None and final_pitch <= min_pitch_v + 1e-9,
                "pitch_max": max_pitch_v is not None and final_pitch >= max_pitch_v - 1e-9,
                "yaw_min": min_yaw_v is not None and final_yaw <= min_yaw_v + 1e-9,
                "yaw_max": max_yaw_v is not None and final_yaw >= max_yaw_v - 1e-9,
                "distance_min": min_distance_v is not None
                and final_distance <= min_distance_v + 1e-9,
                "distance_max": max_distance_v is not None
                and final_distance >= max_distance_v - 1e-9,
            },
        }
