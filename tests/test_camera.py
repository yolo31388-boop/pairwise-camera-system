"""相机系统手感修复测试（9 个）。"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camera.system import CameraSystem, EffectResult


class TestFollow:
    def test_smooth_follow(self):
        """1. 跟随使用阻尼插值，单帧不硬跳到目标，持续跟随最终收敛。"""
        cs = CameraSystem(damping=10.0)
        cs.cam.pos = [0.0, 0.0]
        cs.follow([100.0, 0.0], 0.016)
        assert 0.0 < cs.cam.pos[0] < 100.0, "相机跟随应使用平滑插值"
        for _ in range(300):
            cs.follow([100.0, 50.0], 0.016)
        assert math.isclose(cs.cam.pos[0], 100.0, abs_tol=0.5)
        assert math.isclose(cs.cam.pos[1], 50.0, abs_tol=0.5)

    def test_follow_respects_max_speed(self):
        """2. 跟随受最大速度限制。"""
        cs = CameraSystem(damping=1000.0, max_follow_speed=100.0)
        cs.cam.pos = [0.0, 0.0]
        cs.follow([10000.0, 0.0], 0.1)
        assert 0.0 < cs.cam.pos[0] <= 10.0 + 1e-9


class TestShake:
    def test_shake_decays_and_stops(self):
        """3. 震动幅度随时间衰减，到时后归零。"""
        cs = CameraSystem()
        cs.shake(10.0, 0.5)
        cs.update_shake(0.25)
        assert 0.0 < cs.cam.shake_amp < 10.0, "震动幅度应随时间衰减"
        assert math.isclose(cs.cam.shake_amp, 5.0, rel_tol=0.05)
        cs.update_shake(0.25 + 1e-6)
        assert cs.cam.shake_amp == 0.0
        assert cs.shake_offset() == [0.0, 0.0]

    def test_shake_duration_clamped_to_max(self):
        """4. 震动持续时间不能超过最大持续时间。"""
        cs = CameraSystem(max_shake_duration=1.0)
        cs.shake(10.0, 5.0)
        assert cs.cam.shake_time <= 1.0


class TestClampAngle:
    def test_soft_limit_eases_near_edge(self):
        """5. 软限制：软区内减速滞后，平滑连续地到达极限。"""
        cs = CameraSystem(soft_zone=0.2)
        # 硬区：原样通过
        assert math.isclose(cs.clamp_angle(-0.5, -1.0, 1.0), -0.5, abs_tol=1e-12)
        # 极限：恰好贴住且不越界
        assert math.isclose(cs.clamp_angle(1.0, -1.0, 1.0), 1.0, abs_tol=1e-12)
        assert cs.clamp_angle(1.5, -1.0, 1.0) == 1.0
        # 软区入口：C1 连续（函数值与斜率都与硬区一致）
        assert math.isclose(cs.clamp_angle(0.8, -1.0, 1.0), 0.8, abs_tol=1e-9)
        eps = 1e-7
        d_in = (cs.clamp_angle(0.8 + eps, -1.0, 1.0) - cs.clamp_angle(0.8 - eps, -1.0, 1.0)) / (2 * eps)
        assert math.isclose(d_in, 1.0, abs_tol=1e-6)
        # 极限处斜率平滑降为 0（不是突然卡住）
        d_edge = (cs.clamp_angle(1.0, -1.0, 1.0) - cs.clamp_angle(1.0 - eps, -1.0, 1.0)) / eps
        assert abs(d_edge) < 1e-5
        # 软区内单调、不越限，且越接近极限移动越慢
        vals = [cs.clamp_angle(0.8 + i * 0.01, -1.0, 1.0) for i in range(21)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
        assert all(v <= 1.0 + 1e-12 for v in vals)
        slopes = [(vals[i + 1] - vals[i]) / 0.01 for i in range(len(vals) - 1)]
        assert slopes[-1] < slopes[0]


class TestPath:
    def test_catmull_rom_tangent_continuous(self):
        """6. 过场路径切线连续，端点/控制点插值准确，无拐点。"""
        cs = CameraSystem()
        pts = [(0, 0), (10, 20), (20, 0), (30, 10)]
        assert cs.catmull_rom(pts, 0.0) == [0.0, 0.0]
        assert cs.catmull_rom(pts, 1.0) == [30.0, 10.0]
        assert cs.catmull_rom(pts, 1 / 3) == [10.0, 20.0]
        eps = 1e-6
        node = 1 / 3
        left = cs.catmull_rom(pts, node - eps)
        right = cs.catmull_rom(pts, node + eps)
        at = cs.catmull_rom(pts, node)
        dl = [(at[i] - left[i]) / eps for i in range(2)]
        dr = [(right[i] - at[i]) / eps for i in range(2)]
        for a, b in zip(dl, dr):
            assert math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-3)


class TestCollision:
    def test_raycast_pulls_camera_in(self):
        """7. 玩家到相机连线上有障碍时，把相机拉近到障碍表面。"""
        cs = CameraSystem()
        new_pos = cs.collide([0.0, 0.0], [(5.0, 0.0, 1.0)], cam_pos=[10.0, 0.0])
        assert math.isclose(new_pos[0], 4.0, abs_tol=0.05)
        assert math.isclose(new_pos[1], 0.0, abs_tol=1e-9)
        # 不在连线上的障碍不影响相机
        assert cs.collide([0.0, 0.0], [(5.0, 5.0, 0.5)], cam_pos=[10.0, 0.0]) == [10.0, 0.0]
        # 没有障碍时位置不变
        assert cs.collide([0.0, 0.0], [], cam_pos=[10.0, 0.0]) == [10.0, 0.0]


class TestFov:
    def test_fov_smooth_zoom(self):
        """8. FOV 指数平滑过渡，不瞬间跳变，最终收敛。"""
        cs = CameraSystem(fov_lerp_speed=8.0)
        cs.cam.fov = 60.0
        cs.set_fov(30.0)
        first = cs.update_fov(0.016)
        assert 30.0 < first < 60.0, "FOV 不应瞬间跳变"
        for _ in range(600):
            cs.update_fov(0.016)
        assert math.isclose(cs.cam.fov, 30.0, abs_tol=0.1)


class TestEffects:
    def test_layered_effects_compose_not_overwrite(self):
        """9. 多层效果基于同一基础姿态独立计算，叠加合并而非覆盖。"""
        cs = CameraSystem()
        cs.cam.pos = [10.0, 0.0]
        seen = []

        def shake_layer(cam):
            seen.append(tuple(cam.pos))
            return EffectResult(pos_offset=[2.0, 0.0])

        def collision_layer(cam):
            seen.append(tuple(cam.pos))
            return EffectResult(pos_offset=[0.0, -3.0])

        cs.add_effect(shake_layer)
        cs.add_effect(collision_layer)
        cs.apply_effects()
        assert seen == [(10.0, 0.0), (10.0, 0.0)]
        assert cs.cam.pos == [12.0, -3.0]
