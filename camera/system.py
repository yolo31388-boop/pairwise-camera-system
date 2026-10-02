"""相机视角系统 - 含5个bug"""
from dataclasses import dataclass, field

@dataclass
class Camera:
    position: list
    target: list
    fov: float = 60.0

class CameraSystem:
    def __init__(self):
        self.camera = Camera([0, 5, -10], [0, 0, 0])
        self.locked_target: dict = None  # bug3: 目标死了还锁

    def update_camera(self, player_pos: list, obstacles: list) -> None:
        # bug1: 不做碰撞检测
        self.camera.position = [player_pos[0], player_pos[1] + 5, player_pos[2] - 10]

    def smooth_camera(self, target_pos: list, delta: float) -> None:
        # bug2: 线性插值，帧率相关
        for i in range(3):
            self.camera.position[i] += (target_pos[i] - self.camera.position[i]) * 0.1

    def lock_on_target(self, target: dict) -> None:
        # bug3: 不跟随玩家，不检查存活
        self.locked_target = target
        self.camera.target = target.get("pos", [0,0,0])

    def handle_occlusion(self, obstacles: list) -> None:
        # bug4: 只拉近，不透明化
        for obs in obstacles:
            self.camera.position[1] -= 1

    def process_camera_input(self, dx: float, dy: float) -> None:
        # bug5: 不平滑，不限制角度
        self.camera.target[0] += dx
        self.camera.target[1] += dy
