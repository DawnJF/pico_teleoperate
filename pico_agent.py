import numpy as np
import scipy.spatial.transform as spt
from typing import Optional, Tuple


def apply_transfer(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    # xyz can be 3dim or 4dim (homogeneous) or can be a rotation matrix
    if len(xyz) == 3:
        xyz = np.append(xyz, 1)
    return np.matmul(mat, xyz)[:3]


class SinglePicoAgent:

    PICO_U22 = np.array(
        [
            [0, 0, -1, 0],  ####################
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    PICO_U22_rot = np.array(
        [
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],  # ← Z轴反了（从 [0, 1, 0] 变为 [0, -1, 0]
            [0, 0, 0, 1],
        ]
    )

    def __init__(self, translation_scaling_factor: float = 1.0, verbose: bool = False):
        self.control_active = False
        self.translation_scaling_factor = translation_scaling_factor
        self.verbose = verbose

        # 参考位姿 - 目标对象（机器人末端执行器）
        self.reference_target_rot = None
        self.reference_target_pos = None

        # 参考位姿 - 源设备（Pico手柄）
        self.reference_source_rot = None
        self.reference_source_pos = None

        # 上一次按钮状态，用于检测按钮按下事件
        self.prev_button_state = False

    def act(
        self,
        pos: np.ndarray,
        rot_quat: np.ndarray,
        button: bool,
        current_target_pos: np.ndarray,
        current_target_rot: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        根据Pico手柄输入计算目标位姿

        Args:
            pos: Pico手柄当前位置 [x, y, z]
            rot_quat: Pico手柄当前旋转四元数 [x, y, z, w]
            button: 激活按钮状态 (True/False)
            current_target_pos: 当前目标对象位置（激活时需要）
            current_target_rot: 当前目标对象旋转四元数（激活时需要）

        Returns:
            None: 未激活时
            (target_pos, target_rot): 激活时的目标位置和旋转四元数
        """
        # 检测按钮按下事件（从False到True的转换）
        button_pressed = button and not self.prev_button_state
        self.prev_button_state = button

        if button_pressed:
            if not self.control_active:
                # 激活控制：记录当前的参考位姿
                self._update_reference(pos, rot_quat)
                self._update_target(current_target_pos, current_target_rot)
                self.control_active = True
                if self.verbose:
                    print("Control activated!")
                    print(
                        f"reference: {pos}, {rot_quat} {current_target_pos} {current_target_rot}"
                    )
            else:
                # 停用控制
                self.control_active = False
                if self.verbose:
                    print("Control deactivated!")

        if not self.control_active:
            # 未激活状态：持续更新手柄参考位置，不更新目标参考位置
            self._update_reference(pos, rot_quat)
            return current_target_pos, spt.Rotation.from_quat(
                current_target_rot
            ).as_euler("xyz", degrees=False)

        return self._compute_target_pose(pos, rot_quat)

    def _update_reference(
        self,
        pos: np.ndarray,
        rot_quat: np.ndarray,
    ):
        # skip 0
        if np.all(rot_quat == 0):
            return
        """持续更新手柄参考位姿（未激活时）"""
        self.reference_source_pos = pos.copy()
        self.reference_source_rot = spt.Rotation.from_quat(rot_quat)
        # print(f"xyz: {self.reference_source_pos}")
        # print(f"RPY: {self.reference_source_rot.as_euler('xyz', degrees=False)}")

    def _update_target(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
    ):
        # 记录目标对象的参考位姿（手柄参考位姿已经在持续更新）
        self.reference_target_pos = target_pos.copy()
        self.reference_target_rot = spt.Rotation.from_quat(target_rot)

    def _compute_target_pose(
        self, pos: np.ndarray, rot_quat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算目标位姿"""

        # 确保参考位姿已经设置
        if (
            self.reference_source_pos is None
            or self.reference_source_rot is None
            or self.reference_target_pos is None
            or self.reference_target_rot is None
        ):
            raise ValueError("Reference poses not set. Cannot compute target pose.")

        # 计算手柄相对于参考位置的变化
        delta_pos = pos - self.reference_source_pos
        # print("Delta position:", delta_pos)

        current_source_rot = spt.Rotation.from_quat(rot_quat)
        delta_rot = self.reference_source_rot.inv() * current_source_rot

        delta_pos_on_target = (
            apply_transfer(self.PICO_U22, delta_pos) * self.translation_scaling_factor
        )

        pico_to_target_rot = spt.Rotation.from_matrix(self.PICO_U22_rot[:3, :3])
        delta_rot_on_target = pico_to_target_rot * delta_rot * pico_to_target_rot.inv()

        # 计算目标位姿：参考位姿 + 相对变化
        target_pos = self.reference_target_pos + delta_pos_on_target
        target_rot = self.reference_target_rot * delta_rot_on_target

        # print(f"Target position: {target_rot.as_euler('xyz', degrees=False)}")

        return target_pos, target_rot.as_euler("xyz", degrees=False)

    def get_control_status(self) -> bool:
        """获取控制状态"""
        return self.control_active

    def reset(self):
        """重置Agent状态"""
        self.control_active = False
        self.prev_button_state = False
        self.reference_source_pos = None
        self.reference_source_rot = None
        self.reference_target_pos = None
        self.reference_target_rot = None
        if self.verbose:
            print("Agent reset!")


if __name__ == "__main__":

    delta = [-0.005158, -0.001008, -0.105675]
    r = apply_transfer(SinglePicoAgent.PICO_U22, delta)

    # 创建Agent
    agent = SinglePicoAgent(translation_scaling_factor=1.5, verbose=True)

    # 当前机器人位姿
    current_robot_pos = np.array([0.5, 0.2, 0.8])
    current_robot_rot = np.array([0, 0, 0, 1])

    def get_pico_data():
        """模拟获取Pico数据的函数"""
        # 这里应该是实际的Pico设备接口
        pico_pos = np.array([0.0, 0.0, 0.0])
        pico_quat = np.array([0, 0, 0, 1])
        trigger_button = False
        return pico_pos, pico_quat, trigger_button

    def send_to_robot(pos, rot):
        """模拟发送命令到机器人的函数"""
        # 这里应该是实际的机器人控制接口
        print(f"Robot command: pos={pos}, rot={rot}")

    while True:
        # 从Pico设备获取数据
        pico_pos, pico_quat, trigger_button = get_pico_data()

        # Agent处理
        result = agent.act(
            pos=pico_pos,
            rot_quat=pico_quat,
            button=trigger_button,
            current_target_pos=current_robot_pos,
            current_target_rot=current_robot_rot,
        )

        if result is not None:
            # 激活状态：发送新的目标位姿
            target_pos, target_rot = result
            send_to_robot(target_pos, target_rot)
            current_robot_pos = target_pos
            current_robot_rot = target_rot
            print(f"Moving to: {target_pos}")
        else:
            # 未激活状态：持续更新参考位置，不发送命令
            print("Updating reference position...")

        # 实际使用中应该有适当的循环控制
        break  # 避免无限循环
