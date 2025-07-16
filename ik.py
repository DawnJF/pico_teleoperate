from pathlib import Path
from dm_control import mjcf
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control import viewer
import mujoco.viewer
import mujoco
import numpy as np

from mujoco_robot import MujocoRobot


def print_tree(element, indent=0):
    print("  " * indent + f"{element.tag} name={getattr(element, 'name', '')}")
    for child in element.all_children():
        print_tree(child, indent + 1)


class IKProcessor:
    def __init__(self, name, xml_path):

        mjcf_root = mjcf.from_path(str(xml_path))
        # print(mjcf_root.find("body", "wrist_yaw_link"))
        # print_tree(mjcf_root)
        mjcf_root = self._add_site(mjcf_root)
        self.physics = mjcf.Physics.from_mjcf_model(mjcf_root)
        print(self.physics)
        self.name = name
        self.num_dof = 6

    def _add_site(self, mjcf_root):

        # 2. 找到末端 link 对应的 body（根据你的模型名称替换 'wrist_3_link'）
        end_effector_body = mjcf_root.find("body", "wrist_yaw_link")
        if end_effector_body is None:
            raise ValueError("找不到名为 'wrist_3_link' 的 body，请检查模型结构")

        # 3. 添加一个 site（例如 attachment_site，位置和大小可以自己调整）
        end_effector_body.add(
            "site",
            name="attachment_site",
            pos=[0, 0.1, 0],  # 相对这个body的偏移量，改成你末端中心或末端前方
            size=[0.005],  # site 的显示大小，纯可视
            rgba=[1, 0, 0, 1],  # 红色点，方便你在viewer里看到
        )
        return mjcf_root

    def _ik(self, target_pos, target_quat):
        print(self.physics.named.data.site_xpos["attachment_site"])
        print(self.physics.named.data.site_xmat["attachment_site"])

        ik_result = qpos_from_site_pose(
            self.physics,
            "attachment_site",
            target_pos=target_pos,
            target_quat=target_quat,
            tol=1e-14,
            max_steps=400,
        )

        self.physics.reset()
        if ik_result.success:
            new_qpos = ik_result.qpos[: self.num_dof]
            print(f"IK result: {ik_result.qpos}")
            return new_qpos

        return None

    def _fk(self, current_qpos):
        # run the fk
        self.physics.data.qpos[: self.num_dof] = current_qpos
        self.physics.step()
        ee_rot_mj = np.array(
            self.physics.named.data.site_xmat["attachment_site"]
        ).reshape(3, 3)
        ee_pos_mj = np.array(self.physics.named.data.site_xpos["attachment_site"])

    def process(self, data):
        # Placeholder for processing logic
        print(f"Processing data in {self.name}: {data}")


def try01():
    # Example usage
    xml_path = Path("/Users/majianfei/Projects/Works/westlake_u22/v55.xml")
    # xml_path = Path(
    #     "/Users/majianfei/Projects/Github/gello_software/third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
    # )
    ik_processor = IKProcessor(name="u22", xml_path=xml_path)

    target_position = np.array([0.2, 0.0, -0.2])
    target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Example quaternion

    ik_processor._ik(target_position, target_quaternion)


def try02():
    xml_path = Path("/Users/majianfei/Projects/Works/westlake_u22/v55.xml")
    ik_processor = IKProcessor(name="u22", xml_path=xml_path)

    target_position = np.array([0.2, 0.0, -0.2])
    target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Example quaternion

    new_qpos = ik_processor._ik(target_position, target_quaternion)
    print(new_qpos)

    robot = MujocoRobot(xml_path=str(xml_path))
    obs = robot.get_observations()
    print(obs)
    new_qpos = np.append(new_qpos, [0, 0])
    print(new_qpos)
    robot.command_joint_state(new_qpos)
    robot.serve()


if __name__ == "__main__":
    try02()
