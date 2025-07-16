from pathlib import Path
from dm_control import mjcf
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control import viewer
import mujoco.viewer
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        self.name = name
        self.num_dof = 9

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
            size=[0.05],  # site 的显示大小，纯可视
            rgba=[1, 0, 1, 1],  # 红色点，方便你在viewer里看到
        )
        return mjcf_root

    def ik(self, target_pos, target_quat):
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
            return new_qpos

        return None

    def read_pose(self):
        """
        end effector pose and qpose
        ee_rot.reshape(3, 3)
        """
        ee_pos = self.physics.named.data.site_xpos["attachment_site"]
        ee_rot = self.physics.named.data.site_xmat["attachment_site"]
        qpose = self.physics.data.qpos
        return ee_pos, ee_rot, qpose

    def fk(self, current_qpos):
        # run the fk
        self.physics.data.qpos[: self.num_dof] = current_qpos
        self.physics.step()

    def process(self, data):
        # Placeholder for processing logic
        print(f"Processing data in {self.name}: {data}")


def format_rot_matrix(m):
    # 确保是 3x3 的 shape
    m = np.array(m).reshape(3, 3)
    lines = []
    for row in m:
        formatted_row = "  ".join(f"{x:.2f}" for x in row)
        lines.append(formatted_row)
    return "\n".join(lines)


def format_qpose(qpose):
    # 确保是 9 个元素
    qpose = np.array(qpose).reshape(-1)
    formatted_qpose = "  ".join(f"{x:.2f}" for x in qpose)
    return formatted_qpose


def try01():
    # Example usage
    xml_path = Path("/Users/majianfei/Projects/Works/westlake_u22/v55.xml")
    # xml_path = Path(
    #     "/Users/majianfei/Projects/Github/gello_software/third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
    # )
    ik_processor = IKProcessor(name="u22", xml_path=xml_path)
    ee_pos, ee_rot, qpose = ik_processor.read_pose()
    print(f"End Effector Position: {ee_pos}\nRotation: \n{format_rot_matrix(ee_rot)}")
    print(f"QPose: {format_qpose(qpose)}\n")

    """
    # Example of forward kinematics
    """
    # ik_processor.fk([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.65, -0.013, -0.013])
    ik_processor.fk([0.8, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.013, -0.013])

    ee_pos, ee_rot, qpose = ik_processor.read_pose()
    print(f"End Effector Position: {ee_pos}\nRotation: \n{format_rot_matrix(ee_rot)}")
    print(f"QPose: {format_qpose(qpose)}\n")

    stored_rot = ee_rot.copy()
    print()

    """
    reset
    """
    ik_processor.fk([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.013, -0.013])
    ee_pos, ee_rot, qpose = ik_processor.read_pose()
    print(f"End Effector Position: {ee_pos}\nRotation:\n{format_rot_matrix(ee_rot)}")
    print(f"QPose: {format_qpose(qpose)}\n")

    """
    Example of inverse kinematics
    """
    target_position = np.array([0.2120537, 0.31275817, -0.31000975])
    target_quaternion = R.from_matrix(stored_rot.reshape(3, 3)).as_quat()[[3, 0, 1, 2]]
    # target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Example quaternion

    print()
    print(f"ik: {target_position}, {target_quaternion}")
    print(f"stored_rot:\n{format_rot_matrix(stored_rot)}")
    qpose = ik_processor.ik(target_position, target_quaternion)
    print(f"New QPose: {format_qpose(qpose)}\n")

    """
    fk
    """
    ik_processor.fk(qpose)
    ee_pos, ee_rot, qpose = ik_processor.read_pose()
    print(f"End Effector Position: {ee_pos}\nRotation:\n{format_rot_matrix(ee_rot)}")
    print(f"QPose: {format_qpose(qpose)}\n")


def try02():
    xml_path = Path("/Users/majianfei/Projects/Works/westlake_u22/v55.xml")
    ik_processor = IKProcessor(name="u22", xml_path=xml_path)

    target_position = np.array([0.21204976, 0.31275784, -0.31002123])
    target_quaternion = np.array(
        [9.88767100e-01, -3.65374398e-07, 1.49464451e-01, 1.74056028e-07]
    )  # Example quaternion

    ee_pos, ee_rot, qpose = ik_processor.read_pose()
    print(f"End Effector Position: {ee_pos}\nRotation: \n{format_rot_matrix(ee_rot)}")
    print(f"QPose: {format_qpose(qpose)}\n")

    new_qpos = ik_processor.ik(target_position, target_quaternion)
    print(f"New QPose: {format_qpose(new_qpos)}\n")

    new_qpos = new_qpos[:8]

    robot = MujocoRobot(xml_path=str(xml_path))
    obs = robot.get_observations()
    print(obs)
    # new_qpos = np.append(new_qpos, [0, 0])
    print(f"New QPose with zeros: {format_qpose(new_qpos)}\n")
    robot.command_joint_state(new_qpos)
    robot.serve()


if __name__ == "__main__":
    # try01()
    try02()
