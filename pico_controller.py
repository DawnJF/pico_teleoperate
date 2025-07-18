# from wl_robot_python_sdk.teleoperate.pico_agent import SinglePicoAgent
import time
from pico_agent import SinglePicoAgent
import enet
import numpy as np
from scipy.spatial.transform import Rotation
import pyautogui


def parse_event_string(data: str) -> tuple[int, dict]:
    """
    解析由 C++ 用逗号拼接的 ENet 数据包:
    前 7 项是 left.hand
    中间 7 项是 right.hand
    最后 4 项是 button 状态
    """
    try:
        text = data.strip().rstrip("\x00").rstrip(",")  # 防止末尾多一个逗号
        message = text.split(":")
        time = message[0]
        parts = message[1].split(",")

        if len(parts) != 20:
            print(f"Unexpected data length: {len(parts)} (expected 20)")
            return 0, {}

        parts = list(map(float, parts))  # 所有字段都是数字（float/bool）

        result = {
            "left_hand": np.array(parts[:8]),
            "right_hand": np.array(parts[8:16]),
            "left_click_x": bool(parts[16]),
            "left_click_y": bool(parts[17]),
            "right_click_a": bool(parts[18]),
            "right_click_b": bool(parts[19]),
        }

        return int(time), result

    except Exception as e:
        print("Failed to parse data:", e)
        return 0, {}


class PicoController:
    def __init__(self, manager, pose_cmd, verbose: bool = False):
        self.left_agent = SinglePicoAgent("l", verbose=verbose)
        self.right_agent = SinglePicoAgent("r", verbose=verbose)
        self.verbose = verbose
        self.manager = manager
        self.pose_cmd = pose_cmd

        self.time = 0
        # Add frequency tracking
        self.data_count = 0
        self.last_print_time = time.time()
        # Add process timing tracking
        self.process_times = []
        self.total_process_time = 0.0

        self.last_button_x = False
        self.last_button_y = False

    def run(self):
        """
        启动 ENet thread
        """
        self.running = True
        self._run()

    def stop(self):
        """
        停止 ENet thread
        """
        self.running = False

    def act(self, pose_state, data):
        left_pose = pose_state.state[0]
        left_xyz = np.array(
            [left_pose.position_x, left_pose.position_y, left_pose.position_z]
        )
        left_euler = np.array([left_pose.roll, left_pose.pitch, left_pose.yaw])

        right_pose = pose_state.state[1]
        right_xyz = np.array(
            [
                right_pose.position_x,
                right_pose.position_y,
                right_pose.position_z,
            ]
        )
        right_euler = np.array([right_pose.roll, right_pose.pitch, right_pose.yaw])

        if data is None:
            return (left_xyz, left_euler), (right_xyz, right_euler), 0.0, 0.0

        button_state = data["right_click_a"]
        data_left_quat = data["left_hand"][3:7]
        data_right_quat = data["right_hand"][3:7]
        try:
            data_left_rpy = Rotation.from_quat(data_left_quat).as_euler(
                "ZYX", degrees=False
            )
            data_right_rpy = Rotation.from_quat(data_right_quat).as_euler(
                "ZYX", degrees=False
            )
        except ValueError as e:
            data_left_rpy = np.zeros(3)
            data_right_rpy = np.zeros(3)

        left_cmd = self.left_agent.act(
            data["left_hand"][:3],
            data_left_rpy,
            button_state,
            left_xyz,
            left_euler,
        )

        right_cmd = self.right_agent.act(
            data["right_hand"][:3],
            data_right_rpy,
            button_state,
            right_xyz,
            right_euler,
        )

        if self.verbose and self.data_count % 50 == 0:
            # print(f"left_xyz: {left_xyz}")
            # print(f"ref: {self.left_agent.reference_source_pos}")
            # print(f"ref: {self.left_agent.reference_target_pos}")
            # print(f"return left_cmd: {left_cmd}")

            print()
            print(f"data_right_rpy: {data_right_rpy}")
            print(f"right_cmd: {right_cmd}")
            print(f"data_left_rpy: {data_left_rpy}")
            print(f"left_cmd: {left_cmd}")

        return (
            left_cmd,
            right_cmd,
            float(data["left_hand"][7]),
            float(data["right_hand"][7]),
        )

    def process_xy(self, data):
        button_x = data["left_click_x"]
        button_y = data["left_click_y"]
        if button_x and not self.last_button_x:
            # 按下 X 键
            print("X button pressed, trigger b key")
            pyautogui.press("b")

        if button_y and not self.last_button_y:
            # 按下 Y 键
            print("Y button pressed, trigger e key")
            pyautogui.press("e")
        self.last_button_x = button_x
        self.last_button_y = button_y

    def process(self, data):
        start_time = time.time()

        pose_state = self.manager.get_pose_state()
        if pose_state is None:
            return
        left_cmd, right_cmd, left_trigger, right_trigger = self.act(pose_state, data)

        pose_cmd = self.pose_cmd

        pose_cmd.cmd[0].opening_state = left_trigger  # 打开左夹爪
        pose_cmd.cmd[0].force = 0.0  # 设置左夹爪要施加的力
        pose_cmd.cmd[0].position_x = left_cmd[0][0]  # 左夹爪末端中心坐标x
        pose_cmd.cmd[0].position_y = left_cmd[0][1]  # 左夹爪末端中心坐标y
        pose_cmd.cmd[0].position_z = left_cmd[0][2]  # 左夹爪末端中心坐标z
        pose_cmd.cmd[0].roll = left_cmd[1][0]  # 左夹爪旋转方向roll
        pose_cmd.cmd[0].pitch = left_cmd[1][1]  # 左夹爪旋转方向pitch
        pose_cmd.cmd[0].yaw = left_cmd[1][2]  # 左夹爪旋转方向yaw

        pose_cmd.cmd[1].opening_state = right_trigger  # 打开右夹爪
        pose_cmd.cmd[1].force = 0.0  # 设置右夹爪要施加的力
        pose_cmd.cmd[1].position_x = right_cmd[0][0]  # 右夹爪末端中心坐标x
        pose_cmd.cmd[1].position_y = right_cmd[0][1]  # 右夹爪末端中心坐标y
        pose_cmd.cmd[1].position_z = right_cmd[0][2]  # 右夹爪末端中心坐标z
        pose_cmd.cmd[1].roll = right_cmd[1][0]  # 右夹爪旋转方向roll
        pose_cmd.cmd[1].pitch = right_cmd[1][1]  # 右夹爪旋转方向pitch
        pose_cmd.cmd[1].yaw = right_cmd[1][2]  # 右夹爪旋转方向yaw

        # # 发送位姿控制指令(高层控制，用于控制夹爪末端中心位置的运动)
        self.manager.high_level_control(pose_cmd)

        end_time = time.time()
        process_time = end_time - start_time
        self.total_process_time += process_time

    def _run(self):
        host = enet.Host(enet.Address(b"0.0.0.0", 2233), 32, 2, 0, 0)

        print("ENet server listening on port 2233...")

        while self.running:
            event = host.service(1000)
            # print(f"ENet event type: {event.type}")
            if event.type == enet.EVENT_TYPE_CONNECT:
                print(f"Client connected from {event.peer.address}.")
                print(f"  -> Peer round trip time: {event.peer.roundTripTime}")
            elif event.type == enet.EVENT_TYPE_RECEIVE:
                time_val, data = parse_event_string(event.packet.data.decode())

                self.data_count += 1
                current_time = time.time()

                # Check if 2 seconds have passed since last print
                if current_time - self.last_print_time >= 2.0:
                    elapsed = current_time - self.last_print_time
                    frequency = self.data_count / elapsed
                    avg_process_time = (
                        (self.total_process_time / self.data_count * 1000)
                        if self.data_count > 0
                        else 0
                    )
                    print(
                        f"Data reception frequency: {frequency:.2f} Hz, Avg process time: {avg_process_time:.2f} ms"
                    )
                    self.data_count = 0
                    self.last_print_time = current_time
                    self.total_process_time = 0.0

                if self.time == 0:
                    self.time = time_val
                if time_val < self.time:
                    print(f"Received old data: {time_val} < {self.time}, ignoring.")
                    continue

                self.process_xy(data)
                self.process(data)

            elif event.type == enet.EVENT_TYPE_DISCONNECT:
                print(f"Client disconnected.")
                print(f"  -> Peer state: {event.peer.state}")
                print(f"  -> Peer last receive time: {event.peer.lastReceiveTime}")
                print(f"  -> Peer round trip time: {event.peer.roundTripTime}")


def test():
    """
    Test function for PicoController
    """

    class CMD:
        def __init__(self):
            self.opening_state = 0
            self.force = 0.0
            self.position_x = 0.0
            self.position_y = 0.0
            self.position_z = 0.0
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0

    class PoseCmd:
        def __init__(self):
            # Simulate the command structure
            self.cmd = [CMD(), CMD()]

    class PoseState:
        def __init__(self):
            self.state = [STATE(), STATE()]

    class STATE:
        def __init__(self):
            self.position_x = 0.0
            self.position_y = 0.0
            self.position_z = 0.0
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0

    class PicoManager:
        def __init__(self):
            self.pose_state = PoseState()
            self.count = 0

        def get_pose_state(self):
            """
            left_pose = pose_state.state[1]
            left_xyz = [left_pose.position_x, left_pose.position_y, left_pose.position_z]
            """
            # Simulate getting pose state
            return self.pose_state

        def high_level_control(self, pose_cmd):
            if self.count % 100 == 0:
                print(
                    f"high_level_control: {pose_cmd.cmd[0].position_x}, {pose_cmd.cmd[1].position_x}"
                )
            self.count += 1

    manager = PicoManager()
    pose_cmd = PoseCmd()

    controller = PicoController(manager, pose_cmd, verbose=True)
    controller.run()


if __name__ == "__main__":
    # controller = PicoController(verbose=True)
    # thread = controller.run()

    # # sleep 1 min
    # time.sleep(300)

    test()
