# from wl_robot_python_sdk.teleoperate.pico_agent import SinglePicoAgent
import time
from pico_agent import SinglePicoAgent
import enet
import numpy as np
import threading
import scipy.spatial.transform as spt


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
    def __init__(self, verbose: bool = False):
        self.left_agent = SinglePicoAgent(verbose=verbose)
        self.left_state = None
        self.running = False
        self.lock = threading.Lock()
        self.time = 0
        # Add frequency tracking
        self.data_count = 0
        self.last_print_time = time.time()

    def run(self):
        """
        启动 ENet thread
        """
        self.running = True

        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

        return thread

    def stop(self):
        """
        停止 ENet thread
        """
        self.running = False

    def act(self, pose_state):
        left_pose = pose_state.state[1]
        left_xyz = [left_pose.position_x, left_pose.position_y, left_pose.position_z]
        left_euler = [left_pose.roll, left_pose.pitch, left_pose.yaw]
        left_quat = spt.Rotation.from_euler("xyz", left_euler).as_quat()
        with self.lock:
            data = self.left_state

        if data is None:
            return left_xyz, left_euler
        return self.left_agent.act(
            data["right_hand"][:3],
            data["right_hand"][3:7],
            data["right_click_a"],
            left_xyz,
            left_quat,
        )

    def _run(self):
        host = enet.Host(enet.Address(b"0.0.0.0", 2233), 32, 2, 0, 0)

        print("ENet server listening on port 2233...")

        while self.running:
            event = host.service(1000)
            if event.type == enet.EVENT_TYPE_CONNECT:
                print(f"Client connected from {event.peer.address}.")
                print(f"  -> Peer round trip time: {event.peer.roundTripTime}")
            elif event.type == enet.EVENT_TYPE_RECEIVE:
                time_val, data = parse_event_string(event.packet.data.decode())

                with self.lock:
                    self.data_count += 1
                    current_time = time.time()

                    # Check if 2 seconds have passed since last print
                    if current_time - self.last_print_time >= 2.0:
                        elapsed = current_time - self.last_print_time
                        frequency = self.data_count / elapsed
                        print(
                            f"Data reception frequency: {frequency:.2f} Hz ({self.data_count} packets in {elapsed:.1f}s)"
                        )
                        self.data_count = 0
                        self.last_print_time = current_time

                if self.time == 0:
                    self.time = time_val
                if time_val < self.time:
                    print(f"Received old data: {time_val} < {self.time}, ignoring.")
                    continue

                with self.lock:
                    self.left_state = data

            elif event.type == enet.EVENT_TYPE_DISCONNECT:
                print(f"Client disconnected.")
                print(f"  -> Peer state: {event.peer.state}")
                print(f"  -> Peer last receive time: {event.peer.lastReceiveTime}")
                print(f"  -> Peer round trip time: {event.peer.roundTripTime}")


class PicoControllerV2:
    def __init__(self, manager, pose_cmd, verbose: bool = False):
        self.left_agent = SinglePicoAgent(verbose=verbose)
        self.manager = manager
        self.pose_cmd = pose_cmd
        self.time = 0
        # Add frequency tracking
        self.data_count = 0
        self.last_print_time = time.time()

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
        left_pose = pose_state.state[1]
        left_xyz = [left_pose.position_x, left_pose.position_y, left_pose.position_z]
        left_euler = [left_pose.roll, left_pose.pitch, left_pose.yaw]
        left_quat = spt.Rotation.from_euler("xyz", left_euler).as_quat()

        if data is None:
            return left_xyz, left_euler
        return self.left_agent.act(
            data["right_hand"][:3],
            data["right_hand"][3:7],
            data["right_click_a"],
            left_xyz,
            left_quat,
        )

    def process(self, data):
        pose_state = self.manager.get_pose_state()
        right_cmd = self.act(pose_state, data)

        pose_cmd = self.pose_cmd

        pose_cmd.cmd[0].opening_state = 1  # 打开左夹爪
        pose_cmd.cmd[0].force = 0.0  # 设置左夹爪要施加的力
        pose_cmd.cmd[0].position_x = 0.3865504860877991  # 左夹爪末端中心坐标x
        pose_cmd.cmd[0].position_y = 0.22568735480308533  # 左夹爪末端中心坐标y
        pose_cmd.cmd[0].position_z = -0.3093690574169159  # 左夹爪末端中心坐标z
        pose_cmd.cmd[0].roll = 0.0  # 左夹爪旋转方向roll
        pose_cmd.cmd[0].pitch = 0.0  # 左夹爪旋转方向pitch
        pose_cmd.cmd[0].yaw = 0.0  # 左夹爪旋转方向yaw

        pose_cmd.cmd[1].opening_state = 1  # 打开右夹爪
        pose_cmd.cmd[1].force = 0.0  # 设置右夹爪要施加的力
        pose_cmd.cmd[1].position_x = right_cmd[0][0]  # 右夹爪末端中心坐标x
        pose_cmd.cmd[1].position_y = right_cmd[0][1]  # 右夹爪末端中心坐标y
        pose_cmd.cmd[1].position_z = right_cmd[0][2]  # 右夹爪末端中心坐标z
        # pose_cmd.cmd[1].position_x = 0.385504860877991    # 左夹爪末端中心坐标x
        # pose_cmd.cmd[1].position_y = -0.22568735480308533    # 左夹爪末端中心坐标y
        # pose_cmd.cmd[1].position_z = -0.3093690574169159    # 左夹爪末端中心坐标z
        pose_cmd.cmd[1].roll = 0.0  # 右夹爪旋转方向roll
        pose_cmd.cmd[1].pitch = 0.0  # 右夹爪旋转方向pitch
        pose_cmd.cmd[1].yaw = 0.0  # 右夹爪旋转方向yaw

        # # 发送位姿控制指令(高层控制，用于控制夹爪末端中心位置的运动)
        self.manager.high_level_control(pose_cmd)

    def _run(self):
        host = enet.Host(enet.Address(b"0.0.0.0", 2233), 32, 2, 0, 0)

        print("ENet server listening on port 2233...")

        while self.running:
            event = host.service(1000)
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
                    print(
                        f"Data reception frequency: {frequency:.2f} Hz ({self.data_count} packets in {elapsed:.1f}s)"
                    )
                    self.data_count = 0
                    self.last_print_time = current_time

                if self.time == 0:
                    self.time = time_val
                if time_val < self.time:
                    print(f"Received old data: {time_val} < {self.time}, ignoring.")
                    continue

                self.process(data)

            elif event.type == enet.EVENT_TYPE_DISCONNECT:
                print(f"Client disconnected.")
                print(f"  -> Peer state: {event.peer.state}")
                print(f"  -> Peer last receive time: {event.peer.lastReceiveTime}")
                print(f"  -> Peer round trip time: {event.peer.roundTripTime}")


if __name__ == "__main__":
    controller = PicoController(verbose=True)
    thread = controller.run()

    # sleep 1 min
    time.sleep(300)
