import enet
import numpy as np
import time
from pico_agent import SinglePicoAgent
from pico_controller import parse_event_string

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


class RealTimePlotter:
    def __init__(self, max_len=100):
        plt.ion()
        self.max_len = max_len
        self.xs, self.ys, self.zs = [], [], []

        self.fig, self.ax = plt.subplots()
        self.scatter_x = None
        self.scatter_y = None
        self.scatter_z = None

        self.ax.set_xlim(0, max_len)
        self.ax.set_ylim(-1, 1)
        self.ax.set_title("Real-time Position (Scatter)")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Position")
        self.ax.legend(["x", "y", "z"])

        self.annotations = []

    def update(self, xyz):
        self.xs.append(xyz[0])
        self.ys.append(xyz[1])
        self.zs.append(xyz[2])

        if len(self.xs) > self.max_len:
            self.xs = self.xs[-self.max_len :]
            self.ys = self.ys[-self.max_len :]
            self.zs = self.zs[-self.max_len :]

        steps = list(range(len(self.xs)))

        # 删除旧的散点图
        if self.scatter_x:
            self.scatter_x.remove()
        if self.scatter_y:
            self.scatter_y.remove()
        if self.scatter_z:
            self.scatter_z.remove()

        # 画新的点
        self.scatter_x = self.ax.scatter(steps, self.xs, color="r", label="x")
        self.scatter_y = self.ax.scatter(steps, self.ys, color="g", label="y")
        self.scatter_z = self.ax.scatter(steps, self.zs, color="b", label="z")

        # 动态调整 y 轴范围
        all_vals = self.xs + self.ys + self.zs
        ymin, ymax = min(all_vals), max(all_vals)
        margin = 0.05 * (ymax - ymin + 1e-5)
        self.ax.set_ylim(ymin - margin, ymax + margin)

        # 清除旧的标注
        for ann in self.annotations:
            ann.remove()
        self.annotations.clear()

        # 添加新的标注（最后一个点）
        last_step = steps[-1]
        ann_x = self.ax.annotate(
            f"{self.xs[-1]:.2f}",
            (last_step, self.xs[-1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color="r",
            fontsize=8,
        )
        ann_y = self.ax.annotate(
            f"{self.ys[-1]:.2f}",
            (last_step, self.ys[-1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color="g",
            fontsize=8,
        )
        ann_z = self.ax.annotate(
            f"{self.zs[-1]:.2f}",
            (last_step, self.zs[-1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color="b",
            fontsize=8,
        )

        self.annotations.extend([ann_x, ann_y, ann_z])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():

    data_list = []
    data_count = 0
    last_print_time = time.time()

    agent = SinglePicoAgent(verbose=True)

    host = enet.Host(enet.Address(b"0.0.0.0", 2233), 32, 2, 0, 0)
    print("ENet server listening on port 2233...")

    while True:
        event = host.service(1000)
        if event.type == enet.EVENT_TYPE_CONNECT:
            print(f"Client connected from {event.peer.address}.")
            print(f"  -> Peer round trip time: {event.peer.roundTripTime}")
        elif event.type == enet.EVENT_TYPE_RECEIVE:
            # print(f"Received data: {event.packet.data.decode()}")
            t, data = parse_event_string(event.packet.data.decode())
            # print(f"Received data: {data}")
            if data is None:
                # print("No valid data received.")
                continue

            if data["right_click_a"]:
                print("Right click detected")

            xyz, rot = agent.act(
                data["right_hand"][:3],
                data["right_hand"][3:7],
                data["right_click_a"],
                np.array([0, 0.0, 0.0]),
                np.array([0, 0, 0, 1]),
            )
            data_list.append(xyz)
            if len(data_list) > 5 * 90:
                break

            # print("Agent result:", xyz)
            current_time = time.time()
            data_count += 1
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - last_print_time
                frequency = data_count / elapsed
                print(
                    f"Data reception frequency: {frequency:.2f} Hz ({data_count} packets in {elapsed:.1f}s)"
                )
                data_count = 0
                last_print_time = current_time

        elif event.type == enet.EVENT_TYPE_DISCONNECT:
            print(f"Client disconnected.")
            print(f"  -> Peer state: {event.peer.state}")
            print(f"  -> Peer last receive time: {event.peer.lastReceiveTime}")
            print(f"  -> Peer round trip time: {event.peer.roundTripTime}")

    p = RealTimePlotter()
    for xyz in data_list:
        p.update(xyz)
        time.sleep(0.01)  # 模拟实时更新


def test():
    # xyab
    msg = "-0.252687,-0.0824492,-0.284689,0.18412,-0.056925,-0.410397,0.89131,0.12486,-0.07093,-0.277762,0.225583,0.143015,0.392541,0.880097,0,0,1,0"
    data = parse_event_string(msg)

    agent = SinglePicoAgent(translation_scaling_factor=1.5, verbose=True)

    result = agent.act(
        data["left_hand"][:3],
        data["left_hand"][3:7],
        data["right_click_a"],
        np.array([0.1, 0.0, 0.0]),
        np.array([0, 0, 0, 1]),
    )

    print("Agent result:", result)


if __name__ == "__main__":
    main()
    # test()
