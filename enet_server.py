import enet
import numpy as np

from pico_agent import SinglePicoAgent


def main():

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
            data = parse_event_string(event.packet.data.decode())

            result = agent.act(
                data["left_hand"][:3],
                data["left_hand"][3:7],
                data["right_click_a"],
                np.array([0.1, 0.0, 0.0]),
                np.array([0, 0, 0, 1]),
            )
            # print("Agent result:", result[1])

        elif event.type == enet.EVENT_TYPE_DISCONNECT:
            print(f"Client disconnected.")
            print(f"  -> Peer state: {event.peer.state}")
            print(f"  -> Peer last receive time: {event.peer.lastReceiveTime}")
            print(f"  -> Peer round trip time: {event.peer.roundTripTime}")


def parse_event_string(data: str) -> dict:
    """
    解析由 C++ 用逗号拼接的 ENet 数据包:
    前 7 项是 left.hand
    中间 7 项是 right.hand
    最后 4 项是 button 状态
    """
    try:
        text = data.strip().rstrip("\x00").rstrip(",")  # 防止末尾多一个逗号
        parts = text.split(",")

        if len(parts) != 18:
            print(f"Unexpected data length: {len(parts)} (expected 18)")
            return None

        parts = list(map(float, parts))  # 所有字段都是数字（float/bool）

        result = {
            "left_hand": np.array(parts[:7]),
            "right_hand": np.array(parts[7:14]),
            "left_click_x": bool(parts[14]),
            "left_click_y": bool(parts[15]),
            "right_click_a": bool(parts[16]),
            "right_click_b": bool(parts[17]),
        }

        return result

    except Exception as e:
        print("Failed to parse data:", e)
        return None


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
