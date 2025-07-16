import time
from pathlib import Path

import mujoco
import mujoco.viewer


def main():

    _MENAGERIE_ROOT: Path = Path(
        "/Users/majianfei/Projects/Github/gello_software/third_party/mujoco_menagerie"
    )

    xml = _MENAGERIE_ROOT / "franka_emika_panda/panda.xml"
    xml = _MENAGERIE_ROOT / "universal_robots_ur5e/ur5e.xml"

    xml = xml.as_posix()
    # xml = "/Users/majianfei/Projects/Github/gello_software/third_party/mujoco_menagerie/unitree_h1/h1.xml"
    xml = "/Users/majianfei/Projects/Works/westlake_u22/v55.xml"

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

                formatted = " ".join(f"{x:.3f}" for x in d.qpos)
                print(f"qpos: {formatted}")

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
