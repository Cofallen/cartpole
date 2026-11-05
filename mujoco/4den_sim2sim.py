import numpy as np
import mujoco
import mujoco.viewer
import torch
import time

ACTION_SCALE = 100.0        # 训练时的 action_scale [N]
CTRL_DT = 1.0 / 60.0        # = sim.dt(1/120) × decimation(2)  # 没有用
TARGET = 1.0                # 部署时由你给的实参 x*
MJCF = "mujoco/pendulum.xml"
POLICY = "logs/rsl_rl/cartpole_direct/2026-10-31_17-23-01/exported/policy.pt"


class PendulumSim:
    def __init__(self, mjcf=MJCF, policy=POLICY):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(mjcf)
        self.data = mujoco.MjData(self.model)

        self.slider_to_cart_id = self.model.joint("slider_to_cart").id
        self.pendulum_joint_id = self.model.joint("pendulum_joint").id
        self.cart_force_id = self.model.actuator("cart_force").id

        self.policy = torch.jit.load(policy)
        self.target = TARGET
        self.dt = 0.002
        self.frame_skip = 8  # 8 倍速

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.slider_to_cart_id] = 0.0
        self.data.qpos[self.pendulum_joint_id] = 0  # ?
        self.data.qvel[self.slider_to_cart_id] = 0.0
        self.data.qvel[self.pendulum_joint_id] = 0.0

    def _get_obs(self):
        obs = np.array([self.data.qpos[self.pendulum_joint_id],
                        self.data.qvel[self.pendulum_joint_id],
                        self.data.qpos[self.slider_to_cart_id] - self.target,
                        self.data.qvel[self.slider_to_cart_id]], dtype=np.float32)
        return obs

    def step(self):
        obs = self._get_obs()
        action = self.policy(torch.from_numpy(obs).unsqueeze(0)).item()
        self.data.ctrl[self.cart_force_id] = action * ACTION_SCALE
        # mujoco.mj_step(self.model, self.data)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return obs, action


def main():
    sim = PendulumSim()
    sim.reset()

    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    next_tick = time.time() + sim.dt

    while viewer.is_running():
        # sim.target = 1.0 * np.sin(2 * np.pi * 0.1 * sim.data.time)
        obs, action = sim.step()
        next_tick += sim.dt
        time.sleep(sim.dt)
        viewer.sync()

if __name__ == "__main__":
    main()