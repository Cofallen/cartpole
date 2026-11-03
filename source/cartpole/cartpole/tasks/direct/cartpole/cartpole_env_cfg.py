# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuatorCfg  # by user


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 1
    observation_space = 5
    state_space = 0

    # by user
    target_range = [-2.0, 2.0]  # 目标采样区间
    rew_scale_target = -2.0     # 追踪惩罚

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # by user
    # robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(
    #     prim_path="/World/envs/env_.*/Robot",
    #     actuators={
    #         "cart_actuator": ImplicitActuatorCfg(
    #             joint_names_expr=["slider_to_cart"],
    #             effort_limit_sim=400.0,
    #             stiffness=400.0,     # 原来是 0（纯力矩）
    #             damping=40.0,        # 原来是 10
    #         ),
    #         "pole_actuator": ImplicitActuatorCfg(
    #             joint_names_expr=["cart_to_pole"],
    #             effort_limit_sim=400.0,
    #             stiffness=0.0,       # 杆是被动的，保持 0
    #             damping=0.0,
    #         ),
    #     },
    # )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = 200.0  # [N]
    # action_scale = 0.5 # by user [m] 位置幅值

    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -1.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.001
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.1, 0.1]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]