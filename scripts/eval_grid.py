# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T2.3 评估脚本：把目标位置逐个钉死，统计策略的到位率。

为什么不看训练 reward：reward 里混着 clamp、权重和各种负项，读不出"离目标多远"。
这里直接看几何量 |x - target|，才是 tracking 任务真正关心的指标。

把目标钉死的技巧：把 `cfg.target_range` 改成 `[t, t]`。
`_reset_idx` 里 `sample_uniform(t, t)` 每次采出来都是 `t`，
所以任何一次 reset（包括中途摔倒后的自动复位）目标都还是 `t`，
完全不用改环境代码。

用法::

    python scripts/eval_grid.py \
        --task=Template-Cartpole-Track-Direct-v0 \
        --policy logs/rsl_rl/cartpole_direct/<run>/exported/policy.pt \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="T2.3 tracking 评估：逐个目标统计到位率。")
parser.add_argument("--task", type=str, default="Template-Cartpole-Track-Direct-v0", help="已注册的任务 id")
parser.add_argument("--num_envs", type=int, default=64, help="并行环境数（评估用小一点即可）")
parser.add_argument(
    "--targets",
    type=float,
    nargs="+",
    default=[-2.0, -1.0, 0.0, 1.0, 2.0],
    help="要测的目标位置列表 [m]",
)
parser.add_argument("--tolerance", type=float, default=0.1, help="判为「到位」的距离阈值 [m]")
parser.add_argument("--episodes", type=int, default=5, help="每个目标跑几个回合")
parser.add_argument(
    "--policy", type=str, required=True, help="导出的 TorchScript 策略（play.py 生成 .../exported/policy.pt）"
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# append AppLauncher cli args（提供 --headless / --device 等）
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401

import cartpole.tasks  # noqa: F401


def load_policy(path: str, device: str) -> torch.jit.ScriptModule:
    """读导出的 TorchScript 策略；导出的模块已经把 obs normalizer 包进去了。"""
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}\n先跑 scripts/rsl_rl/play.py 生成 .../exported/policy.pt")
    return torch.jit.load(path, map_location=device).eval()


def main() -> None:
    """逐个目标评估到位率。"""
    # 环境配置：num_envs / device 由 parse_env_cfg 从命令行填进来
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 直接摸底层 env：拿关节索引、每回合步数、设备
    base = env.unwrapped
    policy = load_policy(args_cli.policy, base.device)
    cart_idx = base._cart_dof_idx[0]
    steps_per_episode = int(base.max_episode_length)

    print(f"[INFO] policy        : {args_cli.policy}")
    print(f"[INFO] 每回合步数     : {steps_per_episode}（= episode_length_s / step_dt）")
    print(f"[INFO] 到位阈值       : ±{args_cli.tolerance} m")
    print(f"[INFO] 每个目标回合数 : {args_cli.episodes}")
    print()
    print("hit_rate = 到位步数占比；alive = 杆没倒且没超时的步数占比；mean_err = 平均 |x - target|")
    print(f"{'target':>7} {'hit_rate':>9} {'alive':>7} {'mean_err':>9}")

    # 注意这里用 no_grad 而不是 inference_mode：
    #   inference_mode 会把仿真内部 buffer 标记成 inference tensor，
    #   之后 env.reset() 往里 inplace 写就会报
    #   "Inplace update to inference tensor outside InferenceMode is not allowed"。
    #   仓库里的 play.py 之所以能用 inference_mode，是因为它只在最后 reset 一次、之后不再 reset。
    #   我们要反复 reset，所以必须用 no_grad（它只关梯度，不产生 inference tensor）。
    hit_rates = []
    for target in args_cli.targets:
        # 关键一行：把采样区间压成一个点，之后每次 reset 目标都是 target。
        # _reset_idx 里 sample_uniform(t, t) 恒等于 t，摔倒自动复位也不会跑掉。
        env_cfg.target_range = [target, target]

        n_hit = n_alive = n_total = 0
        err_sum = 0.0

        for _ in range(args_cli.episodes):
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"]
            for _ in range(steps_per_episode):
                with torch.no_grad():
                    actions = policy(obs)                                       # obs -> action
                    obs_dict, _, terminated, truncated, _ = env.step(actions)   # 环境推进一步
                obs = obs_dict["policy"]

                # 位置误差：每环境一个标量 [num_envs]
                err = (base.joint_pos[:, cart_idx] - target).abs()

                n_hit += (err < args_cli.tolerance).sum().item()        # 到位
                n_alive += (~(terminated | truncated)).sum().item()     # 杆还站着 / 没超时
                err_sum += err.sum().item()
                n_total += err.numel()

        hit_rate = n_hit / n_total
        hit_rates.append(hit_rate)
        print(f"{target:>7.1f} {hit_rate:>9.1%} {n_alive / n_total:>7.1%} {err_sum / n_total:>9.3f}")

    print()
    print(f"平均到位率：{sum(hit_rates) / len(hit_rates):.1%}")
    print("读法：到位率高但存活率低 = 车开到目标了但杆子倒了；")
    print("      两者都低 = 策略没学会去目标（先检查 rew_target 有没有加进 return）。")

    env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
