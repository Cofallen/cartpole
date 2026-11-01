#!/usr/bin/env python3
"""策略探针：把导出的 policy.pt 当成一个普通函数调用，量出它学到的控制律。

回答两个问题（第三阶段教案 课时 6）：
1. 「这个模型要怎么调用？」—— obs(1x4) in -> action(1x1) out，就这么多。
2. 「它到底学了个什么东西？」—— 扫一维输入、拟合线性系数，看它是不是一个 PD 控制器。

用法::

    python scripts/policy_probe.py --policy logs/rsl_rl/cartpole_direct/<run>/exported/policy.pt
    python scripts/policy_probe.py --sweep           # 只打印一维扫描表

只依赖 torch，不需要 Isaac Sim / GPU。
"""

from __future__ import annotations

import argparse
import os

import torch

# 观测顺序，来自 cartpole_env.py::_get_observations 的 torch.cat 顺序
OBS_NAMES = ["pole_angle", "pole_vel", "cart_pos", "cart_vel"]


def load_module(path: str) -> torch.jit.ScriptModule:
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}，先跑 play.py 导出（见 STAGE3_LESSON_PLAN.md 课时 6）")
    return torch.jit.load(path, map_location="cpu").eval()


def act(module, obs: list[float]) -> float:
    with torch.no_grad():
        return float(module(torch.tensor([obs], dtype=torch.float32)).reshape(-1)[0])


def sweep(module, idx: int, values: list[float], action_scale: float) -> list[tuple[float, float]]:
    out = []
    for v in values:
        obs = [0.0, 0.0, 0.0, 0.0]
        obs[idx] = v
        a = act(module, obs)
        out.append((v, a))
    return out


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--policy",
        default=os.path.join(here, "logs/rsl_rl/cartpole_direct/2026-10-31_17-23-01/exported/policy.pt"),
        help="导出的 torchscript 策略",
    )
    parser.add_argument("--action-scale", type=float, default=100.0, help="训练该策略时使用的 action_scale")
    parser.add_argument("--sweep", action="store_true", help="只打印一维扫描表")
    args = parser.parse_args()

    module = load_module(args.policy)
    print(f"policy : {args.policy}")
    print(f"action_scale = {args.action_scale:g} N / action unit")
    print()

    grids = {
        "pole_angle": [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5],
        "pole_vel": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        "cart_pos": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "cart_vel": [-2.0, -1.0, 0.0, 1.0, 2.0],
    }
    for idx, name in enumerate(OBS_NAMES):
        print(f"-- {name} (其它输入置 0)")
        for v, a in sweep(module, idx, grids[name], args.action_scale):
            print(f"   {name}={v:+.2f}  action={a:+.4f}  force={a * args.action_scale:+7.1f} N")
        print()

    if args.sweep:
        return

    # 用两端点差商估计线性增益：a ~ b0 + sum_i k_i * obs_i
    gains = {}
    for idx, name in enumerate(OBS_NAMES):
        vals = [v for v, _ in sweep(module, idx, grids[name], args.action_scale)]
        acts = [a for _, a in sweep(module, idx, grids[name], args.action_scale)]
        span = vals[-1] - vals[0]
        gains[name] = (acts[-1] - acts[0]) / span if span else 0.0

    b0 = act(module, [0.0, 0.0, 0.0, 0.0])
    print("线性近似（在零点附近一阶展开），action 单位：")
    print(f"   action ≈ {b0:+.3f} " + " ".join(f"{gains[n]:+.2f}*{n}" for n in OBS_NAMES))
    print("同一控制律换成力（乘以 action_scale）：")
    print(f"   force  ≈ {b0 * args.action_scale:+.1f} N " + " ".join(
        f"{gains[n] * args.action_scale:+.1f}*{n}" for n in OBS_NAMES
    ))
    print()
    print("读法：pole_angle / pole_vel 的增益远大于 cart_pos / cart_vel，说明策略本体是一个")
    print("      「角度 + 角速度」反馈（PD 型）控制器，车位置只是弱修正项。")
    print("      符号以本 USD 的关节约定为准，不要跨环境照抄。")


if __name__ == "__main__":
    main()
