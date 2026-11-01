#!/usr/bin/env python3
"""把 ``logs/rsl_rl/<experiment_name>/<timestamp>/`` 下的所有训练轮次压成一张对比表。

用途（第二轮/第三阶段教案 课时 1）：把「我感觉变慢了」变成「len>=284 用了 65 轮还是 74 轮」。
只读日志，不启动 Isaac Sim，不需要 GPU。

用法::

    python scripts/compare_runs.py                      # 默认读 logs/rsl_rl/cartpole_direct
    python scripts/compare_runs.py --md                 # 输出 markdown 表格，可直接贴进 EXPERIMENT_LOG.md
    python scripts/compare_runs.py --root logs/rsl_rl --min-iters 100

依赖：tensorboard（Isaac 环境里已自带）、pyyaml。
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# --- Isaac Lab 的 env.yaml 里含有 python/tuple 标签，需要放宽 loader -------------------
class _LenientLoader(yaml.SafeLoader):
    pass


_LenientLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
_LenientLoader.add_multi_constructor("tag:yaml.org,2002:python/object", lambda loader, suffix, node: None)

# --- 本任务固定的常量（来自 cartpole_env_cfg.py，改配置后请同步）---------------------
EPISODE_CAP = 299  # episode_length_s=5.0, dt=1/120, decimation=2 -> 300 步，timeout 在 299
SHOW_KEYS = [
    "action_scale",
    "rew_scale_alive",
    "rew_scale_terminated",
    "rew_scale_pole_pos",
    "rew_scale_cart_vel",
    "rew_scale_pole_vel",
    "initial_pole_angle_range",
    "max_cart_pos",
]


def load_cfg(run_dir: str) -> dict:
    path = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.load(f, _LenientLoader) or {}


def load_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    events = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))
    if not events:
        return {}
    ea = EventAccumulator(events[0], size_guidance={"scalars": 0})
    ea.Reload()
    return {tag: [(s.step, s.value) for s in ea.Scalars(tag)] for tag in ea.Tags()["scalars"]}


def first_below(series: list[tuple[int, float]], thr: float, after: int = 50) -> int | None:
    """第一次「持续」低于阈值的迭代号：要求之后连续 5 个点也低于阈值，避免单点噪声。"""
    for i, (step, value) in enumerate(series):
        if step < after or value > thr:
            continue
        window = series[i : i + 5]
        if all(v <= thr for _, v in window):
            return step
    return None


def first_above(series: list[tuple[int, float]], thr: float) -> int | None:
    for i, (step, value) in enumerate(series):
        if value < thr:
            continue
        window = series[i : i + 5]
        if len(window) == 5 and all(v >= thr for _, v in window):
            return step
    return None


def summarize(run_dir: str) -> dict:
    cfg = load_cfg(run_dir)
    sc = load_scalars(run_dir)
    rew = sc.get("Train/mean_reward", [])
    length = sc.get("Train/mean_episode_length", [])
    std = sc.get("Policy/mean_noise_std", [])
    lr = sc.get("Loss/learning_rate", [])
    vf = sc.get("Loss/value_function", [])

    row = {
        "run": os.path.basename(run_dir.rstrip("/")),
        "iters": len(rew),
        "cfg": cfg,
    }
    if not rew:
        return row

    alive = float(cfg.get("rew_scale_alive", 1.0))
    row["len95"] = first_above(length, 0.95 * EPISODE_CAP)
    row["std02"] = first_below(std, 0.2)
    row["std01"] = first_below(std, 0.1)
    row["reward"] = rew[-1][1]
    row["util"] = rew[-1][1] / (EPISODE_CAP * alive)  # 奖励利用率：离理论上限还差多少
    row["jit"] = statistics.pstdev([v for _, v in rew[-100:]]) if len(rew) >= 100 else None
    row["vf"] = vf[-1][1] if vf else None
    row["lr_min"] = min(v for _, v in lr) if lr else None
    row["std_end"] = std[-1][1] if std else None
    return row


FORMATS = {
    "action_scale": ("a", "{:.0f}"),
    "rew_scale_alive": ("alive", "{:.0f}"),
    "rew_scale_terminated": ("term", "{:.0f}"),
    "rew_scale_pole_pos": ("pole_f", "{:.0f}"),
    "rew_scale_cart_vel": ("cart_v", "{:.3f}"),
    "rew_scale_pole_vel": ("pole_v", "{:.3f}"),
}


def fmt_init(cfg: dict) -> str:
    rng = cfg.get("initial_pole_angle_range")
    if not rng:
        return "-"
    return f"[{rng[0]:g},{rng[1]:g}]pi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs/rsl_rl/cartpole_direct", help="实验目录（含多个时间戳子目录）")
    parser.add_argument("--min-iters", type=int, default=0, help="只显示迭代数 >= 该值的轮次")
    parser.add_argument("--md", action="store_true", help="输出 markdown 表格")
    args = parser.parse_args()

    rows = [summarize(d) for d in sorted(glob.glob(os.path.join(args.root, "*/")))]
    rows = [r for r in rows if r["iters"] >= args.min_iters]

    headers = ["run", "iters", "a", "alive", "term", "pole_f", "cart_v", "init", "len95", "std.2", "std.1", "reward", "util", "jit100", "vf", "lr_min"]

    def cells(r: dict) -> list[str]:
        c = r["cfg"]
        out = [r["run"], str(r["iters"])]
        for key in ("action_scale", "rew_scale_alive", "rew_scale_terminated", "rew_scale_pole_pos", "rew_scale_cart_vel"):
            _, f = FORMATS[key]
            out.append(f.format(c.get(key, float("nan"))) if key in c else "-")
        out.append(fmt_init(c))
        if r["iters"] < 10:
            out += ["-"] * 9
            return out
        out += [
            str(r["len95"]) if r["len95"] is not None else "-",
            str(r["std02"]) if r["std02"] is not None else "-",
            str(r["std01"]) if r["std01"] is not None else "-",
            f"{r['reward']:.1f}",
            f"{r['util']:.3f}",
            f"{r['jit']:.2f}" if r["jit"] is not None else "-",
            f"{r['vf']:.4f}" if r["vf"] is not None else "-",
            f"{r['lr_min']:.1e}" if r["lr_min"] is not None else "-",
        ]
        return out

    table = [cells(r) for r in rows]
    widths = [max(len(h), *(len(t[i]) for t in table)) if table else len(h) for i, h in enumerate(headers)]

    def line(cs, sep="  "):
        return sep.join(c.ljust(w) for c, w in zip(cs, widths)).rstrip()

    if args.md:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join(["---"] * len(headers)) + "|")
        for t in table:
            print("| " + " | ".join(t) + " |")
    else:
        print(line(headers))
        print(line(["-" * w for w in widths]))
        for t in table:
            print(line(t))
        print()
        print("len95 = 平均回合长度首次 >= 284（即 95% 上限）的迭代；std.2/std.1 = 动作噪声首次 <= 0.2/0.1 的迭代")
        print("util  = 末轮 reward / (299 * alive)，1.0 表示几乎不再受惩罚；jit100 = 末 100 轮 reward 的标准差")


if __name__ == "__main__":
    main()
