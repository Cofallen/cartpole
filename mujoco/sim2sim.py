"""Isaac Lab tracking-cartpole 的 MuJoCo sim2sim（带键盘控制目标位置）。

================================ 策略契约 ================================
必须与 source/cartpole/cartpole/tasks/direct/cartpole/cartpole_env.py 完全一致：

    obs     = [theta, omega, x, v, target - x]        # 5 维，float32
    action  = 1 维，语义是**力矩**：force = action * action_scale(200 N)
    执行器  = 纯力矩 + joint damping 10，没有 PD 位置环
    控制频率 = 60 Hz （sim.dt = 1/120，decimation = 2）
    终止条件 = |theta| > pi/2 或 |x| > 3.0

================================ 运行方式 ================================
必须用装了 mujoco/torch 的解释器（本机的 conda env `isaac`）。
如果用 `python` 报 numpy 的 ImportError，说明当前解释器不对，换成：

    /home/cofallen/miniconda3/envs/isaac/bin/python mujoco/sim2sim.py

常用命令：

    # 开窗口 + 键盘遥控目标（默认）
    <python> mujoco/sim2sim.py

    # 指定策略 / 初始目标
    <python> mujoco/sim2sim.py --policy logs/rsl_rl/cartpole_direct/xxx/exported/policy.pt --target 1.5

    # 无窗口跑 5 秒（等价于 Isaac 的一个 episode），打印到位率并落 CSV
    <python> mujoco/sim2sim.py --headless --duration 5 --csv /tmp/run.csv

================================ 键盘 ================================
    ← / A        目标左移 0.10 m        → / D        目标右移 0.10 m
    ,            目标左移 0.02 m        .            目标右移 0.02 m
    ↓            目标左移 0.50 m        ↑            目标右移 0.50 m
    Space        暂停 / 继续
    R            重置 episode（车回 0，杆角在 ±0.1π 内随机）
    Backspace    目标归零
    Q / Esc      退出

窗口里的相机可以用鼠标左键拖动旋转、右键平移、滚轮缩放。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# 与 Isaac 训练配置对齐的常量（改动前请先看 cartpole_env_cfg.py）
# --------------------------------------------------------------------------- #
ACTION_SCALE = 200.0  # [N]   env cfg: action_scale
EFFORT_LIMIT = 400.0  # [N]   env cfg: effort_limit_sim
CTRL_DT = 1.0 / 60.0  # [s]   sim.dt(1/120) * decimation(2)
EPISODE_TIME = 5.0  # [s]   env cfg: episode_length_s
MAX_CART_POS = 3.0  # [m]   env cfg: max_cart_pos
POLE_ANGLE_LIMIT = np.pi / 2  # [rad] env: _get_dones 里的 theta 阈值
INIT_POLE_ANGLE = 0.1  # [*pi rad] env cfg: initial_pole_angle_range

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = REPO_ROOT / "mujoco" / "cartpole.xml"
DEFAULT_POLICY = (
    REPO_ROOT / "logs" / "rsl_rl" / "cartpole_direct"
    / "2026-11-04_17-52-24" / "exported" / "policy.pt"
)

# GLFW 键码（mujoco.viewer 的 key_callback 直接给这个）
KEY_LEFT, KEY_RIGHT, KEY_DOWN, KEY_UP = 263, 262, 264, 265
KEY_SPACE, KEY_BACKSPACE, KEY_ESCAPE = 32, 259, 256


# --------------------------------------------------------------------------- #
# 键盘遥控：只维护一个 target，物理循环每帧读它算 observation
# --------------------------------------------------------------------------- #
@dataclass
class TargetControl:
    """目标点的"遥控器"。key_callback 在 viewer 线程写，主循环读。"""

    target: float = 1.0
    low: float = -2.5
    high: float = 2.5
    coarse: float = 0.10
    fine: float = 0.02
    big: float = 0.50
    paused: bool = False
    reset_requested: bool = False
    quit_requested: bool = False

    def nudge(self, delta: float) -> None:
        self.target = float(np.clip(self.target + delta, self.low, self.high))

    def on_key(self, key: int) -> None:
        if key in (KEY_LEFT, ord("A"), ord("a")):
            self.nudge(-self.coarse)
        elif key in (KEY_RIGHT, ord("D"), ord("d")):
            self.nudge(+self.coarse)
        elif key == ord(","):
            self.nudge(-self.fine)
        elif key == ord("."):
            self.nudge(+self.fine)
        elif key == KEY_DOWN:
            self.nudge(-self.big)
        elif key == KEY_UP:
            self.nudge(+self.big)
        elif key == KEY_BACKSPACE:
            self.target = 0.0
        elif key == KEY_SPACE:
            self.paused = not self.paused
        elif key in (ord("R"), ord("r")):
            self.reset_requested = True
        elif key in (ord("Q"), ord("q"), KEY_ESCAPE):
            self.quit_requested = True


# --------------------------------------------------------------------------- #
# 仿真封装
# --------------------------------------------------------------------------- #
class CartpoleSim:
    """状态读取、观测拼装、策略推理、物理推进 —— 一切与 Isaac 的契约都在这里。"""

    CART_JOINT = "slider_to_cart"
    POLE_JOINT = "cart_to_pole"
    TARGET_BODY = "target_marker"

    def __init__(self, mjcf_path: Path, policy_path: Path, seed: int = 0) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.n_sub = max(1, int(round(CTRL_DT / self.model.opt.timestep)))

        # 用名字取索引，而不是硬编码 0/1：以后往 XML 里加关节也不会错位
        cart = self._id(mujoco.mjtObj.mjOBJ_JOINT, self.CART_JOINT)
        pole = self._id(mujoco.mjtObj.mjOBJ_JOINT, self.POLE_JOINT)
        self.cart_qpos, self.cart_dof = int(self.model.jnt_qposadr[cart]), int(self.model.jnt_dofadr[cart])
        self.pole_qpos, self.pole_dof = int(self.model.jnt_qposadr[pole]), int(self.model.jnt_dofadr[pole])

        # 目标点是一个 mocap body：不受物理影响，改 mocap_pos 就能移动它
        target_body = self._id(mujoco.mjtObj.mjOBJ_BODY, self.TARGET_BODY)
        self.mocap_id = int(self.model.body_mocapid[target_body])
        self.marker_z = float(self.model.body_pos[target_body][2])

        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.reset()

    # ---------------- 基础工具 ----------------
    def _id(self, obj_type, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(f"MJCF 里找不到 {name!r}（检查 cartpole.xml）")
        return idx

    @property
    def x(self) -> float:
        return float(self.data.qpos[self.cart_qpos])

    @property
    def v(self) -> float:
        return float(self.data.qvel[self.cart_dof])

    @property
    def theta(self) -> float:
        return float(self.data.qpos[self.pole_qpos])

    @property
    def omega(self) -> float:
        return float(self.data.qvel[self.pole_dof])

    @property
    def force(self) -> float:
        return float(self.data.ctrl[0])

    # ---------------- episode 管理 ----------------
    def reset(self) -> None:
        """对齐 Isaac 的 _reset_idx：x=0，杆角在 ±0.1π 内均匀采样，速度清零。"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.cart_qpos] = 0.0
        self.data.qpos[self.pole_qpos] = self.rng.uniform(-INIT_POLE_ANGLE, INIT_POLE_ANGLE) * np.pi
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)

    def terminated(self) -> bool:
        """与 env._get_dones 一致。"""
        return abs(self.theta) > POLE_ANGLE_LIMIT or abs(self.x) > MAX_CART_POS

    # ---------------- 与策略交互 ----------------
    def observation(self, target: float) -> np.ndarray:
        """顺序必须是 [theta, omega, x, v, target - x]，符号不能反。"""
        return np.array(
            [self.theta, self.omega, self.x, self.v, target - self.x],
            dtype=np.float32,
        )

    def policy_action(self, target: float) -> tuple[float, float]:
        obs = self.observation(target)
        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs).unsqueeze(0))
        action = float(action.reshape(-1)[0])
        force = float(np.clip(action * ACTION_SCALE, -EFFORT_LIMIT, EFFORT_LIMIT))
        return action, force

    def set_marker(self, target: float) -> None:
        self.data.mocap_pos[self.mocap_id] = (target, 0.0, self.marker_z)
        self.data.mocap_quat[self.mocap_id] = (1.0, 0.0, 0.0, 0.0)

    def advance(self, target: float) -> tuple[float, float]:
        """跑一个控制周期：推理一次 + n_sub 个物理步。返回 (action, force)。"""
        action, force = self.policy_action(target)
        self.data.ctrl[0] = force
        self.set_marker(target)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        return action, force


# --------------------------------------------------------------------------- #
# 无窗口：跑固定时长，输出与 Isaac 同口径的指标
# --------------------------------------------------------------------------- #
def run_headless(sim: CartpoleSim, target: float, duration: float, csv_path: Path | None) -> None:
    n_steps = max(1, int(round(duration / CTRL_DT)))
    errors, rows, terminations = [], [], 0

    print(f"[headless] target={target:+.2f} m, {n_steps} 步（{duration:.1f} s）, 策略输入 5 维")
    print(f"{'t':>6} {'x':>8} {'v':>8} {'theta':>8} {'omega':>8} {'force':>8} {'err':>7}")

    start = time.perf_counter()
    for step in range(n_steps):
        _, force = sim.advance(target)
        err = abs(sim.x - target)
        errors.append(err)
        rows.append((sim.data.time, sim.x, sim.v, sim.theta, sim.omega, force, target))

        if step % 30 == 0:
            print(
                f"{sim.data.time:6.2f} {sim.x:8.3f} {sim.v:8.3f} {sim.theta:8.3f} "
                f"{sim.omega:8.3f} {force:8.1f} {err:7.3f}"
            )
        if sim.terminated():
            terminations += 1
            print(f"  ⚠ 第 {step} 步被终止（|theta|={abs(sim.theta):.2f}, |x|={abs(sim.x):.2f}），按 Isaac 规则重开")
            sim.reset()

    errors_arr = np.asarray(errors)
    print("-" * 60)
    print(f"最终 x = {sim.x:+.3f} m（目标 {target:+.2f}）  最终 |err| = {abs(sim.x - target):.3f} m")
    print(f"平均 |err| = {errors_arr.mean():.3f} m   到位率(|err|<0.1) = {np.mean(errors_arr < 0.1):.1%}   终止次数 = {terminations}")
    print(f"墙钟耗时 {time.perf_counter() - start:.1f} s（仿真 {sim.data.time:.1f} s）")

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            csv_path,
            np.asarray(rows),
            delimiter=",",
            header="t,x,v,theta,omega,force,target",
            comments="",
            fmt="%.6f",
        )
        print(f"轨迹已写入 {csv_path}")


# --------------------------------------------------------------------------- #
# 有窗口：键盘遥控目标 + 实时仿真
# --------------------------------------------------------------------------- #
def run_viewer(sim: CartpoleSim, ctrl: TargetControl) -> None:
    model, data = sim.model, sim.data

    def key_callback(key: int) -> None:
        # 注意：这个回调在 viewer 自己的线程里执行，只做"改一个标量"这种安全操作
        ctrl.on_key(key)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 侧视相机会一直看着轨道；鼠标可以自由拖动
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.0, 0.0, 0.62)
        viewer.cam.distance = 5.4
        viewer.cam.azimuth = 100
        viewer.cam.elevation = -12

        sim.set_marker(ctrl.target)

        print("=" * 74)
        print("键盘遥控：←/→(A/D) 目标 ±0.10m   ,/. 目标 ±0.02m   ↑/↓ 目标 ±0.50m")
        print("          Space 暂停   R 重置   Backspace 目标归零   Q/Esc 退出")
        print("=" * 74)

        next_tick = time.perf_counter()
        next_hud = 0.0
        hud_warned = False  # HUD 出错只提醒一次，不要每帧刷屏

        while viewer.is_running() and not ctrl.quit_requested:
            if ctrl.reset_requested:
                ctrl.reset_requested = False
                sim.reset()
                print(f"[reset] 车回到 x=0，杆角 {sim.theta:+.3f} rad，目标仍是 {ctrl.target:+.2f} m")

            if not ctrl.paused:
                _, force = sim.advance(ctrl.target)

                if sim.terminated():
                    print(
                        f"[terminated] t={data.time:5.2f}s  |theta|={abs(sim.theta):.2f}  "
                        f"|x|={abs(sim.x):.2f}  → 自动重置"
                    )
                    sim.reset()

                # 实时节拍：一步控制周期 = CTRL_DT
                next_tick += CTRL_DT
                lag = next_tick - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                else:
                    next_tick = time.perf_counter()  # 追不上就丢掉欠账，别越拖越慢
            else:
                sim.set_marker(ctrl.target)
                mujoco.mj_forward(model, data)
                next_tick = time.perf_counter()
                time.sleep(CTRL_DT)

            # 每 0.1 s 刷一次 HUD
            now = time.perf_counter()
            if now >= next_hud:
                next_hud = now + 0.1
                err = sim.x - ctrl.target
                # 注意：这里必须是 list，不能是 tuple。
                # 这版 viewer.set_texts() 会先判断 isinstance(texts, tuple)，
                # 是 tuple 就当成"单条文本"再包一层 -> 解包时报
                # ValueError: not enough values to unpack (expected 4, got 3)
                hud = [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f"target x* = {ctrl.target:+.2f} m",
                        f"x = {sim.x:+.3f} m",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        f"err = {err:+.3f} m",
                        f"theta = {sim.theta:+.3f} rad",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                        "←/→ 0.10m   ,/. 0.02m   ↑/↓ 0.50m   Space 暂停   R 重置   Backspace 归零   Q 退出",
                        f"force = {sim.force:.1f} N" + ("   [PAUSED]" if ctrl.paused else ""),
                    ),
                ]
                try:
                    viewer.set_texts(hud)
                except Exception as exc:  # HUD 只是装饰，绝不能因为它把整个仿真带崩
                    if not hud_warned:
                        print(f"[warn] HUD 不可用（{exc}），仿真继续；键盘遥控照常工作")
                        hud_warned = True

            viewer.sync()

    print("窗口已关闭")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isaac Lab tracking-cartpole 的 MuJoCo sim2sim（键盘控制目标位置）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="MuJoCo 模型路径")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="导出的 TorchScript 策略")
    parser.add_argument("--target", type=float, default=1.0, help="初始目标位置 x* [m]")
    parser.add_argument("--headless", action="store_true", help="不开窗口，跑到 duration 结束")
    parser.add_argument("--duration", type=float, default=EPISODE_TIME, help="headless 模式时长 [s]")
    parser.add_argument("--csv", type=Path, default=None, help="把轨迹写成 CSV（headless 用）")
    parser.add_argument("--seed", type=int, default=0, help="重置时杆角随机采样种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, what in ((args.mjcf, "MJCF"), (args.policy, "策略")):
        if not path.exists():
            raise FileNotFoundError(f"{what} 文件不存在：{path}")

    sim = CartpoleSim(args.mjcf, args.policy, seed=args.seed)

    if args.headless:
        run_headless(sim, args.target, args.duration, args.csv)
    else:
        run_viewer(sim, TargetControl(target=float(np.clip(args.target, -2.5, 2.5))))


if __name__ == "__main__":
    main()
