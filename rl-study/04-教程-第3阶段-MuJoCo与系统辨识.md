# 第 3 阶段：MuJoCo 与系统辨识（5 个任务）

> 目标：把 Isaac 训出来的策略放到 MuJoCo 里跑通（sim2sim），然后用实测数据把仿真参数辨识出来。
> 前置：第 2 阶段完成，手里有一个从 tracking 任务训出来的 `policy.pt`（5 维观测、力矩接口）。

---

## T3.1 写 cartpole 的 MJCF

**目的**：搞清"一个机器人模型最少需要哪些东西"：质量、惯量、关节、限位、执行器。

**前置**：T2.1（知道 PD 参数）。

**操作**：新建 `~/Code/IsaacTest/cartpole/mujoco/cartpole.xml`：

```xml
<mujoco model="cartpole">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" range="-3 3" limited="true" damping="10"/>
      <geom name="cart" type="box" size="0.1 0.05 0.05" mass="1.0"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 1" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="force" joint="slider" gear="1" ctrlrange="-400 400"/>
  </actuator>
</mujoco>
```

**每个字段为什么存在**：

| 字段                    | 作用       | 不对会怎样                               |
| --------------------- | -------- | ----------------------------------- |
| `timestep="0.002"`    | 物理步长     | 太大不稳定，太小慢                           |
| `slide` 关节 + `range`  | 小车只能左右走  | 没有 range 就没有限位，会飞出场景                |
| `hinge` 关节            | 杆子绕 y 轴转 | 轴写错 → 杆子绕错方向                        |
| `mass`                | 质量       | 影响动力学，**T3.4 要辨识它**                 |
| `damping`             | 阻尼       | 影响"松手后多久停"，也是辨识对象                   |
| `motor` + `ctrlrange` | 执行器与饱和   | 与 Isaac 的 `effort_limit_sim=400` 对应 |

**验证**

```bash
python -m mujoco.viewer --mjcf=mujoco/cartpole.xml
```

（没装 viewer 就用 `python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mujoco/cartpole.xml'); print(m.nq, m.nv, m.nu)"`，应输出 `2 2 1`。）

**可能报错**

| 报错                               | 原因                                        |
| -------------------------------- | ----------------------------------------- |
| `XML Error: ... unknown element` | 标签拼错，比如 `worldbody` 写成 `worldBody`        |
| `nq=3`                           | 用了 `freejoint` 或 quaternion 关节，说明你写成了浮动基座 |

**验收**：模型能加载，`nq=2, nv=2, nu=1`；能说出每个 `mass` / `damping` 是干什么的。

> 🖊️ **记录**
> 
> 我的 MJCF 参数（质量/长度/阻尼）：
> &nbsp;
> `nq, nv, nu` 输出：\_\_\_\_ / \_\_\_\_ / \_\_\_\_

---

## T3.2 写 sim2sim 推理脚本（tracking 版）

**目的**：让 Isaac 训出来的策略，在 MuJoCo 里闭环跑起来——而且跑的是**你真正训练的那个任务**（把车开到指定目标），不是第 1 阶段的"原地平衡"。

**前置**：T3.1 + 一个从 `Template-Cartpole-Track-Direct-v0` 导出的 `policy.pt`（5 维观测）。

> ⚠️ 用第 1 阶段的 4 维平衡策略跑这一节，契约根本对不上，必然失败。先确认你的 `policy.pt` 来自 tracking 任务。

---

### 第一步：先把"策略的输入输出契约"抄成一张表

sim2sim 失败，90% 是凭记忆写观测。打开 `cartpole_env.py`，把真实契约抄下来：

| 序号  | 观测分量             | 代码来源                           | 单位    |
| --- | ---------------- | ------------------------------ | ----- |
| 1   | 杆角 $\theta$      | `joint_pos[pole]`              | rad   |
| 2   | 杆角速度 $\omega$    | `joint_vel[pole]`              | rad/s |
| 3   | 车位 $x$           | `joint_pos[cart]`              | m     |
| 4   | 车速 $v$           | `joint_vel[cart]`              | m/s   |
| 5   | **目标误差 $x^*-x$** | `target_pos - joint_pos[cart]` | m     |

输出：1 维，**语义是力矩**：

$$
F = a \cdot \texttt{action\_scale},\qquad \texttt{action\_scale}=200\ \text{N}
$$

执行器：`stiffness=0`、`damping=10`、`effort_limit_sim=400`（就是 `CARTPOLE_CFG` 原值）。

> 🔴 **这里没有 PD。** T2.1 的位置版是 `set_joint_position_target` + `stiffness=400`；
> 你现在训的是**纯力矩**版（`set_joint_effort_target`），所以 sim2sim 里必须把 action 直接乘成力，
> **不能**再套 `KP*(q_des - q)`。接口抄错，就是最常见的"杆子立刻倒"。

---

### 第二步：先搞清楚"目标从哪来"（T2.3 已详讲，这里只回顾）

`target_pos` 在 `__init__` 里是 `zeros`，那只是**占位**；真正赋值在 `_reset_idx`（T2.3 改 3），
而 `DirectRLEnv.reset()` 会调 `_reset_idx(全部 env)`，`step()` 里摔倒的 env 也会调。

实测（8 个 env）：

```
gym.make 之后（还没 reset） : target_pos = [0.0, 0.0, 0.0, ...]      ← 占位值，别被骗
第一次 reset 之后          : target_pos = [0.903, 1.311, 1.662, ...] ← 真的采出来了
第二次 reset 之后          : target_pos = [-1.999, -0.013, 0.73, ...] ← 重采样
```

**目标 = 每个 episode 随机抽的一张"考卷"**，通过第 5 维喂给网络；网络不知道"目标是几"，
只知道"还差多少"。

---

### 第三步：我训出来的到底是什么？

一个带参数的控制器：

$$
a = \pi_\theta\left(\theta,\ \omega,\ x,\ v,\ x^*-x\right)
$$

- 同一组 $(\theta,\omega,x,v)$，换个目标 $x^*$，输出就会变——这就叫"跟踪"。
- 网络里**没有任何变量叫 target**。target 是你每次调用时传进去的**实参**。
- 所以部署时你必须自己维护 $x^*$，算成 $x^*-x$ 塞进第 5 维。
  训练好的策略 = 一个函数；$x^*$ = 你调用它时给的参数。

**这就是"训练时就把目标考虑进去"的意思**：不是网络记住了某个目标，而是它学会了"误差 → 该出多大力"这个映射。

---

**操作**：新建 `mujoco/sim2sim.py`：

```python
"""把 Isaac 训出来的 tracking 策略放进 MuJoCo 跑闭环。

契约（必须和 cartpole_env.py 完全一致）：
    obs    = [theta, omega, x, v, target - x]     5 维
    action = 1 维力矩，force = action * 200 N
    执行器 = 纯力矩 + damping=10，没有 PD
"""
import numpy as np
import mujoco
import torch

MJCF = "mujoco/cartpole.xml"
POLICY = "logs/rsl_rl/cartpole_direct/<run>/exported/policy.pt"

ACTION_SCALE = 200.0      # 训练时的 action_scale [N]
CTRL_DT = 1.0 / 60.0      # 控制周期 = sim.dt(1/120) * decimation(2)
EPISODE_STEPS = 300       # 5 秒
TARGET = 1.0              # ← 想让车停在哪：这是"部署时由你给的实参"

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
n_sub = max(1, int(round(CTRL_DT / model.opt.timestep)))

policy = torch.jit.load(POLICY).eval()

# 初始条件对齐 Isaac 的 _reset_idx：x=0，杆角在 ±0.1π 内随机
rng = np.random.default_rng(0)
data.qpos[0] = 0.0
data.qpos[1] = rng.uniform(-0.1, 0.1) * np.pi
mujoco.mj_forward(model, data)

errors = []
for step in range(EPISODE_STEPS):
    x, v = data.qpos[0], data.qvel[0]
    theta, omega = data.qpos[1], data.qvel[1]

    # ---- 观测：顺序一致 + 第 5 维必须是 (target - x)，符号别写反 ----
    obs = np.array([theta, omega, x, v, TARGET - x], dtype=np.float32)

    with torch.no_grad():
        a = float(policy(torch.from_numpy(obs).reshape(1, -1)).item())

    # ---- 动作就是力矩，直接乘 action_scale，没有 PD ----
    force = a * ACTION_SCALE
    data.ctrl[0] = np.clip(force, -400.0, 400.0)   # 对应 effort_limit_sim=400

    # MuJoCo 物理步长比控制周期小，一个控制周期要跑 n_sub 次物理步
    for _ in range(n_sub):
        mujoco.mj_step(model, data)

    errors.append(abs(x - TARGET))
    if step % 30 == 0:
        print(f"t={step*CTRL_DT:4.1f}s  x={x:+.3f}  err={abs(x-TARGET):.3f}  "
              f"theta={theta:+.3f}  F={force:+7.1f}")

    # 终止条件与 Isaac 的 _get_dones 一致
    if abs(theta) > np.pi / 2 or abs(x) > 3.0:
        print(f"terminated at step {step}: |theta|={abs(theta):.2f}, |x|={abs(x):.2f}")
        break

print(f"到位率(|err| < 0.1) = {np.mean(np.array(errors) < 0.1):.1%}")
```

**预期**

| 现象                                          | 含义                                                      |
| ------------------------------------------- | ------------------------------------------------------- |
| `x` 收敛到 `TARGET` ±0.1 并稳住，`theta` 在 ±0.05 内 | 契约对齐了，且策略确实学会了跟踪                                        |
| `x` 停在 `TARGET` 的某个比例（比如 0.7·TARGET）        | 策略只学到"往目标走"，精度不够 → 回 T2.4 改 reward 或加训练轮数               |
| `theta` 立刻发散                                | 观测顺序 / 第 5 维构造 / **力矩 vs PD 接口**抄错                      |
| `x` 往反方向跑                                   | 第 5 维符号写反了（应是 `target - x`，不是 `x - target`）             |
| `x` 一直贴着 0 不动                               | 策略根本没学过 tracking（回到 T2.3 检查 `rew_target` 是否进了 `return`） |

---

### 两种部署方法：5 维（训练时就带目标）vs 4 维（部署时偏移输入）

跟踪一个目标有两种做法，都真实存在，取舍不同。上面那个脚本是**方法 A**，下面补上**方法 B**。

#### 方法 A：训练时把误差放进观测（5 维，推荐）

$$
obs = [\theta,\ \omega,\ x,\ v,\ x^*-x]
$$

网络在训练中见过各种误差，学到的是"误差 → 该出多大力"。

- 优点：精度高（直接优化误差）；网络能看到目标位置，因而能学会避让边界；
  目标范围由 `target_range` 决定，想扩就扩。
- 代价：必须改 env（T2.3），观测多一维。

#### 方法 B：训练 4 维，部署时把 `x` 换成 `x - x*`

训练时还是第 1 阶段那个 4 维策略：

$$
a = \pi_\theta(\theta,\ \omega,\ x,\ v)
$$

部署时只改一处输入：把第 3 维的 $x$ 换成 $x-x^*$：

```python
# 方法 B：4 维策略 + 偏移输入（训练不用改，部署时改这一行）
obs = np.array([theta, omega, x - TARGET, v], dtype=np.float32)
a = float(policy(torch.from_numpy(obs).reshape(1, -1)).item())
```

**为什么会有用**：策略的本事是"把 $x$ 拉回 0"。你把输入换成 $x-x^*$，等于告诉它
"离目标还差这么多"，它就会把这个差拉到 0，也就是让 $x \to x^*$。

对线性反馈 $u = k_1\theta + k_2\omega + k_3 x + k_4 v$，偏移后闭环平衡点是

$$
k_3\left(x - x^*\right) = 0 \;\Longrightarrow\; x = x^*
$$

所以这是**改变平衡点**（reference injection），在一阶线性、平移不变的系统上成立。

**但它有四个硬前提，缺一个就会失效：**

| 前提                | 说明                                                                                         | 不满足会怎样        |
| ----------------- | ------------------------------------------------------------------------------------------ | ------------- |
| ① 4 维策略真的是"位置调节器" | reward 里必须有位置项（如 $-x^2$）。只有 `-                                                             | cart_vel      |
| ② 系统平移不变          | cartpole 在长导轨上成立；有地形、有终点、有障碍就不成立                                                           | 目标附近的动态和原点不一样 |
| ③ 输入别跑出训练分布       | `obs_normalizer` 的均值方差是在**训练分布**上估的，训练时 $x$ 一直在 0 附近（std 很小）；部署时 $x-x^*$ 可能到 ±2，归一化后数值会非常大 | 网络饱和、输出打满     |
| ④ 目标范围受限          | 偏移后网络"看到"的状态是 $x-x^*$，它没见过的区域都是外推                                                          | 大目标精度崩、甚至失稳   |

**实测（本仓库两个 4 维策略，64 env × 2 回合 × 300 步，偏移输入）**：

| 策略                    | 指标       | −2    | −1    | 0     | +1    | +2    |
| --------------------- | -------- | ----- | ----- | ----- | ----- | ----- |
| `2026-10-31_17-23-01` | 到位率      | 0.4%  | 22.2% | 18.9% | 0.0%  | 0.0%  |
| 同上                    | mean_err | 0.938 | 0.440 | 0.244 | 0.917 | 1.652 |
| `2026-11-02_18-48-52` | 到位率      | 3.9%  | 5.0%  | 10.7% | 0.0%  | 0.0%  |
| 同上                    | mean_err | 1.247 | 0.803 | 0.746 | 1.129 | 1.522 |

（两个策略的 alive 都是 99.7%，说明杆一直立着，接口没问题，结论可信。）

**怎么读这张表**：`target = 0` 时偏移等于没偏移，等于直接考"这个策略本来能不能把车稳在 0"。
结果 mean_err 分别是 **0.24 m 和 0.75 m**——说明它们压根没学过位置调节
（reward 里只有 `rew_scale_cart_vel`、没有位置项，否则 target=0 时应该收敛到 0 附近）。
**连 0 都稳不住，偏移到一个新目标当然更不可能准。**

所以方法 B 不是"免费午餐"：它只是因为原策略本来就不管位置，才显得差；
如果你把 4 维策略的 reward 改成 $-x^2$ 让它学会稳在 0，偏移就能把平衡点搬到目标上，效果会好得多。

#### 怎么选

| 维度      | 方法 A（5 维）          | 方法 B（4 维 + 偏移）  |
| ------- | ------------------ | --------------- |
| 要改 env  | 要（T2.3）            | 不用              |
| 观测维数    | 5                  | 4               |
| 目标范围    | `target_range` 说了算 | 受策略状态分布限制       |
| 位置约束/边界 | 网络能看到目标，能学避让       | 完全不知道           |
| 精度      | 高                  | 取决于原策略是不是位置调节器  |
| 适用场景    | 要换目标、有约束、要精度       | 快速验证、平移不变、目标范围小 |

> 💡 **判据**：能不能"部署时换个输入而不重训"，取决于这个输入在训练时是否**在分布内**，
> 以及原策略学到的平衡点是否真的会被这个偏移搬走。两条都满足才敢用方法 B。

---

**可能报错**

| 现象                      | 原因                                            | 处理                                        |
| ----------------------- | --------------------------------------------- | ----------------------------------------- |
| 杆子立刻倒                   | 观测顺序错 / 第 5 维没给                               | 打印前 5 步的 `obs`，和 Isaac 里同一时刻对比            |
| 力一直在 ±400 打满            | 把力矩接口当成位置接口，或 `action_scale` 用了 0.5           | 本任务是 `action_scale=200`，`force = a * 200` |
| 一开始就 terminated         | `qpos[1]` 超 π/2                               | 检查初始角是不是 `*np.pi` 之后超出范围                  |
| `policy(...)` 报 shape 错 | 策略要 `float32` 且 shape `(1,5)`                 | 照抄上面；4 维说明你用错了 `policy.pt`                |
| `x` 完全不动                | MuJoCo 里 `damping` 和 Isaac 的 `damping=10` 差太多 | 对齐 T3.1 的 `damping="10"`                  |

**验收**：能闭环跑满 5 秒；能说出 `target_pos` 在训练的**哪一行**被赋值；
能说出方法 A / 方法 B 各自的**前提**和取舍，并解释方法 B 的四个前提缺一个会怎样。

> 🖊️ **记录**
> 
> 我的 `policy.pt` 来自哪个 run、观测几维：\_\_\_\_
> `ACTION_SCALE / CTRL_DT / damping` 三个数：\_\_\_\_ / \_\_\_\_ / \_\_\_\_
> 我测的 `TARGET` 和最终 `x`：\_\_\_\_ → \_\_\_\_；到位率：\_\_\_\_
> 方法 A 的到位率：\_\_\_\_；方法 B 的到位率：\_\_\_\_
> 方法 B 失效的话，卡在哪个前提（①②③④）：\_\_\_\_

---

## T3.3 对齐三件事：观测、缩放、频率

**目的**：sim2sim 失败，90% 是这三件事没对齐。这一任务就是排错清单。

**前置**：T3.2 能跑。

**检查清单**（一条一条验证，别跳）

| 项     | 怎么验证                                        | 常见错法                                             |
| ----- | ------------------------------------------- | ------------------------------------------------ |
| 观测顺序  | Isaac 里打印一次 obs，MuJoCo 里打印一次，同一初始状态应几乎一样    | 把 $[x,v,\theta,\omega]$ 当成 $[\theta,\omega,x,v]$ |
| 观测维数  | tracking 任务是 **5 维**，第 5 维 = `target - x`   | 只喂 4 维（用错 `policy.pt`），或第 5 维写成 `x - target`     |
| 单位    | 角度是 rad 还是 deg；位置是 m 还是 cm                  | 用 deg 喂给训练时用 rad 的策略                             |
| 动作缩放  | 力矩版：`force = a * 200`；位置版：`q_des = a * 0.5` | 忘了乘、乘了两次，或搞混 200 和 0.5                           |
| 控制频率  | Isaac 是 60 Hz；MuJoCo 用 `n_sub` 子步凑          | 直接 1 kHz 调策略（等于让策略"高频抖"）                         |
| 执行器饱和 | `ctrlrange` 与 `effort_limit_sim` 一致（±400）   | MuJoCo 用 ±1（没设 ctrlrange）                        |
| 阻尼/摩擦 | 两边 `damping` 一致                             | Isaac 的 USD 有阻尼而 MJCF 忘了写                        |

**做法**：每对齐一项，记录一次偏差。

**偏差怎么量**：同一初始状态（$\theta=0.1$，其余 0）、同一控制律（先不用策略，用固定力矩，比如 $F=50$ N），跑 5 s，比较 $\theta(t)$ 的最大绝对差：

$$
e = \max_t \left|\theta^{\text{Isaac}}(t) - \theta^{\text{MuJoCo}}(t)\right|
$$

**验收**：给出"对齐前 / 对齐后"的偏差表；能说出每一行对应哪个错误。

> 🖊️ **记录**
> 
> | 版本        | 观测顺序 | 缩放  | 频率  | 偏差 $e$ |
> | --------- | ---- | --- | --- | ------ |
> | 第一版（随便写的） |      |     |     |        |
> | 对齐后       |      |     |     |        |
> 
> 我踩的三个坑：
> &nbsp;
> &nbsp;

---

## T3.4 系统辨识：把仿真参数对上实测

**目的**：让仿真的"质量、摩擦、延迟"接近真实硬件。**这是 sim2real 的核心工作**，也是面试里能讲 20 分钟的内容。

**前置**：T3.3。

**第一步 · 列出待辨识参数**（含初值与范围）

| 参数       | 符号       | 初值       | 物理范围   | 怎么测  |
| -------- | -------- | -------- | ------ | ---- |
| 小车质量     | $M$      | 1.0 kg   | 0.2-5  | 电子秤  |
| 杆质量      | $m$      | 0.1 kg   | 0.01-1 | 电子秤  |
| 杆长（质心距离） | $l$      | 0.5 m    | 0.1-1  | 卷尺   |
| 小车阻尼     | $b$      | 10 N·s/m | 1-100  | 匀速拖动 |
| 关节摩擦     | $f_c$    | 0.05 N·m | 0-0.5  | 慢速拖动 |
| 控制延迟     | $\tau_d$ | 0 ms     | 0-40   | 阶跃响应 |

**第二步 · 设计激励实验**（每个参数都需要"能激发它"的动作）

| 实验   | 怎么做            | 主要辨识         |
| ---- | -------------- | ------------ |
| 自由摆  | 松手让杆从 30° 自由摆动 | $l$、杆阻尼      |
| 匀速拖动 | 用恒定低速推小车       | $b$          |
| 力阶跃  | 给 50 N 方波      | $M$、$\tau_d$ |
| 正弦扫频 | 0.1-5 Hz 扫频    | 频域响应、延迟      |

**第三步 · 最小二乘拟合**

```python
"""sysid.py：用实测响应拟合仿真参数。"""
import numpy as np
from scipy.optimize import least_squares

def simulate(params, u, T, dt=0.001):
    M, m, l, b = params
    x, v, th, w = 0.0, 0.0, np.deg2rad(30), 0.0
    hist = []
    for k in range(int(T / dt)):
        F = u(k * dt)
        s, c = np.sin(th), np.cos(th)
        den = l * (M + m * s * s)
        thdd = ((M + m) * 9.81 * s - F * c - m * l * w * w * s * c) / den
        xdd = (F + m * l * (w * w * s - thdd * c)) / (M + m) - b * v / (M + m)
        v += xdd * dt; x += v * dt
        w += thdd * dt; th += w * dt
        hist.append((x, th))
    return np.array(hist)

def residual(p):
    pred = simulate(p, u=lambda t: 0.0, T=2.0)      # 自由摆
    return (pred[:, 1] - measured_theta).ravel()

res = least_squares(residual, x0=[1.0, 0.1, 0.5, 10.0],
                    bounds=([0.2, 0.01, 0.1, 1.0], [5.0, 1.0, 1.0, 100.0]))
print("拟合参数:", res.x, "残差:", res.cost)
```

**注意事项**（辨识最容易错的地方）：

1. **一次只拟合一个实验**，别把自由摆和阶跃混在一起；
2. **参数不可辨识**：如果两个参数在方程里总是成对出现，你会拟合出无数组解——需要换激励；
3. **延迟要单独处理**：$\tau_d$ 不是连续参数，用网格搜索 0-40 ms；
4. **先看拟合曲线，再看参数**：曲线对不上，参数再"合理"也是错的。

**验收**：给出"参数表 + 拟合曲线 + 残差"，并说明哪个参数最不确定、为什么。

> 🖊️ **记录**
> 
> 我测到的原始数据（文件名/时长/采样率）：
> &nbsp;
> 拟合出的参数：
> &nbsp;
> 拟合误差最大的实验：
> &nbsp;

---

## T3.5 辨识之后：重训 + 复测

**目的**：证明"辨识过的仿真"能让策略迁移得更好。

**前置**：T3.4。

**操作**

1. 把辨识出的参数写回 MJCF；
2. 在 Isaac 里用**匹配后的参数**重训一版（如果 Isaac 的 USD 参数可改）；
3. 用 T3.2 的脚本复测，记录偏差 $e$；
4. 对比"辨识前训练 → sim2sim"和"辨识后训练 → sim2sim"。

**预期**：偏差 $e$ 下降；如果没降，说明辨识的参数不是影响迁移的主要因素（这时要怀疑延迟或观测，而不是质量）。

**验收**：给出两组（辨识前/后）的偏差与成功率对比表。

> 🖊️ **记录**
> 
> | 版本  | sim2sim 偏差 $e$ | 5 s 成功率 |
> | --- | -------------- | ------- |
> | 辨识前 |                |         |
> | 辨识后 |                |         |
> 
> 结论：
> &nbsp;

---

## 阶段 3 完成检查表

- [ ] T3.1 MJCF 能加载，`nq/nv/nu = 2/2/1`
- [ ] T3.2 tracking 策略能在 MuJoCo 闭环 5 s，并给出到位率
- [ ] T3.3 有对齐前后的偏差表
- [ ] T3.4 有参数表 + 拟合曲线 + 残差
- [ ] T3.5 有辨识前后对比
