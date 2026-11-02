# 第 3 阶段：MuJoCo 与系统辨识（5 个任务）

> 目标：把 Isaac 训出来的策略放到 MuJoCo 里跑通（sim2sim），然后用实测数据把仿真参数辨识出来。
> 前置：第 2 阶段完成，手里有一个能用的 `policy.pt`（位置+PD 版最好）。

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

## T3.2 写 sim2sim 推理脚本

**目的**：让 Isaac 训出来的策略，在 MuJoCo 里闭环跑起来。

**前置**：T3.1 + 一个导出的 `policy.pt`。

**操作**：新建 `mujoco/sim2sim.py`：

```python
"""把 Isaac Lab 导出的策略放进 MuJoCo 跑闭环。"""
import numpy as np
import mujoco
import torch

MJCF = "mujoco/cartpole.xml"
POLICY = "logs/rsl_rl/cartpole_direct/<run>/exported/policy.pt"

ACTION_SCALE = 0.5      # 训练时用的 action_scale（位置版，单位米）
KP, KD = 400.0, 40.0    # 训练时的 PD 增益
CTRL_DT = 1.0 / 60.0    # 训练时的控制周期
EPISODE_STEPS = 300     # 5 秒

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
n_sub = max(1, int(round(CTRL_DT / model.opt.timestep)))

policy = torch.jit.load(POLICY).eval()

# 初始条件：杆偏 0.1 rad（和 Isaac 的 reset 对齐）
data.qpos[0] = 0.0
data.qpos[1] = 0.1
mujoco.mj_forward(model, data)

for step in range(EPISODE_STEPS):
    # ---- 观测：顺序必须和 Isaac 一致：[θ, ω, x, v] ----
    obs = np.array([data.qpos[1], data.qvel[1], data.qpos[0], data.qvel[0]],
                   dtype=np.float32)

    with torch.no_grad():
        a = float(policy(torch.from_numpy(obs).reshape(1, -1)).item())

    # ---- 动作 → 位置目标 → PD → 力 ----
    q_des = a * ACTION_SCALE
    for _ in range(n_sub):
        F = KP * (q_des - data.qpos[0]) - KD * data.qvel[0]
        data.ctrl[0] = np.clip(F, -400, 400)
        mujoco.mj_step(model, data)

    if step % 30 == 0:
        print(f"t={step*CTRL_DT:4.1f}s  x={data.qpos[0]:+.3f}  "
              f"theta={data.qpos[1]:+.3f}  F={data.ctrl[0]:+7.1f}")

    # 终止：与 Isaac 一致
    if abs(data.qpos[1]) > np.pi / 2 or abs(data.qpos[0]) > 3.0:
        print(f"terminated at step {step}: |theta|={abs(data.qpos[1]):.2f}, "
              f"|x|={abs(data.qpos[0]):.2f}")
        break
```

**预期**：杆子保持直立，`theta` 在 ±0.05 内波动，`x` 缓慢漂移（因为策略只管平衡，不管位置）。

**可能报错**

| 现象                      | 原因                            | 处理                              |
| ----------------------- | ----------------------------- | ------------------------------- |
| 杆子立刻倒                   | 初始角符号反了 / 观测顺序错               | 打印前 5 步的 `obs`，和 Isaac 里同一时刻的对比 |
| 一开始就 terminated         | `qpos[1]` 超了 π/2              | 检查你的初始角设置                       |
| 力一直在 ±400 打满            | PD 增益或 `action_scale` 和训练时不一致 | 回看 T2.1 的三个数                    |
| `policy(...)` 报 shape 错 | 策略要 `float32` 且 shape `(1,4)` | 照抄上面                            |

**验收**：能闭环跑满 5 秒，并能打印 `theta` 的时间序列。

> 🖊️ **记录**
> 
> 我的动作接口是（力矩 / 位置+PD）：\_\_\_\_
> `ACTION_SCALE / KP / KD / CTRL_DT` 四个数：\_\_\_\_ / \_\_\_\_ / \_\_\_\_ / \_\_\_\_
> 5 秒后 `theta` 的最大值：\_\_\_\_；`x` 漂到：\_\_\_\_

---

## T3.3 对齐三件事：观测、缩放、频率

**目的**：sim2sim 失败，90% 是这三件事没对齐。这一任务就是排错清单。

**前置**：T3.2 能跑。

**检查清单**（一条一条验证，别跳）

| 项     | 怎么验证                                     | 常见错法                                             |
| ----- | ---------------------------------------- | ------------------------------------------------ |
| 观测顺序  | Isaac 里打印一次 obs，MuJoCo 里打印一次，同一初始状态应几乎一样 | 把 $[x,v,\theta,\omega]$ 当成 $[\theta,\omega,x,v]$ |
| 单位    | 角度是 rad 还是 deg；位置是 m 还是 cm               | 用 deg 喂给训练时用 rad 的策略                             |
| 动作缩放  | 位置版要乘 `action_scale`（米），力矩版乘（牛）          | 忘了乘，或乘了两次                                        |
| 控制频率  | Isaac 是 60 Hz；MuJoCo 用 `n_sub` 子步凑       | 直接 1 kHz 调策略（等于让策略"高频抖"）                         |
| 执行器饱和 | `ctrlrange` 与 `effort_limit_sim` 一致      | MuJoCo 用 ±1（没设 ctrlrange）                        |
| 阻尼/摩擦 | 两边 `damping` 一致                          | Isaac 的 USD 有阻尼而 MJCF 忘了写                        |

**做法**：每对齐一项，记录一次偏差。

**偏差怎么量**：同一初始状态（$\theta=0.1$，其余 0）、同一控制律（先不用策略，用固定 PD），跑 5 s，比较 $\theta(t)$ 的最大绝对差：

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
- [ ] T3.2 策略能在 MuJoCo 闭环 5 s
- [ ] T3.3 有对齐前后的偏差表
- [ ] T3.4 有参数表 + 拟合曲线 + 残差
- [ ] T3.5 有辨识前后对比
