# 第 3 阶段：MuJoCo 与系统辨识（5 个任务）

> 目标：把 Isaac 训出来的策略放到 MuJoCo 里跑通（sim2sim），然后用实测数据把仿真参数辨识出来。
> 前置：第 2 阶段完成，手里有一个从 tracking 任务训出来的 `policy.pt`（5 维观测、力矩接口）。

---

## T3.1 Isaac 风格的 cartpole MJCF + 键盘遥控脚本

**目的**：在 MuJoCo 里复刻出**和 Isaac 里同一个机器人**——几何、质量、关节名、执行器、单位，
甚至连"θ 的正方向"都要对齐；然后再写一个能开窗口、能用键盘改目标位置的遥控脚本。

**前置**：T2.1（知道 PD 参数），以及一个 tracking 任务导出的 `policy.pt`（5 维观测、力矩接口）。

> 一句话：sim2sim 不是"再画一个小车"，而是"把训练时的物理契约抄一遍"。
> 抄错任何一行，策略都会在 0.1 秒内把力矩打到 ±400 N 然后翻车。

---

### 第一步：先把 Isaac 那边的事实抄成一张表

这些数字都不用猜，全在仓库里：

| 对齐项        | Isaac 里的值                           | 从哪看                                      | MJCF 里怎么写                                             |
| ---------- | ----------------------------------- | ---------------------------------------- | ----------------------------------------------------- |
| 关节名        | `slider_to_cart` / `cart_to_pole`   | `isaaclab_assets/robots/cartpole.py`     | **同名** `<joint name=...>`                             |
| 车质量/尺寸     | 1.0 kg，约 0.40 × 0.20 × 0.15 m       | `cartpole.usd`                           | `<geom type="box" size="0.20 0.10 0.075" mass="1.0">` |
| 杆质量/长度     | 0.1 kg，1.0 m，半径 ≈0.03 m             | 同上                                       | `capsule fromto="0 0 0 0 0 1" size="0.03" mass="0.1"` |
| 滑动关节       | 阻尼 10 N·s/m，行程 ±3 m                 | `cart_actuator.damping` / `max_cart_pos` | `slide` + `range="-3 3"` + `damping="10"`             |
| 摆动关节       | 阻尼 0                                | `pole_actuator.damping=0`                | `hinge damping="0"`                                   |
| 执行器        | 纯力矩，±400 N，`action_scale=200`       | `effort_limit_sim` / `action_scale`      | `<motor ctrlrange="-400 400">`                        |
| 物理步长       | 1/120 s，decimation 2 → 控制 60 Hz     | `SimulationCfg(dt=1/120)`                | `timestep="0.002"`（一个控制周期 8 个物理步）                     |
| **θ 的正方向** | 杆往 **−x** 侧倾时 `cart_to_pole` **为正** | `cart_to_pole` 的关节轴                      | `hinge axis="0 -1 0"`（**不是** `0 1 0`，见坑 2）            |

最后一行是这一节最贵的知识点：**几何长得一样、符号不一样，策略照样翻车。**

---

### 第二步：先写"最小可用的物理骨架"

只保留决定动力学的东西，先确认能跑，再往上加外观。新建 `mujoco/cartpole.xml`：

```xml
<mujoco model="cartpole">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <worldbody>
    <body name="cart" pos="0 0 0.5">
      <!-- 滑动自由度：车只能沿 ±x 走，阻尼 10 对齐 Isaac 的 actuator damping -->
      <joint name="slider_to_cart" type="slide" axis="1 0 0"
             range="-3 3" limited="true" damping="10"/>
      <geom name="cart_box" type="box" size="0.20 0.10 0.075" mass="1.0"/>

      <body name="pole" pos="0 0 0.075">
        <!-- 摆动自由度：轴取 −y，让 θ>0 表示"杆往 −x 倒"，与 Isaac 同号 -->
        <joint name="cart_to_pole" type="hinge" axis="0 -1 0" damping="0"/>
        <geom name="pole_rod" type="capsule" fromto="0 0 0 0 0 1.0"
              size="0.03" mass="0.1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- 纯力矩：force = action × 200 N，饱和 ±400 N。注意：这里没有 PD！ -->
    <motor name="cart_force" joint="slider_to_cart" gear="1"
           ctrlrange="-400 400" ctrllimited="true"/>
  </actuator>
</mujoco>
```

**每个字段存在的理由（少一个会怎样）**

| 字段                            | 作用           | 写错/漏掉会怎样                                          |
| ----------------------------- | ------------ | ------------------------------------------------- |
| `timestep="0.002"`            | 物理步长         | 太大：数值发散；太小：一个控制周期跑很多步，慢                           |
| `slide` + `range` + `limited` | 车沿 x 走且有行程限位 | 没 `range`：车飞出场景（`\|x\|>3` 的终止条件形同虚设）              |
| `hinge axis="0 -1 0"`         | 杆绕 −y 轴转     | 写成 `0 1 0`：θ/ω 全都反号 → 策略往反方向推 → 立刻倒（坑 2）          |
| `mass`                        | 质量           | 车质量变了，同样力矩下加速度就变了；**T3.4 要辨识它**                   |
| `damping=10`（车关节）             | 被动阻尼         | 漏了 → 车像在冰面上，Isaac 里学到的"推一下会停下"失效                  |
| `motor` + `ctrlrange`         | 执行器与饱和       | 换成 `position` 执行器或漏 `ctrlrange`：力矩尺度差 200 倍，杆子被打飞 |
| `integrator="implicitfast"`   | 数值积分器        | 默认 Euler 在 0.002 s + 阻尼 10 下也能跑，但 implicitfast 更稳 |

**验证**

```bash
# 1) 能不能加载：自由度数 / 执行器数 / 总质量
python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mujoco/cartpole.xml'); \
           print(m.nq, m.nv, m.nu, m.body_subtreemass[1])"
# 期望：2 2 1 1.1      ← 2 个自由度、1 个执行器、总质量 1.1 kg

# 2) 直接看模型（这时还没有策略，杆会自己倒下是正常的）
python -m mujoco.viewer --mjcf=mujoco/cartpole.xml
```

| 输出不对          | 说明                                     |
| ------------- | -------------------------------------- |
| `nq=3`        | 用了 `freejoint` / quaternion 关节，写成了浮动基座 |
| `nu=0`        | 忘了 `<actuator>`，或 actuator 里的关节名拼错     |
| 总质量 `1.05` 之类 | 有"装饰件"偷偷带了质量（见第三步）                     |

**其他常见报错**

| 报错                               | 原因                                 |
| -------------------------------- | ---------------------------------- |
| `XML Error: ... unknown element` | 标签拼错，比如 `worldbody` 写成 `worldBody` |
| `bad range specification`        | `range` 少了上界，或 `limited` 拼错        |

---

### 第三步：再把它打扮成 Isaac 的样子（但不能碰动力学）

好看和"物理正确"在 MuJoCo 里可以同时满足，关键是用 **`class` + 零质量零碰撞** 把装饰件隔离：

```xml
<default>
  <!-- 装饰件模板：不接触、不碰撞、不参与质量统计 -->
  <default class="visual">
    <geom contype="0" conaffinity="0" group="2" density="0" mass="0"/>
  </default>
</default>

<!-- 用的时候一行就够 -->
<geom name="wheel_fl" type="cylinder" size="0.052 0.014" pos="0.11 0.108 -0.062"
      euler="90 0 0" material="dark" class="visual"/>
```

仓库里的 `mujoco/cartpole.xml` 就是按这个套路长出来的：

| 加了什么                     | 为什么加                     | 会改动力学吗               |
| ------------------------ | ------------------------ | -------------------- |
| 地面 + 棋盘格纹理               | 给一个空间参考系（Isaac 里也是深色网格地） | 不会（`class="visual"`） |
| 滑轨 + 两端限位块               | 车是"挂在轨上"的，不贴地滑           | 不会                   |
| 每 1 m 一道浅色刻线             | 键盘遥控时肉眼读 x               | 不会                   |
| 底盘 / 顶盖 / 四轮 / 铰链 / 杆顶小球 | 让人一眼看出"这是个车 + 这是个摆"      | 不会                   |
| 目标点用 `mocap` body        | **不受物理影响**，脚本随时能挪        | 不会（mocap 不参与动力学）     |
| 深色天光 + 雾                 | 远处不出现生硬的边界               | 不会                   |

> ⚠️ **最常见的隐形 bug**：装饰底盘写成 `mass="0.05"`，车总质量就从 1.0 变成 1.05 kg。
> 策略是按 1.0 kg 学的，同样的力矩下去加速度少 5%，单看不明显；
> 等你去 T3.3 对数时会误判成"观测没对齐"，白查两小时。
> 一行自查：`print(m.body_subtreemass[1])` 必须正好是 `1.1`。

顺便记两个"画面脏"的坑（都很像物理坏了，其实只是渲染）：

| 现象        | 原因                                                             | 处理                                           |
| --------- | -------------------------------------------------------------- | -------------------------------------------- |
| 地面一大片脏条纹影 | `<light>` 的 `castshadow` 默认是 **True**，巨大 plane 上会出 shadow acne | 两盏灯都写 `castshadow="false"`（车挂在轨上，地上本来也不该有影子） |
| 地面像镜子、有鬼影 | 给 plane 设了 `reflectance`                                       | `reflectance="0"`                            |

---

### 第四步：写遥控脚本（这才是能"玩"起来的关键）

`mujoco/sim2sim.py` 已经写好了：

```bash
# 开窗口 + 键盘遥控：←/→ 改目标位置，车自己追过去
python mujoco/sim2sim.py

# 指定策略 / 初始目标
python mujoco/sim2sim.py --policy logs/rsl_rl/cartpole_direct/<run>/exported/policy.pt --target 1.5

# 不开窗口跑 5 秒（= Isaac 一个 episode），打印指标并落 CSV
python mujoco/sim2sim.py --headless --duration 5 --csv /tmp/run.csv
```

**键盘**

| 按键              | 作用                    |
| --------------- | --------------------- |
| `←`/`A`、`→`/`D` | 目标左 / 右移 0.10 m       |
| `,` / `.`       | 目标左 / 右移 0.02 m（微调）   |
| `↓` / `↑`       | 目标左 / 右移 0.50 m（大跳）   |
| `Space`         | 暂停 / 继续               |
| `R`             | 重置 episode（车回 0、杆角随机） |
| `Backspace`     | 目标归零                  |
| `Q` / `Esc`     | 退出                    |

窗口里实时显示 `target / x / err / theta / force`；鼠标左键拖=旋转，右键拖=平移，滚轮=缩放。

**三段核心代码**（其余都是外壳）

1）观测拼装——顺序和符号就是契约本身：

```python
def observation(self, target: float) -> np.ndarray:
    # [theta, omega, x, v, target - x]：第 5 维必须是 target - x，不能反
    return np.array([self.theta, self.omega, self.x, self.v, target - self.x],
                    dtype=np.float32)
```

2）动作 → 力矩（没有 PD，一步都不能多）：

```python
force = float(np.clip(action * ACTION_SCALE, -EFFORT_LIMIT, EFFORT_LIMIT))  # 200 / 400
self.data.ctrl[0] = force
```

3）目标点用 mocap；键盘改的是"目标标量"，不是刚体：

```python
def on_key(self, key: int) -> None:
    if key in (263, ord("A"), ord("a")):      # ← 或 A
        self.nudge(-0.10)
    elif key in (262, ord("D"), ord("d")):    # → 或 D
        self.nudge(+0.10)

# 主循环里：mocap 只影响渲染，不参与动力学
self.data.mocap_pos[self.mocap_id] = (target, 0.0, self.marker_z)
```

> 💡 旧版本里写 `data.xpos[target_marker] = ...` 是**错的**：`xpos` 是 `mj_forward`
> 每帧算出来的派生量，写进去下一帧就被覆盖，标记点看起来"纹丝不动"。
> 要挪一个纯装饰的物体，用 **mocap body**，别去改 `xpos`。

**实测（本仓库 `2026-11-04_17-52-24` 的 5 维策略，每个目标跑 5 s）**

| 目标 x* | 最终 x   | 最终 \|err\| | 平均 \|err\| | 到位率(\|err\|<0.1) |
| ----- | ------ | ---------- | ---------- | ---------------- |
| −2.0  | −2.008 | 0.008      | 0.179      | 72.0%            |
| 0.0   | −0.020 | 0.020      | 0.030      | 100.0%           |
| +1.0  | +0.963 | 0.037      | 0.181      | 76.7%            |
| +2.0  | +1.910 | 0.090      | 0.268      | 59.3%            |

读法：末态误差都很小（策略确实学会了跟踪），**平均**误差大是因为前 1 秒在赶路；
目标越远，赶路时间占比越大，"到位率"自然越低——这是指标口径问题，不是策略崩了。

---

### 第五步：两个一定会踩的坑

#### 坑 1：`ImportError: Error importing numpy ... _multiarray_umath`

```text
Python 版本：3.13（来自 /home/cofallen/miniconda3/bin/python）
NumPy  版本：1.26.4（来自 .../envs/isaac/lib/python3.10/site-packages）
```

不是 numpy 坏了，而是 **`PYTHONPATH` 把 isaac 环境（py3.10）的 site-packages
塞给了 base 环境（py3.13）**，C 扩展的 ABI 对不上，谁都救不了：

```bash
echo $PYTHONPATH
# /home/cofallen/IsaacLab/source:.../envs/isaac/lib/python3.10/site-packages   ← 就是它
```

| 解法                                | 命令                                                                      |
| --------------------------------- | ----------------------------------------------------------------------- |
| **① 用带 mujoco 的解释器**（本仓库约定，推荐）    | `/home/cofallen/miniconda3/envs/isaac/bin/python mujoco/sim2sim.py`     |
| ② 在当前 shell 临时清掉污染（只跑 mujoco 时够用） | `env -u PYTHONPATH python mujoco/sim2sim.py`                            |
| ③ 永久修（会影响 Isaac Lab 的 IDE 联动，谨慎）  | 把 `.bashrc` 里那句 `export PYTHONPATH=...` 改成只在 `conda activate isaac` 后生效 |

#### 坑 2：策略"刚跑 0.1 秒就把力矩打到 ±400 N，杆子乱飞"

现象（当初的报错现场就是这样）：

```text
t=0.10  x=-1.304  v=-22.644  theta=1.903  omega=22.980  F=-400.0
```

**不是质量错，也不是 `action_scale` 错**，而是观测的**符号约定**错了。
排查方法很土但很有效——把可能反号的 4 个量（θ、ω、x、v）做 16 组扫描：

```python
import itertools
for s_th, s_om, s_x, s_v in itertools.product((1, -1), repeat=4):
    # obs = [s_th*theta, s_om*omega, s_x*x, s_v*v, s_x*(target - x)]
    ...  # 每组跑 5 s，记录存活时长和平均 |err|
```

实测只有一组活下来：

| $s_\theta$ | $s_\omega$ | $s_x$  | $s_v$  | 存活         | 平均 \|err\| |
| ---------- | ---------- | ------ | ------ | ---------- | ---------- |
| +1         | +1         | +1     | +1     | 0.08 s     | 1.407      |
| +1         | +1         | −1     | +1     | 0.10 s     | 0.564      |
| −1         | −1         | **+1** | **+1** | **4.98 s** | **0.165**  |
| −1         | −1         | −1     | −1     | 1.60 s     | 1.312      |

结论：**Isaac 的 `cart_to_pole` 正方向是"杆往 −x 侧倾"**，与 MuJoCo 默认的
`axis="0 1 0"` 相反。两种修法，选一种，别两边都改：

| 修法                                           | 优点                 | 缺点                  |
| -------------------------------------------- | ------------------ | ------------------- |
| **改模型**：`axis="0 -1 0"`（本仓库采用）               | 脚本保持教科书式的 obs，不藏魔法 | 模型里必须写注释，否则下一个人一脸问号 |
| 改脚本：`obs = [-theta, -omega, x, v, target-x]` | 模型文件干净             | 反号散在代码里，换任务时容易忘     |

> 🧠 通用版经验：**"符号错"和"参数错"的表现完全不同。**
> 参数错 → 慢慢漂、精度差、偶尔倒；符号错 → 第一帧就饱和、0.1 秒结束。
> 看到力长期贴在 `±ctrlrange` 上，先查符号，再查参数。

---

**验收**：

1. 模型能加载，`nq=2, nv=2, nu=1`，总质量 `1.1 kg`；
2. 开窗口后能用键盘把目标移到 ±2 m，并看着车追过去（不是原地不动、也不是翻车）；
3. 能解释 `hinge axis="0 -1 0"` 为什么不是 `0 1 0`；
4. 能说出坑 1 里 `PYTHONPATH` 为什么会把 py3.10 的 numpy 喂给 py3.13。

> 🖊️ **记录**
> 
> `nq, nv, nu, 总质量` 输出：\_\_2\_\_ / \_\_2\_\_ / \_\_1\_\_ / \_\_1.1\_\_
> 我的车 / 杆 质量与长度：\_\_1.0 \_0.1*0.1*0.1\_ / \_\_0.1 1.0\_\_
> 我用的是 `axis="0 _-1__0"`，理由是：\_\isaacsim 和mujoco joint方向相反_\_\_
> 键盘把目标设到 __1.0_ m，最终 x = __0.963_ m，|err| = __-0.037_ m
> 我的 `PYTHONPATH` 里有没有混进别的环境的 site-packages：\_没有_\_\_

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

**操作**：先写一个**最小闭环**（40 行，只回答"契约对不对"）。
仓库里的 `mujoco/sim2sim.py` 是这个版本的"加壳版"：把常量、观测、推理、步进
拆成 `CartpoleSim` 的方法，再套上键盘遥控 / HUD / headless / CSV 输出。
先跑最小版确认策略没训歪，再去看完整脚本。

```python
"""最小 sim2sim：只验证契约对不对。

    obs    = [theta, omega, x, v, target - x]       5 维，float32
    action = 1 维力矩，force = action * action_scale(200 N)，饱和 ±400 N
    频率   = 60 Hz（一个控制周期 = 8 个 0.002 s 物理步）
"""
import numpy as np
import mujoco
import torch

ACTION_SCALE = 200.0        # 训练时的 action_scale [N]
CTRL_DT = 1.0 / 60.0        # = sim.dt(1/120) × decimation(2)
TARGET = 1.0                # 部署时由你给的实参 x*
MJCF = "mujoco/cartpole.xml"
POLICY = "logs/rsl_rl/cartpole_direct/<run>/exported/policy.pt"

m = mujoco.MjModel.from_xml_path(MJCF)
d = mujoco.MjData(m)
n_sub = max(1, int(round(CTRL_DT / m.opt.timestep)))     # 8

def addr(name):                                          # 用名字取地址，别硬编码 0/1
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])

qc, vc = addr("slider_to_cart")     # 车位 / 车速
qp, vp = addr("cart_to_pole")       # 杆角 / 杆角速度

policy = torch.jit.load(POLICY).eval()

# 初始条件对齐 Isaac 的 _reset_idx：x = 0，杆角在 ±0.1π 内均匀采样
rng = np.random.default_rng(0)
d.qpos[qc] = 0.0
d.qpos[qp] = rng.uniform(-0.1, 0.1) * np.pi
mujoco.mj_forward(m, d)

errs = []
for step in range(300):                                  # 300 × 1/60 s = 5 s
    # ---- 观测：顺序一致，第 5 维必须是 (target - x) ----
    obs = np.array([d.qpos[qp], d.qvel[vp], d.qpos[qc], d.qvel[vc],
                    TARGET - d.qpos[qc]], dtype=np.float32)

    with torch.no_grad():
        a = float(policy(torch.from_numpy(obs)[None]).item())

    # ---- 动作就是力矩，直接乘 action_scale；这里没有 PD ----
    d.ctrl[0] = float(np.clip(a * ACTION_SCALE, -400.0, 400.0))

    for _ in range(n_sub):
        mujoco.mj_step(m, d)

    errs.append(abs(d.qpos[qc] - TARGET))

    # 终止条件与 Isaac 的 _get_dones 一致
    if abs(d.qpos[qp]) > np.pi / 2 or abs(d.qpos[qc]) > 3.0:
        print(f"terminated at step {step}")
        break

print(f"final x = {d.qpos[qc]:+.3f} m | mean|err| = {np.mean(errs):.3f} m | "
      f"到位率(|err|<0.1) = {np.mean(np.array(errs) < 0.1):.1%}")
```

跑通了再切到仓库版（接口一样，功能更全）：

| 完整版多的东西                   | 为什么需要                                            |
| ------------------------- | ------------------------------------------------ |
| `--headless --duration 5` | 不开窗口批量跑，用来看"到位率"这类指标                             |
| `--csv`                   | 把 `t,x,v,theta,omega,force,target` 落盘，T3.4 辨识直接用 |
| 键盘改目标（`TargetControl`）    | 现场把 $x^*$ 挪来挪去，肉眼看跟踪质量                           |
| `R` / `Space`             | 重置、暂停；终止时自动重开，方便连续试                              |
| HUD + 实时节拍                | 屏幕上直接读 `err / theta / force`，不用盯终端               |

> 💡 建议：**先用最小版跑一次 `TARGET = 0`**。如果连"稳在 0"都做不到，
> 说明策略本身没学好（或观测/力矩接口还有问题），这时加多少交互都没意义。

**预期**

| 现象                                          | 含义                                                           |
| ------------------------------------------- | ------------------------------------------------------------ |
| `x` 收敛到 `TARGET` ±0.1 并稳住，`theta` 在 ±0.05 内 | 契约对齐了，且策略确实学会了跟踪                                             |
| `x` 停在 `TARGET` 的某个比例（比如 0.7·TARGET）        | 策略只学到"往目标走"，精度不够 → 回 T2.4 改 reward 或加训练轮数                    |
| `theta` 立刻发散                                | 观测顺序 / 第 5 维构造 / **力矩 vs PD 接口**抄错 / **关节轴符号反了**（见 T3.1 坑 2） |
| `x` 往反方向跑                                   | 第 5 维符号写反了（应是 `target - x`，不是 `x - target`）                  |
| `x` 一直贴着 0 不动                               | 策略根本没学过 tracking（回到 T2.3 检查 `rew_target` 是否进了 `return`）      |

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
> 我的 `policy.pt` 来自哪个 run、观测几维：\_\5_\_\_
> `ACTION_SCALE / CTRL_DT / damping` 三个数：\_\_\200_\_ / \_\_0.002_\_ / \_\_1.0_\_
> 我测的 `TARGET` 和最终 `x`：\_\_1.0\_\_ → \_\_0.963_\_；到位率：\_\_不算_\_
> 方法 A 的到位率：\_\_高\_\_；方法 B 的到位率：\_\低_\_\_
> 方法 B 失效的话，卡在哪个前提（①②③④）：\_\_\1_\_  B压根没有对x做reward.

---

## T3.3 对齐三件事：观测、缩放、频率

**目的**：sim2sim 失败，90% 是这三件事没对齐。这一任务就是排错清单。

**前置**：T3.2 能跑。

**检查清单**（一条一条验证，别跳）

| 项         | 怎么验证                                         | 常见错法                                             |
| --------- | -------------------------------------------- | ------------------------------------------------ |
| 观测顺序      | Isaac 里打印一次 obs，MuJoCo 里打印一次，同一初始状态应几乎一样     | 把 $[x,v,\theta,\omega]$ 当成 $[\theta,\omega,x,v]$ |
| 观测维数      | tracking 任务是 **5 维**，第 5 维 = `target - x`    | 只喂 4 维（用错 `policy.pt`），或第 5 维写成 `x - target`     |
| 单位        | 角度是 rad 还是 deg；位置是 m 还是 cm                   | 用 deg 喂给训练时用 rad 的策略                             |
| 动作缩放      | 力矩版：`force = a * 200`；位置版：`q_des = a * 0.5`  | 忘了乘、乘了两次，或搞混 200 和 0.5                           |
| 控制频率      | Isaac 是 60 Hz；MuJoCo 用 `n_sub` 子步凑           | 直接 1 kHz 调策略（等于让策略"高频抖"）                         |
| 执行器饱和     | `ctrlrange` 与 `effort_limit_sim` 一致（±400）    | MuJoCo 用 ±1（没设 ctrlrange）                        |
| 阻尼/摩擦     | 两边 `damping` 一致                              | Isaac 的 USD 有阻尼而 MJCF 忘了写                        |
| **关节正方向** | 同一姿态下打印 Isaac 与 MuJoCo 的 $(q,\dot q)$，符号必须相同 | 轴写成 `0 1 0`，θ/ω 反号 → 力立刻打满 ±400（T3.1 坑 2）        |

**做法**：每对齐一项，记录一次偏差。

**偏差怎么量**：同一初始状态（$\theta=0.1$，其余 0）、同一控制律（先不用策略，用固定力矩，比如 $F=50$ N），跑 5 s，比较 $\theta(t)$ 的最大绝对差：

$$
e = \max_t \left|\theta^{\text{Isaac}}(t) - \theta^{\text{MuJoCo}}(t)\right|
$$

**验收**：给出"对齐前 / 对齐后"的偏差表；能说出每一行对应哪个错误。

> 🖊️ **记录**
> 
> | 版本        | 观测顺序 | 缩放  | 频率    | 偏差 $e$ |
> | --------- | ---- | --- | ----- | ------ |
> | 第一版（随便写的） | 5    | 200 | 0.002 | 0.37   |
> | 对齐后       | 5    | 200 | 0.002 | 0.37   |
> 
> 我踩的三个坑：
> &nbsp;没有踩坑，主要是joint isaacsim 和mujoco方向不同。
> 
> sim2sim就是吧部署好的，放Mujoco走一遍，注意damping joint方向，dt 和mass, 就是简单的obs step即可。记得action有系数。&nbsp;

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
