以下是将你提供的完整内容整理为 Markdown 的版本：

> **校订（2026-11-01，按你自己的 cartpole 日志校过）**
>
> 这份文档是通用教程，示例用的是 ANYmal-C。有三处和你的实际日志对不上，已在正文对应小节加了 ⚠：
>
> 1. **`Loss/entropy` 会是负数，正常。** 高斯策略 `H = ln(σ√(2πe)) = ln(4.133σ)`，σ<0.242 后必然为负。你 17:00 那轮末值 `-1.939` 对应 σ=0.035，分毫不差。
> 2. **`Loss/learning_rate` 不是"衰减"**，是在 `1e-5 ~ 1e-2` 之间来回跳（KL 刹车）。
> 3. **Direct 环境没有 `Episode_Reward/*`**，要自己在 `_get_rewards` 里往 `self.extras["log"]` 写。
>
> 指标定义的完整版见 [实验台账](../08-实验台账.md) 第一节。

---

# Isaac Lab + RSL-RL 训练可视化说明

> 📦 **归档文档**（历史版本，仅供查证）。主线见 [从这里开始](../00-开始这里.md)。

> 🧭 返回 [学习总纲](../00-开始这里.md) ｜ 查数据 [实验台账](../08-实验台账.md) ｜ 查指标 [指标手册](../06-指标速查.md)

既然你使用的是 Isaac Lab + RSL-RL，我们就以机器人强化学习（比如 Anymal-C 四足机器人行走）为例，从实际训练代码、TensorBoard 页面和曲线含义三个方面来讲。

先明确一个核心问题：

> **TensorBoard 不只是看 Reward 有没有上升，而是用来判断机器人是否学会了任务、训练是否稳定，以及策略网络有没有出现异常。**

在 Isaac Lab + RSL-RL 中，通常使用 PPO（Proximal Policy Optimization）算法训练机器人。TensorBoard 记录的是 PPO 训练过程中产生的各种统计信息。

下面按照实际使用顺序讲解。

## 一、先理解 Isaac Lab 的训练数据从哪里来

整体流程如下：

```text
Isaac Lab 仿真环境
        ↓
机器人运动、接触、速度、奖励计算
        ↓
RSL-RL PPO 算法
        ↓
Actor、Critic、策略更新、梯度计算
        ↓
Logger 日志记录
        ↓
Reward、Loss、Entropy、FPS 等
        ↓
TensorBoard
        ↓
可视化训练过程、分析收敛情况
```

这里需要注意：

- Isaac Lab 负责机器人仿真环境和奖励计算。
- RSL-RL 负责 PPO 强化学习算法。
- TensorBoard 负责展示训练数据。

## 二、启动训练并找到日志

以 Anymal-C 平地行走任务为例。

### 1. 启动训练

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Anymal-C-v0 \
    --headless
```

如果你使用的是其他机器人任务，把 `task` 换成对应任务即可。

### 2. 查看日志目录

一般情况下，日志保存在：

```text
logs/rsl_rl/
```

例如：

```text
logs/
└── rsl_rl/
    └── anymal_c_flat/
        ├── 2026-11-01_10-00-00/
        │   ├── model_0.pt
        │   ├── model_100.pt
        │   ├── model_200.pt
        │   └── events.out.tfevents.xxxxx
        │
        └── 2026-11-01_11-00-00/
            ├── model_0.pt
            ├── model_100.pt
            └── events.out.tfevents.xxxxx
```

其中：

| 文件 | 作用 |
|---|---|
| `model_100.pt` | 第 100 次迭代保存的模型 |
| `events.out.tfevents` | TensorBoard 训练日志 |
| `params/` | 实验配置和参数记录 |

> 注意：模型保存间隔取决于训练配置，不一定是每 100 次。

### 3. 启动 TensorBoard

在另一个终端运行：

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

如果你在远程服务器训练，使用 SSH 端口转发：

```bash
ssh -L 6006:localhost:6006 user@服务器IP
```

然后在本地访问同一个地址。

## 三、TensorBoard 页面怎么看？

打开 TensorBoard 后，主要关注顶部的几个菜单。

| 页面 | 作用 | 重要程度 |
|---|---|---|
| Scalars | 查看训练指标曲线 | 最重要 |
| Graphs | 查看计算图 | 一般不常用 |
| Histograms | 查看参数或数据分布 | 进阶 |
| Distributions | 查看参数分布变化 | 进阶 |
| HParams | 对比超参数实验 | 进阶 |

刚开始学习，建议你主要使用 **Scalars**。

在 Scalars 页面，你会看到：

1. 左侧：指标名称（Tags）。
2. 中间：指标曲线。
3. 上方：实验名称（Runs）。
4. 曲线下方：Step、Value 等信息。
5. 侧边或设置区域：Smoothing（平滑系数）。

### 3.1 先认识坐标轴

假设你打开了：

```text
Train/mean_reward
```

横坐标（X 轴）：

- Step：训练迭代次数（通常是 PPO iteration）。

纵坐标（Y 轴）：

- Value：当前记录的平均奖励。

例如：

| Iteration | Mean Reward |
|---:|---:|
| 0 | -15 |
| 100 | 35 |
| 200 | 120 |
| 300 | 260 |
| 400 | 350 |
| 500 | 420 |

这意味着机器人在训练过程中逐渐获得更高的奖励。

> 注意：这里的 Step 通常不是单个仿真环境的物理步数，而是 Logger 写入数据时使用的训练迭代编号。

### 3.2 Smoothing 是什么？

这是 TensorBoard 中很重要的功能。

例如，原始 Reward 曲线可能是：

```text
100
120
110
150
130
170
145
200
```

训练数据本身有随机性，所以曲线经常上下波动。

Smoothing 可以让曲线更加平滑，帮助你观察整体趋势。

建议：

- `0`：查看原始数据。
- `0.5`：轻度平滑。
- `0.8`：观察整体趋势。
- `0.95`：高度平滑。

我的建议是：分析训练是否收敛时，先看 `0.8` 左右的平滑曲线，再切换到 `0` 检查原始曲线。

不要只看平滑曲线，因为过度平滑可能掩盖训练崩溃或突然出现的异常。

## 四、最重要的指标：Reward

这是机器人强化学习最值得关注的指标。

RSL-RL 中常见的 TensorBoard 标签是：

```text
Train/mean_reward
```

它表示最近一段时间内完成的 Episode 的平均累计奖励。

### 4.1 Reward 曲线如何判断？

我们用四种典型情况理解。

#### 情况 A：Reward 持续上升

通常是好现象。

说明策略可能逐渐学会任务，继续观察是否进入稳定平台期。

#### 情况 B：Reward 一直不变

可能没有学到东西。

可能原因：

- 奖励函数设计不合理。
- 学习率不合适。
- 机器人一直没有完成任务。
- 观测或动作设计有问题。

#### 情况 C：Reward 先上升后下降

需要排查。

可能是 PPO 更新不稳定、奖励尺度变化或训练环境发生了问题。

#### 情况 D：Reward 上升后趋于稳定

可能已经收敛。

说明策略可能进入稳定阶段，但还要进行实际机器人运动测试。

### 4.2 Reward 高就代表机器人训练成功吗？

不一定。

例如四足机器人行走任务，机器人可能通过以下方式获得奖励：

- 跟踪目标线速度。
- 跟踪目标角速度。
- 保持机身姿态。
- 减少关节运动过度。
- 减少脚部滑动。
- 减少不合理的接触。

最终 Reward 高，说明奖励函数所定义的目标得到了较好的优化。

但不一定代表机器人行走自然、稳定或者能够应对复杂地形。

所以还需要观察奖励分项，以及实际运动效果。

## 五、Episode Length：机器人每次能坚持多久？

TensorBoard 中通常是：

```text
Train/mean_episode_length
```

它表示最近完成的 Episode 的平均长度。

例如：

| 训练阶段 | Episode Length |
|---|---:|
| 初期 | 20 |
| 中期 | 100 |
| 后期 | 500 |

如果是四足机器人行走任务，长度不断增加，可能意味着机器人摔倒减少，能够持续运动更久。

但是要注意一个重要细节：

**Episode Length 并不是越大越好。**

假如你的任务设置了 1000 步的 Episode 上限：

- 100 步结束：机器人可能很快摔倒。
- 500 步结束：机器人可能中途失败。
- 1000 步结束：机器人可能完成整个回合。

但是对于目标到达任务，机器人成功到达目标后也可能提前结束 Episode。因此需要结合任务的终止条件判断。

另外，如果 Episode Length 一直达到最大值，并不一定意味着策略真的优秀，还需要检查成功率和奖励。

## 六、Loss：PPO 网络是否正常学习？

这一部分是很多初学者最容易困惑的。

RSL-RL 使用 PPO 算法，主要包含两个网络：

### Actor（策略网络）

输入机器人观测 → 输出机器人动作。

例如：四条腿各个关节的目标动作。

流程：

```text
机器人仿真环境
        ↓
执行动作，计算 Reward
        ↓
Critic（价值网络）
        ↓
评估当前状态的价值，预测未来累计回报
```

简单理解：

- Actor：负责决定机器人怎么动。
- Critic：负责评价当前状态好不好。

PPO 训练时，两个网络会不断优化。

### 6.1 Value Function Loss

TensorBoard 标签：

```text
Loss/value_function
```

它表示 Critic 价值网络的损失。

Critic 的工作是预测状态价值：

$$
V(s_t)\approx
\mathbb{E}\left[
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\right]
$$

其中：

- \(s_t\)：当前状态。
- \(r_t\)：未来奖励。
- \(\gamma\)：折扣因子。

Value Function Loss 衡量价值网络预测和训练目标之间的误差。

如何判断？

- 初期较高：通常正常，Critic 还没有学好。
- 中期逐渐下降：通常说明价值预测正在改善。
- 后期稳定：可能已经趋于稳定。
- 突然大幅上升：需要检查训练是否不稳定。

但这里有一个重要知识：

**Value Loss 不一定需要降到 0。**

因为机器人环境具有随机性，未来奖励也会变化，Critic 不可能完美预测所有状态。

### 6.2 Surrogate Loss

TensorBoard 标签：

```text
Loss/surrogate
```

这是 PPO 的策略优化损失，也就是 Actor 的核心优化目标。

PPO 使用 Clipped Surrogate Objective：

$$
L^{CLIP}(\theta)=
\mathbb{E}_t
\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)
\hat A_t
\right)
\right]
$$

其中：

- \(r_t(\theta)\)：新旧策略概率比。
- \(\hat A_t\)：优势函数。
- \(\epsilon\)：裁剪参数。

你不需要一开始就记住公式，先理解：

它主要衡量 PPO 策略更新的优化目标。

例如：

```text
Surrogate Loss = -0.05
Surrogate Loss = -0.02
Surrogate Loss = -0.01
```

不要认为 Loss 越接近 0，机器人就一定训练得越好。

因为它与优势函数、策略更新方向和数据分布有关。

更重要的是观察：

- 是否出现剧烈震荡。
- 是否出现 NaN。
- 是否突然出现异常大的数值。
- 是否和 Reward 的变化同时出现异常。

### 6.3 Entropy Loss

TensorBoard 中可能看到：

```text
Loss/entropy
```

它与策略的随机性有关。

强化学习刚开始时，机器人需要探索不同动作，因此策略通常具有较高的随机性。

训练后期，机器人逐渐找到有效动作，策略可能变得更加确定。

一般来说：

- Entropy 较高：策略随机性较强。
- Entropy 逐渐降低：策略可能逐渐减少随机探索。
- **Entropy 变成负数：正常，不是 bug。** 高斯策略的微分熵是 `H = ln(σ√(2πe)) ≈ ln(4.133σ)`，σ < 0.242 之后就是负的（微分熵可以为负，和香农熵不同）。
- Entropy 很快接近 0 或很负：确实代表探索已经很弱，但**判断"探索死没死"要看 `Policy/mean_noise_std` 本身**，不要看熵的正负。

> ⚠ 校订：原文写"RSL-RL 不同版本符号不同"。实际你这一版（`rsl_rl/algorithms/ppo.py:408`）记录的就是**真实微分熵**，所以你会看到 `-1.94` 这种值。验证：σ=0.035 → `ln(4.133×0.035) = -1.92`，和你 17:00 那轮日志末值 `-1.939` 一致（误差来自 σ 的取值精度）。
>
> 另外记住：`entropy_coef=0.005` 看起来很小，但 `H≈1.4` 时它对 loss 的贡献是 `0.007`，而 surrogate 本身才 `0.01` 量级 —— **熵项和策略项同量级**，它才是 `std` 缓慢下降的主要推手。

## 七、Mean Action Noise Std：动作噪声标准差

这是机器人强化学习中非常重要的一个指标。

常见标签：

```text
Policy/mean_std
```

部分 RSL-RL 版本会使用：

```text
Policy/mean_noise_std
```

为什么需要关注它？

因为 PPO 的 Actor 通常输出一个动作分布，而不是直接输出完全确定的动作。

例如：

$$
a\sim\mathcal{N}(\mu,\sigma^2)
$$

其中：

- \(\mu\)：Actor 预测的动作均值。
- \(\sigma\)：动作分布的标准差。

标准差越大，动作探索范围通常越大。

假设训练过程是：

| Iteration | Mean Std |
|---:|---:|
| 0 | 1.00 |
| 100 | 0.85 |
| 200 | 0.70 |
| 300 | 0.50 |
| 400 | 0.30 |
| 500 | 0.20 |

可以理解为：

训练开始时，机器人探索不同动作的幅度较大。

随着训练进行，策略逐渐稳定，动作噪声可能逐渐降低。

但需要注意：

- 太快下降：可能探索不足。
- 一直很大：可能策略还没有稳定下来。
- 降得太低：可能限制后续探索。

对于四足机器人，最好结合 Reward、实际行走表现一起判断。

## 八、Learning Rate：学习率

常见标签：

```text
Loss/learning_rate
```

学习率控制 PPO 更新网络参数的步幅。

例如：

```python
@configclass
class AnymalCRoughPPORunnerCfg:
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.01,
    )
```

> 这是配置示意，具体字段要以你当前任务的 Runner 配置为准。

学习率过大：

- 网络更新可能不稳定。
- Reward 可能大幅震荡。
- 策略可能发生退化。

学习率过小：

- 网络学习速度可能很慢。
- Reward 长时间没有明显改善。

特别注意：

如果 `schedule="adaptive"`，学习率**不是一条水平线，也不是一条单调下降的线**，而是在地板和天花板之间**来回跳**。你这一版的实现（`rsl_rl/algorithms/ppo.py:292-297`）：

```python
if kl_mean >  desired_kl * 2.0:              # 0.02
    lr = max(1e-5, lr / 1.5)                 # 刹得太猛 → 砍
elif kl_mean < desired_kl / 2.0 and kl_mean > 0.0:   # 0.005
    lr = min(1e-2, lr * 1.5)                 # 走得太慢 → 涨
```

所以：

- 看到 lr 掉到 `1e-5`：说明上一批数据让策略想大改，被一键刹停。
- 看到 lr 跳回 `1e-2`：说明更新量已经很小，放开跑。
- **它反映的是"这批数据有多出人意料"，不是"训练到了第几阶段"。** 不要拿它当收敛程度读数。

> ⚠ 校订：你的 9 个 run 末轮 lr 基本都在 `1e-5 ~ 1e-4`，`lr_min` 全是 `1e-5`（地板）。这是正常的跳跃结果，不是"学习率衰减到停了"。

## 九、FPS：训练速度

常见标签：

```text
Perf/total_fps
Perf/collection_time
Perf/learning_time
```

这三个指标主要用来判断训练效率。

| 指标 | 意义 |
|---|---|
| `total_fps` | 训练整体吞吐量 |
| `collection_time` | 收集环境数据的耗时 |
| `learning_time` | PPO 网络更新的耗时 |

例如：

```text
total_fps = 50000
collection_time = 1.2 s
learning_time = 0.3 s
```

说明这次迭代中：

- 数据收集耗时约 1.2 秒。
- 网络学习耗时约 0.3 秒。
- 整体耗时约 1.5 秒。

如果 `collection_time` 很高，可能是仿真、观测计算或环境步进成为瓶颈。

如果 `learning_time` 很高，可能是网络规模、Batch Size 或 GPU 计算成为瓶颈。

**FPS 高不代表机器人学得好，它只是反映训练速度。**

## 十、实际项目中非常重要：Reward 分项曲线

这一部分我特别建议你掌握。

假设你训练四足机器人行走，奖励函数可能包括：

```python
rewards = {
    "track_lin_vel_xy": ...,
    "track_ang_vel_z": ...,
    "lin_vel_z_l2": ...,
    "ang_vel_xy_l2": ...,
    "dof_torques_l2": ...,
    "feet_air_time": ...,
    "feet_slide": ...,
    "undesired_contacts": ...,
}
```

> 这些是奖励项名称示意，不是可以直接运行的完整配置。

它们可能对应：

| Reward 项 | 作用 |
|---|---|
| `track_lin_vel_xy` | 跟踪目标线速度 |
| `track_ang_vel_z` | 跟踪目标角速度 |
| `lin_vel_z_l2` | 惩罚垂直方向速度 |
| `ang_vel_xy_l2` | 惩罚机身横滚、俯仰角速度 |
| `dof_torques_l2` | 惩罚过大的关节力矩 |
| `feet_air_time` | 鼓励合理抬脚 |
| `feet_slide` | 惩罚脚部滑动 |
| `undesired_contacts` | 惩罚不希望出现的碰撞 |

假设：

- 总 Reward 上升。
- 速度跟踪 Reward 上升。
- 关节力矩惩罚逐渐减小。
- 脚部滑动惩罚逐渐减小。

这通常比单纯看到总 Reward 上升更有分析价值。

### 10.1 这些奖励分项是怎么进入 TensorBoard 的？

在 Isaac Lab 的 Manager-Based 环境中，奖励项通常由 RewardManager 管理。

例如：

```python
@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"std": 0.5},
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
    )
```

环境会将 Episode 统计信息传递给 RSL-RL。

RSL-RL Logger 再将相应统计量写入 TensorBoard。

通常你会看到类似：

```text
Episode_Reward/track_lin_vel_xy
Episode_Reward/feet_slide
Episode_Reward/...
```

但具体 Tag 命名取决于 Isaac Lab 版本、任务实现和 Logger 代码。并不是所有 Reward 项都会自动以同样的名称记录。

## 十一、如何在代码中找到这些指标？

这一步很重要。以后你修改 Isaac Lab 的奖励函数、PPO 参数，就可以知道 TensorBoard 上的变化究竟来自哪里。

### 11.1 找到 Runner 配置

通常在任务对应的 RSL-RL 配置文件中，例如：

```text
your_project/
└── source/
    └── your_project/
        └── tasks/
            └── locomotion/
                └── agents/
                    └── rsl_rl_ppo_cfg.py
```

找到类似配置：

```python
@configclass
class MyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

这里主要关注：

| 参数 | 作用 |
|---|---|
| `num_steps_per_env` | 每个环境每次迭代采集多少步 |
| `max_iterations` | 最大训练迭代次数 |
| `learning_rate` | 学习率 |
| `entropy_coef` | 熵正则化系数 |
| `gamma` | 奖励折扣因子 |
| `lam` | GAE 参数 |
| `desired_kl` | 自适应学习率的目标 KL |
| `num_learning_epochs` | 每轮数据重复学习多少次 |
| `num_mini_batches` | Mini-batch 数量 |

### 11.2 RSL-RL 是如何记录 TensorBoard 的？

你可以在安装的 RSL-RL 源码中找到：

```text
rsl_rl/runners/on_policy_runner.py
```

以及对应版本的 Logger 实现。

核心逻辑可以简化为：

```python
# 记录奖励
writer.add_scalar(
    "Train/mean_reward",
    mean_reward,
    iteration,
)

# 记录 Episode 长度
writer.add_scalar(
    "Train/mean_episode_length",
    mean_episode_length,
    iteration,
)

# 记录 Value Loss
writer.add_scalar(
    "Loss/value_function",
    value_loss,
    iteration,
)

# 记录策略标准差
writer.add_scalar(
    "Policy/mean_std",
    mean_std,
    iteration,
)

# 记录 FPS
writer.add_scalar(
    "Perf/total_fps",
    fps,
    iteration,
)
```

> 这是帮助理解的简化示意代码，不是要求你替换原来的 Logger。

实际 RSL-RL 不同版本的标签名称可能略有区别。例如有些版本使用 `Policy/mean_noise_std`，有些使用 `Policy/mean_std`。

### 11.3 如何查看你当前版本的真实指标？

直接运行：

```bash
find logs/rsl_rl -name "events.out.tfevents*"
```

然后启动 TensorBoard：

```bash
uv run python -m tensorboard.main --logdir logs/rsl_rl
```

这是当前 Isaac Lab 文档给出的常用方式。旧版工程也可以通过 `isaaclab.sh` 启动。

如果你想确认具体有哪些 Tag，可以使用：

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "logs/rsl_rl/你的实验目录"
ea = EventAccumulator(log_dir)
ea.Reload()

for tag in ea.Tags()["scalars"]:
    print(tag)
```

这样就可以直接打印出你当前训练记录的全部 Scalar 指标。

## 十二、真正训练四足机器人时，应该如何分析？

下面给你一个实际分析案例。

假设你正在训练 ANYmal-C 四足机器人，希望它能够稳定地跟踪目标速度。

我建议按以下顺序观察：

### 四足机器人训练诊断流程

1. **第一步：看 Mean Reward**
   - Reward 是否整体上升？
   - 如果没有上升，优先检查奖励函数、观测、动作和训练配置。

2. **第二步：看 Episode Length**
   - 机器人是否能够持续运动？是否频繁提前结束？

3. **第三步：看 Reward 分项**
   - 速度跟踪、姿态、脚部滑动、力矩等奖励是否符合预期？

4. **第四步：看 Value Loss 和 Surrogate Loss**
   - PPO 更新是否出现异常震荡或数值发散？

5. **第五步：看 Action Std 和 Entropy**
   - 策略是否过早失去探索能力？

6. **第六步：加载模型实际测试**
   - 机器人是否真的能够稳定行走，而不只是获得较高 Reward？

### 12.1 训练成功后，如何验证？

不要只看 TensorBoard。

还应该加载训练好的模型进行仿真测试。

例如，较新的 Isaac Lab CLI 可以使用：

```bash
uv run isaaclab play \
    --rl_library rsl_rl \
    --task Isaac-Velocity-Flat-Anymal-C-v0 \
    --checkpoint latest
```

具体任务名称和命令参数需要与你当前工程版本匹配。

观察：

- 机器人能否稳定站立？
- 能否跟踪目标线速度？
- 能否跟踪目标角速度？
- 是否出现脚部滑动？
- 是否出现身体明显晃动？
- 改变速度指令后能否及时响应？

官方文档也建议结合训练指标与实际策略回放评估模型。

## 十三、最后给你一份 TensorBoard 指标速查表

建议你先把下面这张表记住。

| TensorBoard Tag | 主要含义 | 分析重点 |
|---|---|---|
| `Train/mean_reward` | 平均奖励 | 是否改善 |
| `Train/mean_episode_length` | 平均回合长度 | 是否频繁失败 |
| `Loss/value_function` | Critic 损失 | 是否稳定 |
| `Loss/surrogate` | Actor 策略损失 | 是否异常震荡 |
| `Loss/entropy` | 熵相关损失 | 探索情况 |
| `Loss/learning_rate` | 学习率 | 是否异常变化 |
| `Policy/mean_std` | 动作标准差 | 探索幅度 |
| `Perf/total_fps` | 训练速度 | 训练效率 |
| `Perf/collection_time` | 数据收集耗时 | 仿真性能 |
| `Perf/learning_time` | 网络更新耗时 | 训练性能 |
| `Episode_Reward/...` | 奖励分项统计 | 哪些奖励在改善 |

其中部分名称会因 RSL-RL 版本和任务代码不同而变化。

## 十四、我建议你下一步实际做一个训练曲线分析

如果你正在做 Isaac Lab 四足机器人强化学习，我建议下一步直接分析你自己的训练日志，而不是只看理论。

你可以把 TensorBoard 的 Scalars 页面截图发给我，我可以逐条帮你分析。

### 告诉我你的实际训练环境

1. 使用的机器人
   - ANYmal-C
   - ANYmal-D
   - Unitree Go1
   - Unitree Go2
   - Unitree G1
   - 自定义机器人

2. 训练任务
   - 平地速度跟踪
   - 复杂地形行走
   - 站立平衡
   - 运动模仿
   - 其他自定义任务

3. Isaac Lab 版本
   - Isaac Lab 2.x
   - Isaac Lab 3.x
   - 更早版本
   - 不确定

### 提交训练信息

还可以直接上传你的 TensorBoard 曲线截图或 `events.out.tfevents` 文件。我可以帮你判断当前训练处于探索阶段、学习阶段还是收敛阶段，并分析是否存在奖励设计或 PPO 参数问题。