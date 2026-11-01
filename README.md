# Template for Isaac Lab Projects

## 学习文档索引（`rl-study/`）

两种打开方式：单篇双击 `.md`（MarkText，`Ctrl+K` 大纲）；整本双击 [`rl-study/index.html`](rl-study/index.html)（浏览器，左侧大纲，公式离线渲染）。

**教程（主线，24 个任务）**

| 文件 | 内容 |
|---|---|
| [01-教程总览](rl-study/01-教程-总览.md) | 任务清单：编号 / 改哪里 / 验收 / 预计 |
| [02-第1阶段-仓库与倒立摆](rl-study/02-教程-第1阶段-仓库与倒立摆.md) | T1.1-T1.10 |
| [03-第2阶段-动作空间与新任务](rl-study/03-教程-第2阶段-动作空间与新任务.md) | T2.1-T2.6 |
| [04-第3阶段-MuJoCo与系统辨识](rl-study/04-教程-第3阶段-MuJoCo与系统辨识.md) | T3.1-T3.5 |
| [05-第4阶段-C仿真与自主设计](rl-study/05-教程-第4阶段-C仿真与自主设计.md) | T4.1-T4.4 |

**参考**

| 文件 | 内容 |
|---|---|
| [00-开始这里](rl-study/00-开始这里.md) | 一页导航 |
| [06-指标速查](rl-study/06-指标速查.md) | 曲线含义与排查 |
| [07-论文清单](rl-study/07-论文清单.md) | 分级书单 + 仓库 |
| [08-实验台账](rl-study/08-实验台账.md) | 实验记录 |
| [09-求职](rl-study/09-求职.md) | 岗位映射与面试 |
| [10-练习册A](rl-study/10-练习册A-强化学习原理.md) / [11-B](rl-study/11-练习册B-现代控制与机器人原理.md) / [12-C](rl-study/12-练习册C-VLA与具身原理.md) | 原理 + 填空 |
| [13-试卷](rl-study/13-试卷-A卷B卷.md) / [14-答案](rl-study/14-试卷答案与评分标准.md) | 闭卷自测 |
| [refs/](README.md) | 本地参考书（Sutton/Boyd/CS285） |

工具（在仓库根目录跑）：

```bash
python scripts/compare_runs.py --md      # 所有 run 压成对比表
python scripts/policy_probe.py --sweep   # 量出导出策略的控制律
python scripts/md_book.py                # 重新生成 index.html（需 conda activate isaac）
```

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/cartpole

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/cartpole/cartpole/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/cartpole"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```