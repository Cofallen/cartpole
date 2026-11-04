#!/usr/bin/env bash
# 在 PRIME on-demand 的笔记本上强制 Isaac Sim 用 NVIDIA 独显。
#
# 用法：
#   ./scripts/isaac_gpu.sh python scripts/rsl_rl/play.py --task=... --headless
#   ./scripts/isaac_gpu.sh python scripts/rsl_rl/train.py --task=... --num_envs=64
#
# 背景：你机器是 AMD 核显驱动显示（Provider 0）、NVIDIA 4060 做 offload（Provider 1）。
# 不设这些变量时，Vulkan/GL 可能落到 AMD 核显上，Omniverse 会卡死或丢设备。
set -euo pipefail

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=0
# 关掉 Omni 的崩溃收集器（它偶尔自己卡住）
export OMNIVERSE_DISABLE_CRASHREPORTER="${OMNIVERSE_DISABLE_CRASHREPORTER:-1}"

echo "[isaac_gpu] VK_ICD=$VK_ICD_FILENAMES  PRIME_OFFLOAD=1"
exec "$@"
