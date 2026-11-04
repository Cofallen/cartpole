#!/usr/bin/env bash
# 每 2 秒把 GPU/内存写进 CSV 并 sync。硬卡死重启后，看文件最后几行就知道卡在什么状态。
# 用法：另开一个终端跑 ./scripts/gpu_watch.sh   （Ctrl+C 结束）
set -uo pipefail
mkdir -p logs
out="logs/gpu_watch_$(date +%Y%m%d_%H%M%S).csv"
echo "time,gpu_util_pct,vram_used_MiB,vram_total_MiB,temp_C,power_W,ram_used_MB,ram_free_MB,swap_used_MB" > "$out"
echo "[gpu_watch] 写入 $out"
while true; do
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  mem=$(free -m | awk '/^Mem:/{printf "%s,%s", $3, $4} /^Swap:/{printf ",%s", $3}')
  printf '%s,%s,%s\n' "$(date +%H:%M:%S)" "$gpu" "$mem" >> "$out"
  sync                       # 关键：硬卡死时不 sync 就丢数据
  sleep 2
done
