#!/usr/bin/env bash
# 实验3 全流程（在 lab3/ 目录下运行）：
#   bash run_all.sh                # 默认使用 GPU 0 1 2 4
#   GPUS="0 1 2 3" bash run_all.sh
# 日志：outputs/logs/*.log
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
read -r -a G <<< "${GPUS:-0 1 2 4}"
mkdir -p outputs/logs

run() {
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY train.py --name "$name" "$@" > "outputs/logs/$name.log" 2>&1
  tail -n 2 "outputs/logs/$name.log"
}

# 1.1 基线 + 1.2 类别不均衡处理（各 200 epoch，相同超参与种子，并行跑在不同 GPU 上）
run "${G[0]}" baseline     --imbalance none &
run "${G[1]}" reweight     --imbalance reweight &
run "${G[2]}" resample     --imbalance resample &
run "${G[3]}" logit_adjust --imbalance logit_adjust &
wait

# 按验证集（不看测试集）最后一个 epoch 的 balanced accuracy 选参考模型
read -r REF IMB < <($PY - <<'EOF'
import json
best = max(["baseline", "reweight", "resample", "logit_adjust"],
           key=lambda n: json.load(open(f"outputs/{n}/history.json"))[-1]["val_balanced_acc"])
print(best, json.load(open(f"outputs/{best}/metrics.json"))["imbalance"])
EOF
)
echo "reference model: $REF (imbalance=$IMB)"

# 1.3 轻量化：从参考模型初始化，蒸馏微调 50 epoch
run "${G[0]}" dynamicvit --light dynamicvit --init "outputs/$REF/model.pth" --imbalance "$IMB" &
run "${G[1]}" prune      --light prune      --init "outputs/$REF/model.pth" --imbalance "$IMB" &
wait

# 1.4 复杂度 / 速度测评（单卡顺序执行，含免训练的 ToMe）+ 汇总
CUDA_VISIBLE_DEVICES=${G[0]} $PY benchmark.py --ref "$REF" 2>&1 | tee outputs/logs/benchmark.log
$PY analyze.py --ref "$REF"
