#!/usr/bin/env bash
# ============================================================
# 按 _vN 一一对应启动训练（test 固定 seed 42 / train-val 随训练 seed）
# ------------------------------------------------------------
# 划分口径（论文 Table 8 与 Appendix A.1）:
#   test       由 TEST_SEED = 42 独立划分, 跨 run、跨模型恒定 —— 所有模型在同一批
#              测试图像上评估, 因此 run i 上两个模型的比较构成配对样本。
#   train/val  由训练 seed (SEED) 划分, 故各 run 的训练/验证成员不同(数量不变),
#              run 间差异同时包含训练随机性与数据划分差异。
#   注意: train/val 只在 run 之间变; 同一 run index 下所有模型共用同一 SEED,
#         因此 run i 的 IndTokenSwin 与基线在 train / val / test 三段上完全相同,
#         唯一差异是架构。
#
# 日志核验: 每个 run 会打印一行
#   [Split] test_seed=42 (跨 run/模型恒定)  trainval_seed=<SEED> ... MD5(train/val/test)=a/b/c
# 其中 test 的 MD5 必须跨 run 完全一致, train/val 的 MD5 随 run 变化。
#
# 映射 (run index -> _vN -> 训练 seed):
#   run 1  -> _v0 : SEED=112458
#   run 2  -> _v1 : SEED=142956
#   run 3  -> _v2 : SEED=212855
#   run 4  -> _v3 : SEED=653153
#   run 5  -> _v4 : SEED=940839
#
# n 值: 12 类任务 (B1/B2) 论文 n=3, 只跑 _v0.._v2; SIIM-ISIC / Polyp (C2/G) n=5。
# ============================================================

set -e

# ---------- 路径(按需调整; 此处对应 exp_script 工作副本) ----------
EXP_SCRIPT_DIR='C:/sand_box/exp_script'
PROJECT_DIR="$EXP_SCRIPT_DIR/scripts"
SCRIPT_DIR="$PROJECT_DIR/train"
CORE_DIR="$PROJECT_DIR/core"
PYTHON='C:/ProgramData/anaconda3/python.exe'

# Path A 结果根目录。重跑统一写入 exp_script/rerun/main，由 rerun_all.py 采集归档，
# 不再散落在 scripts/core/checkpoint (口径重构前的旧结果位置)。
OUT_ROOT="${OUT_ROOT:-$EXP_SCRIPT_DIR/rerun/main}"
export OUTPUT_ROOT="$OUT_ROOT"
mkdir -p "$OUT_ROOT"

# 让 datasets / train_utils / SWINv2 等核心模块可被导入
export PYTHONPATH="$CORE_DIR:$SCRIPT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

# ---------- 训练脚本清单(自动发现 scripts/train/train_exp*.py) ----------
# 分两组: 12 类任务 (B1/B2, 论文 n=3) 与 SIIM-ISIC / Polyp (C2/G, 论文 n=5)。
SCRIPTS_N3=()
SCRIPTS_N5=()
for f in "$SCRIPT_DIR"/train_exp*.py; do
  b="$(basename "$f")"
  case "$b" in
    train_expB1_*|train_expB2_*) SCRIPTS_N3+=("$b") ;;
    *)                           SCRIPTS_N5+=("$b") ;;
  esac
done

# ---------- 训练 seed 表(与 _vN 一一对应) ----------
RUN_SEED=(112458 142956 212855 653153 940839)

run_one() {
  local script="$1"
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $script"
  echo "============================================================"
  start_ts=$(date +%s)
  if ! KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 \
    SEED="$SEED" RUN_SUFFIX="$RUN_SUFFIX" \
    "$PYTHON" "$SCRIPT_DIR/$script"; then
    echo "!!! [FAILED] $script 退出码非0 (继续后续任务，不中断整批)"
  fi
  end_ts=$(date +%s)
  duration=$(( (end_ts - start_ts) / 60 ))
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE:  $script (elapsed ${duration} min)"
  echo "============================================================"
}

# ---------- 按 run 注入训练 seed 并启动 ----------
# TEST_SEED=42 硬编码在各训练脚本内, 不由环境变量注入, 以免被误改。
for i in 0 1 2 3 4; do
  export SEED=${RUN_SEED[$i]}
  export RUN_SUFFIX=$i                # -> _v0 .. _v4
  echo "########## RUN $((i+1))  ->  _v$i   (SEED=$SEED  TEST_SEED=42 恒定) ##########"
  for script in "${SCRIPTS_N5[@]}"; do
    run_one "$script"
  done
  # 12 类任务只取前 3 个 run (论文 n=3); _v3/_v4 不产生对应结果
  if [ "$i" -lt 3 ]; then
    for script in "${SCRIPTS_N3[@]}"; do
      run_one "$script"
    done
  fi
done

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL RUNS COMPLETED"
echo "============================================================"
