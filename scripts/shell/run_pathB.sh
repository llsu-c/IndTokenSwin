#!/usr/bin/env bash
# ============================================================
# Path B (续训) 启动脚本 —— ImageNet -> Polyp(G) -> HyperKvasir / Kvasir-Capsule
# ------------------------------------------------------------
# 论文 §3.4 / Table 4 的口径: "initialize from each model's best Polyp checkpoint"。
# 本脚本把该口径固化为可复现的两条显式约定:
#   1) 初始化来源: INIT_FROM = 同一 run index 的 G 组该模型最佳权重
#        run i 的 B1/B2  <-  run i 的 G  (best_model_phase2.pth)
#      同名模型与 run index 一一对应, 保证 G/B 是可配对的样本。
#   2) 输出隔离: OUTPUT_ROOT 指向独立根目录, Path B 结果不写进 Path A 的 checkpoint/。
# 划分口径 (论文 Table 8 / Appendix A.1): test 由 TEST_SEED = 42 固定 (各脚本内硬编码),
#       train/val 随训练 seed (SEED) 重抽; Path A 与 Path B 用同一 run index 即同一 SEED,
#       故两路共用同一 train/val/test, 差异只来自初始化, 不掺入划分差异。
#
# 用法:
#   bash run_pathB.sh --check      只打印 INIT_FROM 解析结果, 不启动训练
#   bash run_pathB.sh              按 run index 0..4 依次启动 Path B
# ============================================================

set -e

# ---------- 路径 ----------
EXP_SCRIPT_DIR='C:/sand_box/exp_script'
PROJECT_DIR="$EXP_SCRIPT_DIR/scripts"
SCRIPT_DIR="$PROJECT_DIR/train"
CORE_DIR="$PROJECT_DIR/core"
PYTHON='C:/ProgramData/anaconda3/python.exe'

# G 组最佳权重所在根目录。
# 重跑时优先取 rerun/G_ckpts —— 由 rerun_all.py 从「本次重跑」的 G 组权重采集而来,
# 保证 Path B 的初始化来自同一套新划分口径 (test seed 42) 训练出的 Polyp 模型,
# 而不是口径重构前的旧归档。若该目录不存在 (例如仅做发布包的独立复现),
# 回退到 exp_script/selected_ckpts/main_G。
_G_NEW="$EXP_SCRIPT_DIR/rerun/G_ckpts"
if [ -n "${G_ROOT:-}" ]; then
  :
elif [ -d "$_G_NEW" ] && ls "$_G_NEW"/*/best_model_phase2.pth >/dev/null 2>&1; then
  G_ROOT="$_G_NEW"
else
  G_ROOT="$EXP_SCRIPT_DIR/selected_ckpts/main_G"
fi
OUT_ROOT="${OUT_ROOT:-$EXP_SCRIPT_DIR/rerun/G_continue}"   # Path B 结果根目录 (与 Path A 隔离)

export PYTHONPATH="$CORE_DIR:$SCRIPT_DIR:$PYTHONPATH"
export OUTPUT_ROOT="$OUT_ROOT"

# ---------- B1 / B2 脚本与其在 G 目录中的模型名 glob ----------
# G 目录里同一模型存在 ConvNextTiny / ConvNeXtTiny, MaxVitTiny / MaxViTTiny 两种写法,
# 用字符类同时匹配, 避免因拼写差异漏匹配。
B_SCRIPTS=(
  "train_expB1_beitv2.py|*Bei[Tt]v2Base*"
  "train_expB1_convnext.py|*ConvN[eE]xtTiny*"
  "train_expB1_maxvit.py|*MaxVi[Tt]Tiny*"
  "train_expB1_swinv2.py|*SWINv2_2026*"
  "train_expB1_swinv2_window_token_final_v2.py|*SWINv2_WindowTokenV2_Final_v2*"
  "train_expB2_beitv2.py|*Bei[Tt]v2Base*"
  "train_expB2_convnext.py|*ConvN[eE]xtTiny*"
  "train_expB2_maxvit.py|*MaxVi[Tt]Tiny*"
  "train_expB2_swinv2.py|*SWINv2_2026*"
  "train_expB2_swinv2_window_token_final_v2.py|*SWINv2_WindowTokenV2_Final_v2*"
)

RUN_SEED=(112458 142956 212855 653153 940839)

# run index 0..4 分别对应 G 组目录里的哪个 __vN 后缀。
# 默认 G_ROOT (selected_ckpts/main_G) 已按归档编号 __v0..__v4 组织, 故一一对应;
# 若改用原始 checkpoint 根目录 (其批次编号可能是 __v7..__v13), 用 G_RUN_VERS 覆盖:
#   G_RUN_VERS="7 8 9 10 11" bash run_pathB.sh
G_RUN_VERS=(${G_RUN_VERS:-0 1 2 3 4})

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# 解析某脚本在 run i 下的初始化权重。
# 两种布局都要支持:
#   a) <G_ROOT>/<glob>__vN/best_model_phase2.pth                (selected_ckpts 布局)
#   b) <G_ROOT>/<glob>__vN/checkpoints/best_model_phase2.pth    (原始 checkpoint 布局)
resolve_init() {
  local i="$2" glob="$3"
  local cand
  cand=$(ls -d "$G_ROOT"/${glob}__v${G_RUN_VERS[$i]}/best_model_phase2.pth 2>/dev/null | head -1)
  if [ -z "$cand" ]; then
    cand=$(ls -d "$G_ROOT"/${glob}__v${G_RUN_VERS[$i]}/checkpoints/best_model_phase2.pth 2>/dev/null | head -1)
  fi
  if [ -z "$cand" ]; then
    # 回退: 该模型在 G_ROOT 下若只有一个 run, 就用它 (并显式告警)
    local all
    all=("$G_ROOT"/${glob}__v*/best_model_phase2.pth
         "$G_ROOT"/${glob}__v*/checkpoints/best_model_phase2.pth)
    if [ ${#all[@]} -eq 1 ] && [ -f "${all[0]}" ]; then
      cand="${all[0]}"
      echo "    [WARN] run $i 无对应 G 权重, 回退到唯一可用: $(basename "$(dirname "$(dirname "$cand")")")" >&2
    fi
  fi
  printf '%s' "$cand"
}

run_one() {
  local script="$1" init="$2"
  echo "------------------------------------------------------------"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $script"
  echo "    INIT_FROM  = $init"
  echo "    OUTPUT_ROOT= $OUTPUT_ROOT"
  start_ts=$(date +%s)
  if ! KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 \
    SEED="$SEED" RUN_SUFFIX="$RUN_SUFFIX" \
    INIT_FROM="$init" OUTPUT_ROOT="$OUTPUT_ROOT" \
    "$PYTHON" "$SCRIPT_DIR/$script"; then
    echo "!!! [FAILED] $script 退出码非0 (继续后续任务，不中断整批)"
  fi
  end_ts=$(date +%s)
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE:  $script (elapsed $(( (end_ts - start_ts) / 60 )) min)"
}

cd "$PROJECT_DIR"

for i in 0 1 2; do   # 续训目标是 12 类任务, 论文 n=3
  export SEED=${RUN_SEED[$i]}
  export RUN_SUFFIX=$i
  echo "########## PATH B  run $((i+1))  ->  _v$i  (SEED=$SEED) ##########"
  for entry in "${B_SCRIPTS[@]}"; do
    script="${entry%%|*}"
    glob="${entry##*|}"
    init=$(resolve_init "$script" "$i" "$glob")
    if [ -z "$init" ]; then
      avail=$(ls -d "$G_ROOT"/${glob}__v*/checkpoints/best_model_phase2.pth 2>/dev/null \
              | sed 's|.*/\([^/]*\)/checkpoints/.*|\1|' | tr '\n' ' ')
      echo "!!! [SKIP] $script : 找不到 run $i (__v${G_RUN_VERS[$i]}) 的 ${glob} 最佳权重"
      echo "           该模型现有 G 目录: ${avail:-无}  -> 用 G_RUN_VERS 指定映射"
      continue
    fi
    if [ "$CHECK_ONLY" = "1" ]; then
      printf '  %-50s <- %s\n' "$script" "${init#$G_ROOT/}"
    else
      run_one "$script" "$init"
    fi
  done
done

echo "============================================================"
if [ "$CHECK_ONLY" = "1" ]; then
  echo "CHECK DONE (未启动训练)"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] PATH B ALL RUNS COMPLETED"
fi
echo "============================================================"
