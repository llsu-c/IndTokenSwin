"""
train_expB1_swinv2_window_token_final_v2.py
实验B1: SWINv2 WindowToken V2 - HyperKvasir 12类精选版
模型: SwinTransformerV2WithGlobalIndTokenV2 (V2 窗口独立 Token)
预训练源: pretrained_ckpts/imagenet_val_example/epoch_022_top1_82.89.pth (top-1 82.80%)
- 使用 HyperKvasirDataset.build_samples()
- 12类, num_classes=12
- 使用 stratified_split 划分
- Batch=64
- Phase1: 50 epochs, LR_HEAD=2e-5
- Phase2: 300 epochs, LR_FULL=1e-5
- 种子: 由环境变量 SEED/TEST_SEED 注入 (见 scripts/shell/run_by_seed.sh)
"""

import logging
import os
import sys
import csv
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import HyperKvasirDataset, get_train_transforms, get_val_transforms, stratified_split, split_with_fixed_test
from train_utils import (
    setup_logging, setup_output_dirs, seed_everything, get_device,
    train_two_phase, build_param_groups,
)
from SWINv2_window_token_final_v2 import SwinTransformerV2WithGlobalIndTokenV2

# 类别名称映射 (用于输出)
CLASS_NAMES = [
    "barretts", "esophagitis", "polyp", "ulcerative-colitis",
    "normal-z-line", "normal-pylorus", "normal-cecum", "retroflex-stomach",
    "retroflex-rectum", "bbps", "dyed", "stool"
]


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f"Confusion Matrix - {TASK_NAME} Test Set")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def save_predictions_csv(image_paths, predictions, labels, class_names, save_path):
    """保存预测结果到CSV文件"""
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'image_path', 'predicted_label', 'predicted_class', 'true_label', 'true_class', 'correct'])
        for img_path, pred, label in zip(image_paths, predictions, labels):
            img_id = Path(img_path).stem
            writer.writerow([
                img_id,
                img_path,
                int(pred),
                class_names[pred] if pred < len(class_names) else f"unknown_{pred}",
                int(label),
                class_names[label] if label < len(class_names) else f"unknown_{label}",
                "YES" if pred == label else "NO"
            ])
    print(f"Predictions saved to: {save_path}")


# ===========================
# 配置
# ===========================
TASK_NAME = "ExpB1_SWINv2_WindowTokenV2_Final_v2"
IMG_SIZE = 256
BATCH_SIZE = 64
NUM_EPOCHS_PHASE1 = 50
NUM_EPOCHS_PHASE2 = 300
TOTAL_EPOCHS = NUM_EPOCHS_PHASE1 + NUM_EPOCHS_PHASE2
LR_HEAD = 2e-5
LR_FULL = 1e-5
WEIGHT_DECAY = 0.02
NUM_CLASSES = 12
NUM_WORKERS = 4
GRAD_CLIP_NORM = 1.0
USE_AMP = True
SEED = int(os.environ.get("SEED", 2000))
TEST_SEED = 42  # test 集与样本池专用 seed: 跨 run、跨模型恒定不变, 由它独立划分出的 test 集固定不变, 所有模型在同一批测试图像上评估; train/val 另按训练 seed 划分 (见 scripts/shell/run_by_seed.sh)
TRAINVAL_FOLLOWS_TRAIN_SEED = True  # train/val 由 SEED 决定, 故各 run 的训练/验证成员不同
INIT_FROM = os.environ.get("INIT_FROM", "")  # Path B 续训: 指向各模型最佳 Polyp checkpoint 的 best_model_phase2.pth; 设置后取代 ImageNet 预训练; head.* 与形状不符的键自动跳过, 即 C=2 -> C=12 重初始化

PROJECT_ROOT = Path(__file__).parent
# Path A 默认预训练源 (论文 §4.1: "the public ImageNet-1K Swin V2 checkpoint"):
# 官方 Swin V2 ImageNet-1K 权重 (top-1 82.89)。Ind Token 相关参数在该权重中不存在,
# 加载时按"键名+形状匹配"自动跳过并随机初始化 —— 与 G 组 Path A 同一来源, 保证跨路径可比。
# Path B (续训) 由 INIT_FROM 指向本模型 G 组最佳权重, 覆盖本默认值。
PRETRAINED_WEIGHTS_DIR = PROJECT_ROOT / ".." / ".." / "pretrained_ckpts" / "imagenet_val_example" / "epoch_022_top1_82.89.pth"

MODEL_CONFIG = dict(
    img_size=256,
    patch_size=4,
    in_chans=3,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.1,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
)

def build_model(num_classes: int = 12):
    """构建 SWINv2 V2 模型 (SwinTransformerV2WithGlobalIndTokenV2)
    从 val_example .pth 加载预训练 (已含 V2 特有参数), 跳过 head.* (因 num_classes 不同)
    """
    model = SwinTransformerV2WithGlobalIndTokenV2(**MODEL_CONFIG, num_classes=num_classes)

    init_path = Path(INIT_FROM) if INIT_FROM else PRETRAINED_WEIGHTS_DIR
    logging.info(f"[Model] 初始化权重: {init_path}" + ("  [Path B 续训]" if INIT_FROM else "  [Path A 默认]"))
    if os.path.exists(init_path):
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        model_state = model.state_dict()
        filtered_state = {}
        for k, v in state_dict.items():
            if k.startswith("head."):
                continue
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        logging.info(f"[Model] 预训练 V2 backbone 加载: loaded={len(filtered_state)}, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            miss_head = [k for k in missing if k.startswith("head.")]
            miss_other = [k for k in missing if not k.startswith("head.")]
            logging.info(f"[Model] missing.head (期望): {len(miss_head)} -> {miss_head[:5]}")
            if miss_other:
                logging.info(f"[Model] missing.other (警告): {len(miss_other)} -> {miss_other[:5]}")
    else:
        logging.info(f"[Model] 警告: 初始化权重不存在: {init_path}")

    return model


def setup_trainable_params(model, lr_head):
    """设置可训练参数 - Phase 1 Head & Norm Warm-up (§3.4): 分类头 + LayerNorm/归一化层 affine 参数"""
    for param in model.parameters():
        param.requires_grad = False

    head_params = []
    for param in model.head.parameters():
        param.requires_grad = True
        head_params.append(param)

    # 论文 §3.4 "Head & Norm Warm-up": Phase 1 除分类头外还训练 LayerNorm/归一化层 affine 参数。
    # 只命中尚未解冻、名字含 norm 的低维参数: 自动排除已解冻的 head 参数与卷积核 (ndim>2)。
    for name, param in model.named_parameters():
        if "norm" in name.lower() and param.ndim <= 2 and not param.requires_grad:
            param.requires_grad = True
            head_params.append(param)

    param_groups = [{"params": head_params, "lr": lr_head, "weight_decay": WEIGHT_DECAY}]
    optimizer = optim.AdamW(param_groups)

    return optimizer, sum(p.numel() for p in head_params)


def main():
    # 设置输出目录
    task_dir, checkpoint_dir = setup_output_dirs(TASK_NAME)
    logger = setup_logging(TASK_NAME, task_dir)
    logger.info(f"[Run] SEED={SEED}  TEST_SEED={TEST_SEED}  "
                f"RUN_SUFFIX={os.environ.get('RUN_SUFFIX', '-')}  "
                f"INIT_FROM={os.environ.get('INIT_FROM', '') or '(默认预训练源)'}")

    logger.info("=" * 60)
    logger.info("实验B1: SWINv2 WindowTokenV2 - HyperKvasir 12类精选版")
    logger.info(f"模型: {SwinTransformerV2WithGlobalIndTokenV2.__name__}")
    logger.info(f"预训练: {PRETRAINED_WEIGHTS_DIR}")
    logger.info("=" * 60)
    logger.info(f"数据集: HyperKvasirDataset.build_samples()")
    logger.info(f"类别数: {NUM_CLASSES}")
    logger.info(f"总Epochs: {TOTAL_EPOCHS} (P1={NUM_EPOCHS_PHASE1}, P2={NUM_EPOCHS_PHASE2})")
    logger.info(f"Batch: {BATCH_SIZE}, LR_Head: {LR_HEAD}, LR_Full: {LR_FULL}")
    logger.info(f"输出目录: {task_dir}")
    logger.info("=" * 60)

    seed_everything(SEED)
    device = get_device()

    # 构建样本列表（用 TEST_SEED 对齐基础版 train_expB1_swinv2.py）
    all_samples = HyperKvasirDataset.build_samples(seed=TEST_SEED)

    logger.info(f"[Dataset] 总样本数: {len(all_samples)}")

    # 划分数据集（用 TEST_SEED 对齐基础版）
    # test 集由 TEST_SEED 固定(跨 run / 跨模型恒定); train/val 由训练 seed 重抽(只换成员, 数量不变)
    train_samples, val_samples, test_samples = split_with_fixed_test(
        all_samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        test_seed=TEST_SEED, trainval_seed=SEED, stratified=True,
    )

    logger.info(f"[Dataset] 训练集: {len(train_samples)}")
    logger.info(f"[Dataset] 验证集: {len(val_samples)}")
    logger.info(f"[Dataset] 测试集: {len(test_samples)}")

    # 创建数据集
    train_ds = HyperKvasirDataset(train_samples, img_size=IMG_SIZE, transform=get_train_transforms(IMG_SIZE))
    val_ds = HyperKvasirDataset(val_samples, img_size=IMG_SIZE, transform=get_val_transforms(IMG_SIZE))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    logger.info(f"[Data] Train={len(train_ds)}, Val={len(val_ds)}, Batches={len(train_loader)}")

    # 模型
    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    # torch.compile: 短期方案,不动模型代码,trace 优化 V2 模型 line 496 的 for w in range(num_windows) Python 循环
    # 需先做 smoke test 触发 lazy 编译,失败时回退到未编译模型
    try:
        compiled = torch.compile(model, mode="default", fullgraph=False)
        with torch.no_grad():
            _ = compiled(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))
        model = compiled
        logger.info("[Model] torch.compile 已应用并通过 smoke test (mode=default)")
    except Exception as e:
        logger.warning(f"[Model] torch.compile 失败({type(e).__name__}),继续使用未编译模型: {str(e)[:120]}")

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase1 优化器 (head + V2 特有参数)
    optimizer_p1, warmup_count = setup_trainable_params(model, LR_HEAD)
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=NUM_EPOCHS_PHASE1, eta_min=1e-6)

    # Phase2 优化器 (全参数，但 bias/位置偏置/缩放系数豁免 weight decay)
    param_groups = build_param_groups(model, WEIGHT_DECAY)
    optimizer_p2 = optim.AdamW(param_groups, lr=LR_FULL)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=NUM_EPOCHS_PHASE2, eta_min=1e-6)

    scaler = GradScaler() if USE_AMP and device.type == "cuda" else None

    # 配置
    config = {
        "phase1_epochs": NUM_EPOCHS_PHASE1,
        "phase2_epochs": NUM_EPOCHS_PHASE2,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "early_stop_patience": 5,
        "phase2_load_best_phase1": True,
    }

    logger.info(f"[Train] Phase1 可训练参数: {warmup_count} (head + V2 特有参数)")

    # 训练
    history, best_metric, best_epoch = train_two_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer_phase1=optimizer_p1,
        optimizer_phase2=optimizer_p2,
        scheduler_phase1=scheduler_p1,
        scheduler_phase2=scheduler_p2,
        scaler=scaler,
        device=device,
        config=config,
        logger=logger,
        task_dir=task_dir,
        is_binary=False,
        num_classes=NUM_CLASSES,
        is_v6=False,
        custom_val_metric="macro_f1",
        seed=SEED,
    )

    logger.info(f"\n训练完成! Best Macro F1={best_metric:.4f} @ Epoch {best_epoch}")

    # ========== 测试集评估 ==========
    logger.info("\n" + "=" * 60)
    logger.info("测试集评估")
    logger.info("=" * 60)

    # 加载最佳模型
    best_model_path = task_dir / "checkpoints" / "best_model_phase2.pth"
    if not best_model_path.exists():
        best_model_path = task_dir / "checkpoints" / "best_model_phase1.pth"

    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"已加载最佳模型: {best_model_path}")
        logger.info(f"Best Val (macro_f1): {checkpoint.get('best_metric', 'N/A')}")
    else:
        logger.warning("未找到最佳模型，跳过测试集评估")

    test_ds = HyperKvasirDataset(test_samples, img_size=IMG_SIZE, transform=get_val_transforms(IMG_SIZE))
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    logger.info(f"测试集大小: {len(test_ds)}")

    from train_utils import evaluate_multiclass
    test_stats, all_labels, all_preds, _ = evaluate_multiclass(model, test_loader, criterion, device, NUM_CLASSES)
    logger.info(f"测试集结果:")
    logger.info(f"  Accuracy: {test_stats['accuracy']:.2f}%")
    logger.info(f"  Macro F1: {test_stats['macro_f1']:.4f}")
    logger.info(f"  Weighted F1: {test_stats['weighted_f1']:.4f}")
    logger.info(f"  Macro AUC: {test_stats.get('mAUC', 0):.4f}")
    logger.info(f"  Combined Score (test, secondary): {test_stats.get('combined_score', 0):.4f}")
    logger.info(f"  Macro Precision: {test_stats['macro_precision']:.4f}")
    logger.info(f"  Macro Recall: {test_stats['macro_recall']:.4f}")

    # ========== 保存 Confusion Matrix ==========
    cm = test_stats['confusion_matrix']

    cm_img_path = task_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_img_path)

    cm_txt_path = task_dir / "confusion_matrix.txt"
    with open(cm_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Confusion Matrix - {TASK_NAME} Test Set\n")
        f.write("=" * 60 + "\n\n")
        f.write("Class Names:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {name}\n")
        f.write("\nConfusion Matrix (rows=true, cols=pred):\n\n")
        header = "      " + " ".join(f"{i:>6}" for i in range(NUM_CLASSES))
        f.write(header + "\n")
        for i, row in enumerate(cm):
            f.write(f"{i:>4}:  " + " ".join(f"{v:>6}" for v in row) + "\n")
        f.write(f"\nTotal Samples: {len(test_ds)}\n")
        f.write(f"Accuracy: {test_stats['accuracy']:.2f}%\n")
        f.write(f"Macro F1: {test_stats['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {test_stats['weighted_f1']:.4f}\n")
        f.write(f"Macro AUC: {test_stats.get('mAUC', 0):.4f}\n")
        f.write(f"Combined Score: {test_stats.get('combined_score', 0):.4f}\n")
        f.write(f"Macro Precision: {test_stats['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {test_stats['macro_recall']:.4f}\n")
    logger.info(f"Confusion matrix (text) saved to: {cm_txt_path}")

    # ========== 保存预测结果 CSV ==========
    test_image_paths = [sample[0] for sample in test_samples]
    predictions_csv_path = task_dir / "predictions.csv"
    save_predictions_csv(test_image_paths, all_preds, all_labels, CLASS_NAMES, predictions_csv_path)

    # 保存测试结果
    test_results_path = task_dir / "test_results.txt"
    with open(test_results_path, "w", encoding="utf-8") as f:
        f.write("Test Set Evaluation Results (12-class classification)\n")
        f.write(f"Sample Count: {len(test_ds)}\n")
        f.write(f"Best Val (macro_f1): {best_metric:.4f}\n")
        f.write(f"\nTest Set Metrics:\n")
        for k, v in test_stats.items():
            if k != 'confusion_matrix':
                f.write(f"  {k}: {v}\n")
        f.write(f"\nOutput Files:\n")
        f.write(f"  Confusion Matrix (Image): {cm_img_path}\n")
        f.write(f"  Confusion Matrix (Text): {cm_txt_path}\n")
        f.write(f"  Predictions CSV: {predictions_csv_path}\n")
    logger.info(f"测试结果已保存: {test_results_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("训练被用户中断")
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
