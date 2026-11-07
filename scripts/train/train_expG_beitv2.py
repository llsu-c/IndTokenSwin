"""
train_expG_beitv2.py
实验G: BeiTv2-Base 息肉二分类实验 (与 SWINv2 同配置, 仅模型替换)
- 预训练: timm beitv2_base_patch16_224 (ImageNet-22k → 1k)
- 输入尺寸: 224x224 (与 beitv2 预训练要求一致)
- 与 train_expG_swinv2.py 同数据集/同超参（种子由环境变量 SEED/TEST_SEED 注入，见 scripts/shell/run_by_seed.sh）
"""

import os
import sys
import csv
import random
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from datasets import PolypDataset, get_train_transforms, get_val_transforms, _partition_fingerprint
from train_utils import (
    setup_logging, setup_output_dirs, seed_everything, get_device,
    train_two_phase, evaluate_binary,
)
import timm

CLASS_NAMES = ["non-polyp", "polyp"]


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def save_predictions_csv(image_paths, predictions, labels, class_names, save_path):
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


def evaluate_single_test_set(model, test_samples, task_dir, device, logger, test_set_name):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{test_set_name}")
    logger.info(f"{'=' * 60}")

    test_ds = PolypDataset(test_samples, img_size=IMG_SIZE, transform=get_val_transforms(IMG_SIZE))
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    pos_count = sum(1 for _, l in test_samples if l == 1)
    neg_count = sum(1 for _, l in test_samples if l == 0)
    logger.info(f"测试集大小: {len(test_ds)} (正{pos_count}, 负{neg_count})")

    criterion = nn.CrossEntropyLoss()
    test_stats, all_labels, all_probs = evaluate_binary(model, test_loader, criterion, device)
    all_preds = (all_probs >= 0.5).astype(int)

    logger.info(f"{test_set_name} 结果:")
    logger.info(f"  AUC: {test_stats['auc']:.4f}")
    logger.info(f"  Accuracy: {test_stats['acc']:.2f}%")
    logger.info(f"  Precision: {test_stats['precision']:.4f}")
    logger.info(f"  Recall (Sens): {test_stats['sens']:.4f}")
    logger.info(f"  Specificity: {test_stats['spec']:.4f}")
    logger.info(f"  F1: {test_stats['f1']:.4f}")
    logger.info(f"  MCC: {test_stats['mcc']:.4f}")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)

    cm_img_path = task_dir / "confusion_matrix.png"
    title = f"Confusion Matrix - {TASK_NAME} Test Set"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_img_path, title)

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
        f.write(f"\nTotal Samples: {len(test_ds)} (pos={pos_count}, neg={neg_count})\n")
        f.write(f"AUC: {test_stats['auc']:.4f}\n")
        f.write(f"Accuracy: {test_stats['acc']:.2f}%\n")
        f.write(f"Precision: {test_stats['precision']:.4f}\n")
        f.write(f"Recall (Sens): {test_stats['sens']:.4f}\n")
        f.write(f"Specificity: {test_stats['spec']:.4f}\n")
        f.write(f"F1: {test_stats['f1']:.4f}\n")
        f.write(f"MCC: {test_stats['mcc']:.4f}\n")
    logger.info(f"Confusion matrix (text) saved to: {cm_txt_path}")

    test_image_paths = [sample[0] for sample in test_samples]
    predictions_csv_path = task_dir / "predictions.csv"
    save_predictions_csv(test_image_paths, all_preds, all_labels, CLASS_NAMES, predictions_csv_path)

    return test_stats, all_labels, all_probs, all_preds, pos_count, neg_count


def evaluate_test_set(model, independent_test_samples, task_dir, device, logger):
    best_model_path = task_dir / "checkpoints" / "best_model_phase2.pth"
    if not best_model_path.exists():
        best_model_path = task_dir / "checkpoints" / "best_model_phase1.pth"

    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"已加载最佳模型: {best_model_path}")
        logger.info(f"Best Val AUC: {checkpoint.get('best_metric', 'N/A')}")
    else:
        logger.warning("未找到最佳模型，跳过独立测试集评估")
        return

    stats1, labels1, probs1, preds1, pos1, neg1 = evaluate_single_test_set(
        model, independent_test_samples, task_dir, device, logger,
        "独立测试集 (CVC+ETIS+ColonDB正样本 vs 预留负样本, 1:1平衡)"
    )

    test_results_path = task_dir / "test_results.txt"
    with open(test_results_path, "w", encoding="utf-8") as f:
        f.write("Independent Test Set Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("[Independent Test Set] CVC+ETIS+ColonDB positive samples vs reserved negative samples (1:1 balanced)\n")
        f.write(f"Sample Count: {len(independent_test_samples)} (pos {pos1}, neg {neg1})\n")
        f.write(f"Best Val AUC: {checkpoint.get('best_metric', 'N/A')}\n\n")
        f.write("Test Set Metrics:\n")
        for k, v in stats1.items():
            if k != 'loss':
                f.write(f"  {k}: {v}\n")
        f.write(f"\nOutput Files:\n")
        f.write(f"  Confusion Matrix (Image): {task_dir / 'confusion_matrix.png'}\n")
        f.write(f"  Confusion Matrix (Text): {task_dir / 'confusion_matrix.txt'}\n")
        f.write(f"  Predictions CSV: {task_dir / 'predictions.csv'}\n")
    logger.info(f"测试结果已保存: {test_results_path}")


# ===========================
# 配置
# ===========================
TASK_NAME = "ExpG_Beitv2Base"
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS_PHASE1 = 50
NUM_EPOCHS_PHASE2 = 300
TOTAL_EPOCHS = NUM_EPOCHS_PHASE1 + NUM_EPOCHS_PHASE2
LR_HEAD = 2e-5
LR_FULL = 1e-5
WEIGHT_DECAY = 0.02
NUM_CLASSES = 2
NUM_WORKERS = 4
GRAD_CLIP_NORM = 1.0
USE_AMP = True
SEED = int(os.environ.get("SEED", 2000))
TEST_SEED = 42  # test 集与样本池专用 seed: 跨 run、跨模型恒定不变, 由它独立划分出的 test 集固定不变, 所有模型在同一批测试图像上评估; train/val 另按训练 seed 划分 (见 scripts/shell/run_by_seed.sh)
TRAINVAL_FOLLOWS_TRAIN_SEED = True  # train/val 由 SEED 决定, 故各 run 的训练/验证成员不同

POS_SOURCES = {
    "polypgen_positive": 3762,
    "hyperkvasir_polyp": 1028,
    "gastro_polyp": 1188,
}
NEG_SOURCES = {
    "polypgen_negative": 4275,
    "hyperkvasir_cecum": 1009,
    "hyperkvasir_zline": 932,
    "hyperkvasir_pylorus": 999,
    "gastro_polyp_normal": 1172,
}
RESERVE_TEST_NEG = 1188

PROJECT_ROOT = Path(__file__).parent
PRETRAINED_WEIGHTS_DIR = PROJECT_ROOT / ".." / ".." / "pretrained_weights" / "beitv2_base_patch16_224" / "model.safetensors"
MODEL_NAME = "beitv2_base_patch16_224"
HEAD_PREFIX = "head."  # 预训练 head.weight / head.bias 在此处, 需过滤


def build_model(num_classes: int = 2):
    """构建 timm beitv2_base_patch16_224 模型
    使用 timm.create_model (pretrained=False), 然后用 safetensors 加载本地权重 (严格按 layer 名匹配)
    """
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)

    if os.path.exists(PRETRAINED_WEIGHTS_DIR):
        from safetensors.torch import load_file as load_safetensors
        state_dict = load_safetensors(PRETRAINED_WEIGHTS_DIR)
        model_state = model.state_dict()
        filtered_state = {}
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape and not k.startswith(HEAD_PREFIX):
                filtered_state[k] = v
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        miss_head = [k for k in missing if k.startswith(HEAD_PREFIX)]
        miss_other = [k for k in missing if not k.startswith(HEAD_PREFIX)]
        print(f"[Model] 预训练 backbone 加载: loaded={len(filtered_state)}, missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"[Model] missing.{HEAD_PREFIX[:-1]} (期望, num_classes 变化): {len(miss_head)} -> {miss_head[:3]}")
        if miss_other:
            print(f"[Model] missing.other (警告): {len(miss_other)} -> {miss_other[:5]}")
    else:
        print(f"[Model] 警告: 预训练权重不存在: {PRETRAINED_WEIGHTS_DIR}")

    return model


def setup_trainable_params(model, lr_head):
    """设置可训练参数 - Phase 1 Head & Norm Warm-up (§3.4): 分类头 + LayerNorm/归一化层 affine 参数"""
    for param in model.parameters():
        param.requires_grad = False

    head_params = []
    for name, param in model.named_parameters():
        if name.startswith(HEAD_PREFIX):
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
    task_dir, checkpoint_dir = setup_output_dirs(TASK_NAME)
    logger = setup_logging(TASK_NAME, task_dir)
    logger.info(f"[Run] SEED={SEED}  TEST_SEED={TEST_SEED}  "
                f"RUN_SUFFIX={os.environ.get('RUN_SUFFIX', '-')}  "
                f"INIT_FROM={os.environ.get('INIT_FROM', '') or '(默认预训练源)'}")

    seed_everything(SEED)
    device = get_device()

    from datasets import DATA_ROOT as data_root

    total_pos = 3762 + 1028 + 1188
    total_neg = 4275 + 1009 + 932 + 999 + 1172
    total_all = total_pos + total_neg

    logger.info("=" * 60)
    logger.info("实验G: BeiTv2-Base 息肉二分类 (timm)")
    logger.info("=" * 60)
    logger.info(f"模型: timm {MODEL_NAME}")
    logger.info(f"输入尺寸: {IMG_SIZE} (预训练要求)")
    logger.info("【数据来源构成】")
    logger.info(f"  正类: PolypGen正类(3762) + HyperKvasir-polyp(1028) + GastroEndoNet-Polyp(1188) = {total_pos}")
    logger.info(f"  负类: PolypGen负类(4275) + HyperKvasir-cecum(1009) + HyperKvasir-zline(932) + HyperKvasir-pylorus(999) + GastroEndoNet-Polyp_N(1172) = {total_neg}")
    logger.info(f"  来源总计: {total_all}")
    logger.info("")
    logger.info("【预留与划分】")
    logger.info(f"  负样本预留: {RESERVE_TEST_NEG}")
    logger.info(f"  训练/验证可用: 正类{total_pos} + 负类{total_neg - RESERVE_TEST_NEG} = {total_pos + total_neg - RESERVE_TEST_NEG}")
    logger.info("")
    logger.info(f"总Epochs: {TOTAL_EPOCHS} (P1={NUM_EPOCHS_PHASE1}, P2={NUM_EPOCHS_PHASE2})")
    logger.info(f"Batch: {BATCH_SIZE}, LR_Head: {LR_HEAD}, LR_Full: {LR_FULL}")
    logger.info(f"输出目录: {task_dir}")
    logger.info("=" * 60)

    all_samples = PolypDataset.build_samples(
        pos_sources=POS_SOURCES,
        neg_sources=NEG_SOURCES,
        seed=TEST_SEED,
    )
    logger.info(f"[Dataset] 总样本数: {len(all_samples)}")

    pos_samples = [(p, l) for p, l in all_samples if l == 1]
    neg_samples = [(p, l) for p, l in all_samples if l == 0]
    logger.info(f"[Dataset] 正样本: {len(pos_samples)}, 负样本: {len(neg_samples)}")

    assert len(neg_samples) > RESERVE_TEST_NEG, f"负样本数量不足: {len(neg_samples)} < {RESERVE_TEST_NEG}"
    # 预留哪些负样本由 TEST_SEED 固定 -> test 集跨 run / 跨模型恒定
    random.Random(TEST_SEED).shuffle(neg_samples)
    neg_reserved = neg_samples[:RESERVE_TEST_NEG]
    neg_for_split = neg_samples[RESERVE_TEST_NEG:]
    pos_for_split = pos_samples

    cvc_dir = data_root / "CVC-ClinicDB" / "Original"
    etis_dir = data_root / "ETIS-LaribPolypDB" / "images"
    colondb_dir = data_root / "CVC-ColonDB" / "images"

    independent_test_pos = []
    for p in sorted(cvc_dir.glob("*.png")):
        independent_test_pos.append((str(p), 1))
    for p in sorted(etis_dir.glob("*.png")):
        independent_test_pos.append((str(p), 1))
    for p in sorted(colondb_dir.glob("*.png")):
        independent_test_pos.append((str(p), 1))

    independent_test_samples = independent_test_pos + neg_reserved

    all_for_split = pos_for_split + neg_for_split
    # train/val 随训练 seed 重抽(只换成员, 数量不变)
    random.Random(SEED).shuffle(all_for_split)

    train_size = int(len(all_for_split) * 0.8)
    train_samples = all_for_split[:train_size]
    val_samples = all_for_split[train_size:]

    logging.info(
        "[Split] test_seed=%d (固定)  trainval_seed=%d (随训练 seed)  "
        "n_train=%d n_val=%d n_test=%d  MD5(train/val/test)=%s/%s/%s"
        % (TEST_SEED, SEED, len(train_samples), len(val_samples),
           len(independent_test_samples),
           _partition_fingerprint(train_samples),
           _partition_fingerprint(val_samples),
           _partition_fingerprint(independent_test_samples))
    )

    random.Random(SEED).shuffle(train_samples)
    random.Random(SEED).shuffle(val_samples)
    random.Random(TEST_SEED).shuffle(independent_test_samples)

    train_pos = sum(1 for _, l in train_samples if l == 1)
    train_neg = sum(1 for _, l in train_samples if l == 0)
    val_pos = sum(1 for _, l in val_samples if l == 1)
    val_neg = sum(1 for _, l in val_samples if l == 0)

    total_available = len(pos_samples) + len(neg_samples) - RESERVE_TEST_NEG

    logger.info("")
    logger.info("【训练/验证数据划分结果】")
    logger.info(f"  训练集: {len(train_samples)} (正{train_pos}, 负{train_neg})")
    logger.info(f"  验证集: {len(val_samples)} (正{val_pos}, 负{val_neg})")
    logger.info(f"  独立测试集: {len(independent_test_samples)} (正{len(independent_test_pos)}, 负{len(neg_reserved)})")

    train_ds = PolypDataset(train_samples, img_size=IMG_SIZE, transform=get_train_transforms(IMG_SIZE))
    val_ds = PolypDataset(val_samples, img_size=IMG_SIZE, transform=get_val_transforms(IMG_SIZE))

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

    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    try:
        compiled = torch.compile(model, mode="default", fullgraph=False)
        with torch.no_grad():
            _ = compiled(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))
        model = compiled
        logger.info("[Model] torch.compile 已应用并通过 smoke test (mode=default)")
    except Exception as e:
        logger.warning(f"[Model] torch.compile 失败({type(e).__name__}),继续使用未编译模型: {str(e)[:120]}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer_p1, warmup_count = setup_trainable_params(model, LR_HEAD)
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=NUM_EPOCHS_PHASE1, eta_min=1e-6)

    all_params = list(model.parameters())
    optimizer_p2 = optim.AdamW(all_params, lr=LR_FULL, weight_decay=WEIGHT_DECAY)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=NUM_EPOCHS_PHASE2, eta_min=1e-6)

    scaler = GradScaler() if USE_AMP and device.type == "cuda" else None

    config = {
        "phase1_epochs": NUM_EPOCHS_PHASE1,
        "phase2_epochs": NUM_EPOCHS_PHASE2,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "early_stop_patience": 5,
        "phase2_load_best_phase1": True,
    }

    logger.info(f"[Train] Phase1 可训练参数: {warmup_count}")

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
        is_binary=True,
        num_classes=NUM_CLASSES,
        is_v6=False,
        seed=SEED,
    )


    logger.info(f"\n训练完成! Best AUC={best_metric:.4f} @ Epoch {best_epoch}")

    evaluate_test_set(model, independent_test_samples, task_dir, device, logger)


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
