"""
train_utils.py
通用训练工具模块
"""

import os
import sys
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score,
    accuracy_score,
)

from datasets import get_train_transforms, get_val_transforms


# ===========================
# 日志配置
# ===========================

def setup_logging(task_name: str, run_dir: Path) -> logging.Logger:
    """配置日志

    日志文件名统一为 ``{run 目录名}_{YYYYmmdd_HHMMSS}.log``，即前缀与所在 run 目录名
    严格一致（含 ``_vN`` 运行序号），使归档内"日志 <-> run 目录"的对应关系可直接校验。
    """
    log_dir = run_dir / "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = log_dir / f"{run_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def setup_output_dirs(task_name: str) -> Tuple[Path, Path]:
    """设置输出目录

    若设置环境变量 RUN_SUFFIX，则输出目录固定为 {task_name}_{date}__v{RUN_SUFFIX}，
    以便与 run 编号 (_vN) 一一对应、内部路径自洽；否则维持原有的"日期 + 碰撞后缀"逻辑。
    """
    project_root = Path(__file__).parent
    # OUTPUT_ROOT 允许把结果写到独立根目录（Path B 续训用，避免与 Path A 混淆）
    out_root = os.environ.get("OUTPUT_ROOT")
    checkpoint_dir = Path(out_root) if out_root else project_root / "checkpoint"

    # 创建任务子文件夹，名称含任务名称及日期
    date_str = datetime.now().strftime('%Y%m%d')
    run_suffix = os.environ.get("RUN_SUFFIX")
    if run_suffix is not None:
        task_dir = checkpoint_dir / f"{task_name}_{date_str}__v{run_suffix}"
    else:
        task_dir = checkpoint_dir / f"{task_name}_{date_str}"
        run_num = 1
        while task_dir.exists():
            task_dir = checkpoint_dir / f"{task_name}_{date_str}_v{run_num}"
            run_num += 1

    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return task_dir, task_dir / "checkpoints"


# ===========================
# 随机种子
# ===========================

def seed_everything(seed: int):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """获取设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


# ===========================
# 评估指标
# ===========================

def compute_binary_metrics(all_labels, all_probs, threshold: float = 0.5) -> Dict:
    """计算二分类指标"""
    preds = (all_probs >= threshold).astype(int)
    labels = all_labels.astype(int)

    auc = roc_auc_score(labels, all_probs)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    tp = np.sum((labels == 1) & (preds == 1))
    mcc = 0
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)

    # 综合评分 = 2 项真调和均值: 2 / (1/AUC + 1/F1)
    # 防止除零: 各分量下界 1e-6
    _eps = 1e-6
    _auc = max(auc, _eps)
    _f1 = max(f1, _eps)
    auc_f1_score = 2.0 / (1.0 / _auc + 1.0 / _f1)

    # 平衡准确率 = (sens + spec) / 2 — 对不平衡二分类鲁棒
    balanced_acc = 0.5 * (sens + spec)

    # 医学偏 sens 综合分: 0.5*sens + 0.5*balanced_acc
    # = 0.75*sens + 0.25*spec — 强拒坍缩 (0.25), 弱权 spec (可接受临床过杀)
    combined_sens_balacc = 0.5 * sens + 0.5 * balanced_acc

    return {
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "sens": sens,
        "spec": spec,
        "acc": acc * 100,
        "auc_f1_score": auc_f1_score,
        "balanced_accuracy": balanced_acc,
        "combined_sens_balacc": combined_sens_balacc,
    }


# ===========================
# Focal Loss（极不平衡任务）
# ===========================

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    适用极不平衡二分类 / 多分类。Lin et al. 2017 (RetinaNet) 推荐 gamma=2.0, alpha=0.25。

    Args:
        gamma: 聚焦参数，越大对易分样本压制越强（默认 2.0）
        alpha: 每类权重，None 表示均匀加权（默认 None）
        num_classes: 类别数（默认 2，用于 scatter 的 one-hot 构造）
        label_smoothing: 标签平滑（默认 0.0，与 CrossEntropyLoss 一致用法）
        reduction: 'mean' / 'sum' / 'none'
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[Sequence[float]] = None,
                 num_classes: int = 2, label_smoothing: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.num_classes = num_classes
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            if self.label_smoothing > 0 and self.num_classes > 1:
                true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        probs = log_probs.exp()
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * true_dist * log_probs
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = loss * alpha_t.unsqueeze(1)
        if self.reduction == "mean":
            return loss.sum() / max(logits.size(0), 1)
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_param_groups(model, weight_decay: float) -> List[Dict]:
    """按 timm 惯例把参数分成「需要 weight decay」与「豁免 weight decay」两组。

    豁免集合来自模型自身的 `no_weight_decay_keywords()`（子串匹配），覆盖
    logit_scale、ind_q_bias/ind_v_bias、same_window_bias、decay_factor、
    cpb_mlp、window_bias.mlp、ind_initializer。

    理由：这些是 bias / 相对位置偏置 / 缩放系数，不是权重矩阵。对它们施加
    weight decay 会把它们持续推向 0，与学习率缩放混淆（logit_scale 尤其明显：
    它是 exp() 前的乘性因子，被 decay 压低等价于强行降低注意力温度）。

    若模型没有 `no_weight_decay_keywords` 方法，则退化为单一参数组，
    与改动前行为完全一致。

    Returns:
        两个 param group 的列表，可直接传给 optim.AdamW。
    """
    keywords = None
    if hasattr(model, "no_weight_decay_keywords"):
        try:
            keywords = set(model.no_weight_decay_keywords())
        except Exception:
            keywords = None

    if not keywords:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def compute_multiclass_metrics(all_labels, all_preds, all_probs=None, num_classes: int = 2) -> Dict:
    """计算多分类指标"""
    labels = all_labels.astype(int)
    preds = all_preds.astype(int)

    macro_prec = precision_score(labels, preds, average='macro', zero_division=0)
    macro_rec = recall_score(labels, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)

    weighted_prec = precision_score(labels, preds, average='weighted', zero_division=0)
    weighted_rec = recall_score(labels, preds, average='weighted', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    per_class_prec = precision_score(labels, preds, average=None, zero_division=0)
    per_class_rec = recall_score(labels, preds, average=None, zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    cm = confusion_matrix(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    accuracy = accuracy_score(labels, preds) * 100.0  # 0-100 百分数

    mAUC = None
    if all_probs is not None and num_classes > 2:
        try:
            mAUC = roc_auc_score(labels, all_probs, multi_class='ovr', average='macro')
        except:
            pass

    # 综合评分 = 4 项真调和均值: 4 / (1/mF1 + 1/wF1 + 1/Acc + 1/mAUC)
    # 防止除零: 各分量下界 1e-6
    # 注意: accuracy 是 0-100 百分数,需除 100 与 mF1/wF1/mAUC (0-1) 同量级
    _eps = 1e-6
    _mF1 = max(macro_f1, _eps)
    _wF1 = max(weighted_f1, _eps)
    _acc = max(accuracy / 100.0, _eps)
    _mAUC = max(mAUC, _eps) if mAUC is not None else _eps
    combined_score = 4.0 / (1.0 / _mF1 + 1.0 / _wF1 + 1.0 / _acc + 1.0 / _mAUC)

    return {
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "per_class_precision": per_class_prec,
        "per_class_recall": per_class_rec,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "kappa": kappa,
        "accuracy": accuracy,
        "mAUC": mAUC,
        "combined_score": combined_score,
    }


# ===========================
# 训练循环
# ===========================

def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch: int,
    phase_name: str = "Phase",
    grad_clip_norm: float = 1.0,
    logger=None,
) -> Dict:
    """训练一个epoch"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if scaler:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            acc = 100.0 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            msg = f"  {phase_name} Epoch {epoch} [LR={current_lr:.2e}] [{batch_idx+1}/{len(dataloader)}] Loss={avg_loss:.4f} Acc={acc:.2f}%"
            print(msg)
            if logger:
                logger.info(msg)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": 100.0 * correct / total
    }


@torch.no_grad()
def evaluate_binary(model, dataloader, criterion, device) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """评估二分类模型"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    metrics = compute_binary_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics, all_labels, all_probs


@torch.no_grad()
def evaluate_multiclass(model, dataloader, criterion, device, num_classes: int = 2) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """评估多分类模型"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_preds_list, all_labels = [], [], []

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        total_loss += loss.item()
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        all_probs.append(probs.cpu().numpy())
        all_preds_list.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels)

    metrics = compute_multiclass_metrics(all_labels, all_preds, all_probs, num_classes)
    metrics["loss"] = total_loss / len(dataloader)
    metrics["accuracy"] = 100.0 * correct / total

    return metrics, all_labels, all_preds, all_probs


# ===========================
# V6 极坐标分析工具
# ===========================

def print_v6_fusion_gate_values(model, logger, prefix=""):
    """打印所有 fusion_gate 参数的当前值"""
    gate_vals = []
    for name, param in model.named_parameters():
        if 'fusion_gate' in name:
            val = param.detach().flatten().tolist()
            if len(val) == 1:
                gate_vals.append(f"{name.split('.')[-1]}={val[0]:.4f}")
            else:
                gate_vals.append(f"{name.split('.')[-1]}=[{', '.join(f'{v:.2f}' for v in val[:4])}]")
    if gate_vals:
        logger.info(f"[V6/FusionGate] {prefix} " + " | ".join(gate_vals[:6]))


def print_v6_center_preference(model, logger, prefix=""):
    """打印多头图像中心偏好权重 - 每个center的详细权重"""
    for name, module in model.named_modules():
        if hasattr(module, 'get_head_image_center_preference'):
            try:
                prefs = module.get_head_image_center_preference()
                if prefs is not None:
                    # prefs shape: [num_heads, num_centers]
                    n_heads, n_centers = prefs.shape

                    # 从模块获取center名称
                    if hasattr(module, 'IMAGE_POSITION_CENTERS'):
                        center_names = [n.split(',')[0].strip('[] ') for n in str(module.IMAGE_POSITION_CENTERS[:n_centers]).split('],')]
                        center_names = ['C', 'TL', 'TR', 'BL', 'BR', 'T', 'B', 'L', 'R'][:n_centers]
                    else:
                        center_names = [f"C{i}" for i in range(n_centers)]

                    # 转换为百分比便于阅读
                    prefs_percent = prefs * 100

                    # 各头对不同center的平均权重
                    avg_prefs = prefs.mean(dim=0)  # [num_centers]
                    dominant_center = avg_prefs.argmax().item()
                    dom_name = center_names[dominant_center] if dominant_center < len(center_names) else f"C{dominant_center}"

                    # 构建详细日志
                    logger.info(f"[V6/CenterPref] {prefix} heads={n_heads} centers={n_centers} dominant={dom_name}")

                    # 打印平均各center权重 (百分比)
                    center_strs = []
                    for i, avg_w in enumerate(prefs_percent.mean(dim=0)):
                        c_name = center_names[i] if i < len(center_names) else f"C{i}"
                        center_strs.append(f"{c_name}={avg_w.item():.1f}%")
                    logger.info(f"[V6/CenterPref] {prefix} avg_weights: " + " | ".join(center_strs))

                    # 打印每个head的详细权重
                    for h in range(min(n_heads, 8)):  # 最多打印8个head
                        head_strs = []
                        for c in range(n_centers):
                            c_name = center_names[c] if c < len(center_names) else f"C{c}"
                            head_strs.append(f"{c_name}={prefs_percent[h, c].item():.1f}%")
                        logger.info(f"[V6/CenterPref] {prefix} head{h}: " + " | ".join(head_strs))
            except Exception as e:
                logger.info(f"[V6/CenterPref] {prefix} error: {e}")


def print_v6_radial_encoding_stats(model, logger, prefix=""):
    """打印径向编码统计"""
    for name, module in model.named_modules():
        if hasattr(module, 'polar_encoding'):
            try:
                enc = module.polar_encoding
                # 径向 MLP 参数统计
                if hasattr(enc, 'radial_mlp'):
                    for pname, param in enc.radial_mlp.named_parameters():
                        if param.requires_grad:
                            logger.info(f"[V6/Radial] {prefix} radial_mlp.{pname}: mean={param.mean().item():.4f} std={param.std().item():.4f}")

                # Omega 和 Phi (角度Fourier编码)
                if hasattr(enc, 'omega'):
                    omega = enc.omega.detach()
                    logger.info(f"[V6/Radial] {prefix} omega: mean={omega.mean().item():.4f} std={omega.std().item():.4f} range=[{omega.min().item():.4f}, {omega.max().item():.4f}]")

                if hasattr(enc, 'phi'):
                    phi = enc.phi.detach()
                    logger.info(f"[V6/Radial] {prefix} phi: mean={phi.mean().item():.4f} std={phi.std().item():.4f} range=[{phi.min().item():.4f}, {phi.max().item():.4f}]")

                # 融合门控
                if hasattr(enc, 'fusion_gate'):
                    gate = enc.fusion_gate.detach()
                    logger.info(f"[V6/Radial] {prefix} fusion_gate: mean={gate.mean().item():.4f} std={gate.std().item():.4f}")

                # 图像中心权重
                if hasattr(enc, 'head_image_center_logits'):
                    logits = enc.head_image_center_logits.detach()
                    logger.info(f"[V6/Radial] {prefix} center_logits: mean={logits.mean().item():.4f} std={logits.std().item():.4f}")
            except Exception as e:
                logger.info(f"[V6/Radial] {prefix} error: {e}")


def v6_epoch_analysis(model, logger, epoch, phase_name):
    """V6 极坐标分析 - 每个epoch结束后调用"""
    prefix = f"{phase_name}{epoch}"
    print_v6_fusion_gate_values(model, logger, prefix=prefix)
    print_v6_center_preference(model, logger, prefix=prefix)
    print_v6_radial_encoding_stats(model, logger, prefix=prefix)


# ===========================
# 检查点保存
# ===========================

def save_checkpoint(
    model,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    path: Path,
    phase: Optional[int] = None,
    optimizer=None,
    scheduler=None,
):
    """保存检查点"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "phase": phase,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    return path


# ===========================
# 训练主流程
# ===========================

def train_two_phase(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer_phase1,
    optimizer_phase2,
    scheduler_phase1,
    scheduler_phase2,
    scaler,
    device,
    config: Dict,
    logger,
    task_dir: Path,
    is_binary: bool = True,
    num_classes: int = 2,
    is_v6: bool = False,
    seed: Optional[int] = None,
    custom_val_metric: Optional[str] = None,
):
    """
    两阶段训练主流程

    Args:
        is_v6: 是否为V6模型，开启后将输出极坐标分析日志
        custom_val_metric: 自定义 val 选模指标 key (覆盖默认 "auc"/"macro_f1")。
                           需在 evaluate_* 返回的 val_stats dict 中存在 (如 "macro_f1" / "balanced_accuracy")。
        seed: 起始随机种子。若不为 None，则每个 epoch 训练前会以
              `train_loader.generator.manual_seed(seed + epoch)`
              重置 DataLoader 的 shuffle 顺序，使每 epoch 数据顺序
              都不同（要求 train_loader 创建时设置了 generator）。
    """
    from typing import Optional  # 局部导入，避免影响其他模块

    phase1_epochs = config.get("phase1_epochs", 10)
    phase2_epochs = config.get("phase2_epochs", 10)
    grad_clip_norm = config.get("grad_clip_norm", 1.0)
    metric_name = "auc" if is_binary else "macro_f1"
    if custom_val_metric is not None:
        metric_name = custom_val_metric
    early_stop_patience = config.get("early_stop_patience", 0)  # 0=禁用
    phase2_load_best_phase1 = config.get("phase2_load_best_phase1", False)

    best_metric = 0.0
    best_epoch = 0
    best_val_stats = None  # 记录 best epoch 的完整 val 指标 (供训练结束打印)
    history = []
    patience_counter = 0  # 每个 phase 独立计数的早停计数器

    # Phase 1
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1: Epochs 1-{phase1_epochs} (freeze backbone, train head)")
    if is_v6:
        logger.info(f"[V6] 极坐标模块参与训练")
    logger.info(f"{'='*60}")

    for epoch in range(1, phase1_epochs + 1):
        epoch_start = time.time()

        if seed is not None and hasattr(train_loader, "generator") and train_loader.generator is not None:
            train_loader.generator.manual_seed(seed + epoch)

        if hasattr(train_loader, "sampler") and train_loader.sampler is not None and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_epoch(
            model, train_loader, criterion, optimizer_phase1,
            scheduler_phase1, scaler, device, epoch, "P1", grad_clip_norm, logger
        )

        if is_binary:
            val_stats, _, _ = evaluate_binary(model, val_loader, criterion, device)
        else:
            val_stats, _, _, _ = evaluate_multiclass(model, val_loader, criterion, device, num_classes)

        if scheduler_phase1 is not None:
            scheduler_phase1.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler_phase1.get_last_lr()[0] if scheduler_phase1 else optimizer_phase1.param_groups[0]['lr']

        is_best = val_stats[metric_name] > best_metric
        if is_best:
            best_metric = val_stats[metric_name]
            best_epoch = epoch
            best_val_stats = dict(val_stats)
            patience_counter = 0
            ckpt_path = task_dir / "checkpoints" / f"best_model_phase1.pth"
            save_checkpoint(model, epoch, best_metric, best_epoch, ckpt_path, phase=1)
        elif early_stop_patience > 0:
            patience_counter += 1

        history.append({
            "epoch": epoch,
            "phase": 1,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            **{k: v for k, v in val_stats.items() if k != "loss"},
            "epoch_time": epoch_time,
            "lr": current_lr,
        })

        logger.info(format_epoch_log(epoch, phase1_epochs, epoch_time, current_lr, train_stats, val_stats, is_best, is_binary, metric_name))

        # V6 极坐标分析
        if is_v6:
            v6_epoch_analysis(model, logger, epoch, "P1-")

        # Phase 1 早停
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            logger.info(f"[EarlyStop] Phase 1: {early_stop_patience} epoch 无提升，提前停止于 epoch {epoch}")
            break

    # Phase 2
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 2: Epochs {phase1_epochs+1}-{phase1_epochs+phase2_epochs} (full training)")
    logger.info(f"{'='*60}")

    # Phase 2 启动前：可选用 Phase 1 最佳模型覆盖当前 model
    if phase2_load_best_phase1:
        p1_ckpt = task_dir / "checkpoints" / "best_model_phase1.pth"
        if p1_ckpt.exists():
            _ckpt = torch.load(p1_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(_ckpt["model_state_dict"])
            logger.info(f"[Phase2Init] 已从 Phase 1 最佳模型加载: {p1_ckpt} (best_metric={_ckpt.get('best_metric', 'N/A')})")
        else:
            logger.warning(f"[Phase2Init] Phase 1 最佳模型不存在: {p1_ckpt}，沿用当前 model")

    # Phase 2 早停计数器重置
    patience_counter = 0
    # Phase 2 独立最佳 - 重置 best_metric / best_epoch，不与 P1 比较
    best_metric = 0.0
    best_epoch = 0
    best_val_stats = None

    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(phase1_epochs + 1, phase1_epochs + phase2_epochs + 1):
        epoch_start = time.time()

        if seed is not None and hasattr(train_loader, "generator") and train_loader.generator is not None:
            train_loader.generator.manual_seed(seed + epoch)

        if hasattr(train_loader, "sampler") and train_loader.sampler is not None and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_epoch(
            model, train_loader, criterion, optimizer_phase2,
            scheduler_phase2, scaler, device, epoch, "P2", grad_clip_norm, logger
        )

        if is_binary:
            val_stats, _, _ = evaluate_binary(model, val_loader, criterion, device)
        else:
            val_stats, _, _, _ = evaluate_multiclass(model, val_loader, criterion, device, num_classes)

        if scheduler_phase2 is not None:
            scheduler_phase2.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler_phase2.get_last_lr()[0] if scheduler_phase2 else optimizer_phase2.param_groups[0]['lr']

        is_best = val_stats[metric_name] > best_metric
        if is_best:
            best_metric = val_stats[metric_name]
            best_epoch = epoch
            best_val_stats = dict(val_stats)
            patience_counter = 0
            ckpt_path = task_dir / "checkpoints" / f"best_model_phase2.pth"
            save_checkpoint(model, epoch, best_metric, best_epoch, ckpt_path, phase=2)
        elif early_stop_patience > 0:
            patience_counter += 1

        history.append({
            "epoch": epoch,
            "phase": 2,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            **{k: v for k, v in val_stats.items() if k != "loss"},
            "epoch_time": epoch_time,
            "lr": current_lr,
        })

        logger.info(format_epoch_log(epoch, phase1_epochs + phase2_epochs, epoch_time, current_lr, train_stats, val_stats, is_best, is_binary, metric_name))

        # V6 极坐标分析
        if is_v6:
            v6_epoch_analysis(model, logger, epoch, "P2-")

        # Phase 2 早停
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            logger.info(f"[EarlyStop] Phase 2: {early_stop_patience} epoch 无提升，提前停止于 epoch {epoch}")
            break

    # 保存最终模型
    final_path = task_dir / "checkpoints" / "final_model.pth"
    save_checkpoint(model, phase1_epochs + phase2_epochs, best_metric, best_epoch, final_path)

    # 打印训练历史
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING HISTORY")
    logger.info("=" * 60)
    for h in history:
        if is_binary:
            logger.info(f"Epoch {h['epoch']}: LR={h['lr']:.2e} | Train={h['train_acc']:.1f}% | Val AUC={h['auc']:.4f} F1={h['f1']:.4f}")
        else:
            logger.info(f"Epoch {h['epoch']}: LR={h['lr']:.2e} | Train={h['train_acc']:.1f}% | Val mF1={h['macro_f1']:.4f} wF1={h['weighted_f1']:.4f}")
    logger.info("=" * 60)
    logger.info(f"Best: {metric_name}={best_metric:.4f} @ Epoch {best_epoch}")
    if best_val_stats is not None:
        if is_binary:
            logger.info(
                f"Best Epoch {best_epoch} Val 全指标: "
                f"AUC={best_val_stats['auc']:.4f} "
                f"Sens={best_val_stats['sens']:.4f} "
                f"Spec={best_val_stats['spec']:.4f} "
                f"Prec={best_val_stats['precision']:.4f} "
                f"F1={best_val_stats['f1']:.4f} "
                f"MCC={best_val_stats['mcc']:.4f} "
                f"Bal-Acc={best_val_stats['balanced_accuracy']:.4f} "
                f"Combined={best_val_stats['combined_sens_balacc']:.4f}"
            )
        else:
            logger.info(
                f"Best Epoch {best_epoch} Val 全指标: "
                f"mF1={best_val_stats['macro_f1']:.4f} "
                f"wF1={best_val_stats['weighted_f1']:.4f} "
                f"Acc={best_val_stats['accuracy']:.2f}% "
                f"mAUC={best_val_stats.get('mAUC', 0):.4f} "
                f"Combined={best_val_stats['combined_score']:.4f}"
            )
    logger.info(f"Model saved: {final_path}")

    return history, best_metric, best_epoch


def format_epoch_log(epoch: int, total_epochs: int, epoch_time: float, lr: float,
                     train_stats: Dict, val_stats: Dict, is_best: bool, is_binary: bool,
                     metric_name: str = "auc") -> str:
    """格式化epoch日志

    Score= 显示当前 epoch 的实际选模指标 (与 [BEST] 判断一致),
    而非默认的 auc_f1_score (旧指标).
    """
    if is_binary:
        score = val_stats.get(metric_name, val_stats.get('auc_f1_score', 0))
        # 二分类额外显示 sens/spec, 方便诊断 combined_sens_balacc 内部构成
        sens = val_stats.get('sens', 0)
        spec = val_stats.get('spec', 0)
        return (f"Epoch {epoch}/{total_epochs} | {epoch_time:.1f}s | LR={lr:.2e} | "
                f"Train: {train_stats['accuracy']:.1f}% | "
                f"Val: AUC={val_stats['auc']:.4f} F1={val_stats['f1']:.4f} "
                f"Acc={val_stats['acc']:.1f}% "
                f"Sens={sens:.4f} Spec={spec:.4f} "
                f"Score({metric_name})={score:.4f}"
                f"{' [BEST]' if is_best else ''}")
    else:
        score = val_stats.get(metric_name, val_stats.get('combined_score', 0))
        return (f"Epoch {epoch}/{total_epochs} | {epoch_time:.1f}s | LR={lr:.2e} | "
                f"Train: {train_stats['accuracy']:.1f}% | "
                f"Val: mF1={val_stats['macro_f1']:.4f} wF1={val_stats['weighted_f1']:.4f} "
                f"Acc={val_stats['accuracy']:.1f}% mAUC={val_stats.get('mAUC', 0):.4f} "
                f"Score({metric_name})={score:.4f}"
                f"{' [BEST]' if is_best else ''}")
