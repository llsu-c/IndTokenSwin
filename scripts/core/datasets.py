"""
datasets.py
通用医学图像数据集加载模块
支持实验A/B/C/D/E的数据加载
"""

import hashlib
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image


# ===========================
# 数据集路径配置
# ===========================
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = Path("E:/data")

DATA_PATHS = {
    # PolypGen
    "polypgen_positive": DATA_ROOT / "PolypGen" / "positive",
    "polypgen_negative": DATA_ROOT / "PolypGen" / "negative",

    # HyperKvasir
    "hyperkvasir_labels": DATA_ROOT / "HyperKvasir" / "image-labels.csv",

    # Kvasir-Capsule
    "kvasir_capsule": DATA_ROOT / "Kvasir-Capsule",

    # GastroEndoNet
    "gastro_gerd": DATA_ROOT / "GastroEndoNet Comprehensive Endoscopy Image Dataset for GERD and Polyp Detection" / "Gerd",
    "gastro_gerd_normal": DATA_ROOT / "GastroEndoNet Comprehensive Endoscopy Image Dataset for GERD and Polyp Detection" / "Gerd Normal",
    "gastro_polyp": DATA_ROOT / "GastroEndoNet Comprehensive Endoscopy Image Dataset for GERD and Polyp Detection" / "Polyp",
    "gastro_polyp_normal": DATA_ROOT / "GastroEndoNet Comprehensive Endoscopy Image Dataset for GERD and Polyp Detection" / "Polyp Normal",

    # CVC/ETIS/ColonDB (独立测试集)
    "cvc_clinicdb": DATA_ROOT / "CVC-ClinicDB",
    "cvc_colondb": DATA_ROOT / "CVC-ColonDB",
    "etis": DATA_ROOT / "ETIS-LaribPolypDB",

    # Messidor-2
    "messidor_images": DATA_ROOT / "Messidor-2" / "images",
    "messidor_labels": DATA_ROOT / "Messidor-2" / "messidor_data.csv",

    # SIIM-ISIC
    "siim_images": DATA_ROOT / "SIIM-ISIC Melanoma Classification_256_256" / "image",
    "siim_labels": DATA_ROOT / "SIIM-ISIC Melanoma Classification_256_256" / "train-metadata.csv",

    # Refined IDRiD
    "idrid_train": DATA_ROOT / "Refined IDRiD" / "Train" / "Images",
    "idrid_test": DATA_ROOT / "Refined IDRiD" / "Test" / "Images",
}


# ===========================
# 通用数据集类
# ===========================

class MedicalImageDataset(Dataset):
    """通用医学图像分类数据集"""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        img_size: int = 256,
        transform=None,
    ):
        self.samples = samples
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


class PolypDataset(MedicalImageDataset):
    """实验A: 息肉二分类数据集"""

    @staticmethod
    def build_samples(
        pos_sources: Dict[str, int] = None,
        neg_sources: Dict[str, int] = None,
        seed: int = 42,
    ) -> List[Tuple[str, int]]:
        """
        构建息肉二分类样本列表

        Args:
            pos_sources: 正类来源 {"source_name": max_count}
            neg_sources: 负类来源 {"source_name": max_count}
            seed: 随机种子
        """
        import pandas as pd
        random.seed(seed)
        samples = []

        # 正类来源
        if pos_sources is None:
            pos_sources = {
                "polypgen_positive": 3762,
                "hyperkvasir_polyp": 1028,
            }

        for source, max_count in pos_sources.items():
            if source == "polypgen_positive":
                # PolypGen images are in a subfolder called "images"
                pos_dir = Path(DATA_PATHS["polypgen_positive"])
                if (pos_dir / "images").exists():
                    paths = sorted((pos_dir / "images").glob("*.jpg"))[:max_count]
                else:
                    paths = sorted(pos_dir.glob("*.jpg"))[:max_count]
                samples.extend([(str(p), 1) for p in paths])
            elif source == "hyperkvasir_polyp":
                df = pd.read_csv(DATA_PATHS["hyperkvasir_labels"])
                polyp_df = df[df["Finding"] == "polyps"]
                for _, row in polyp_df.head(max_count).iterrows():
                    video_file = row["Video file"]
                    img_path = DATA_ROOT / "HyperKvasir" / "lower-gi-tract" / "pathological-findings" / "polyps" / (video_file + ".jpg")
                    if img_path.exists():
                        samples.append((str(img_path), 1))
            elif source == "gastro_polyp":
                paths = sorted(Path(DATA_PATHS["gastro_polyp"]).glob("*.jpg"))[:max_count]
                samples.extend([(str(p), 1) for p in paths])

        # 负类来源
        if neg_sources is None:
            neg_sources = {
                "polypgen_negative": 4275,
                "hyperkvasir_cecum": 1009,
                "hyperkvasir_zline": 932,
                "hyperkvasir_pylorus": 999,
                "gastro_polyp_normal": 1172,
            }

        for source, max_count in neg_sources.items():
            if source == "polypgen_negative":
                # PolypGen images are in a subfolder called "images"
                neg_dir = Path(DATA_PATHS["polypgen_negative"])
                if (neg_dir / "images").exists():
                    paths = sorted((neg_dir / "images").glob("*.jpg"))[:max_count]
                else:
                    paths = sorted(neg_dir.glob("*.jpg"))[:max_count]
                samples.extend([(str(p), 0) for p in paths])
            elif source == "hyperkvasir_cecum":
                df = pd.read_csv(DATA_PATHS["hyperkvasir_labels"])
                cecum_df = df[(df["Organ"] == "Lower GI") & (df["Finding"] == "cecum")]
                for _, row in cecum_df.head(max_count).iterrows():
                    video_file = row["Video file"]
                    img_path = DATA_ROOT / "HyperKvasir" / "lower-gi-tract" / "anatomical-landmarks" / "cecum" / (video_file + ".jpg")
                    if img_path.exists():
                        samples.append((str(img_path), 0))
            elif source == "hyperkvasir_zline":
                df = pd.read_csv(DATA_PATHS["hyperkvasir_labels"])
                zline_df = df[(df["Organ"] == "Upper GI") & (df["Finding"] == "z-line")]
                for _, row in zline_df.head(max_count).iterrows():
                    video_file = row["Video file"]
                    img_path = DATA_ROOT / "HyperKvasir" / "upper-gi-tract" / "anatomical-landmarks" / "z-line" / (video_file + ".jpg")
                    if img_path.exists():
                        samples.append((str(img_path), 0))
            elif source == "hyperkvasir_pylorus":
                df = pd.read_csv(DATA_PATHS["hyperkvasir_labels"])
                pylorus_df = df[(df["Organ"] == "Upper GI") & (df["Finding"] == "pylorus")]
                for _, row in pylorus_df.head(max_count).iterrows():
                    video_file = row["Video file"]
                    img_path = DATA_ROOT / "HyperKvasir" / "upper-gi-tract" / "anatomical-landmarks" / "pylorus" / (video_file + ".jpg")
                    if img_path.exists():
                        samples.append((str(img_path), 0))
            elif source == "gastro_polyp_normal":
                paths = sorted(Path(DATA_PATHS["gastro_polyp_normal"]).glob("*.jpg"))[:max_count]
                samples.extend([(str(p), 0) for p in paths])

        random.shuffle(samples)
        return samples


class HyperKvasirDataset(MedicalImageDataset):
    """实验B1: HyperKvasir 12类精选版数据集"""

    CLASS_MAPPING = {
        "barretts": 0,
        "esophagitis": 1,
        "polyp": 2,
        "ulcerative-colitis": 3,
        "normal-z-line": 4,
        "normal-pylorus": 5,
        "normal-cecum": 6,
        "retroflex-stomach": 7,
        "retroflex-rectum": 8,
        "bbps": 9,
        "dyed": 10,
        "stool": 11,
    }

    @staticmethod
    def build_samples(seed: int = 42) -> List[Tuple[str, int]]:
        """构建HyperKvasir 12类样本列表"""
        import pandas as pd
        random.seed(seed)

        df = pd.read_csv(DATA_PATHS["hyperkvasir_labels"])

        finding_to_class = {
            "barretts": 0, "barretts-short-segment": 0,
            "esophagitis-a": 1, "esophagitis-b-d": 1,
            "polyps": 2,
            "ulcerative-colitis-grade-1": 3, "ulcerative-colitis-grade-2": 3,
            "ulcerative-colitis-grade-3": 3, "ulcerative-colitis-grade-0-1": 3,
            "ulcerative-colitis-grade-1-2": 3, "ulcerative-colitis-grade-2-3": 3,
            "z-line": 4,
            "pylorus": 5,
            "cecum": 6,
            "retroflex-stomach": 7,
            "retroflex-rectum": 8,
            "bbps-0-1": 9, "bbps-2-3": 9,
            "dyed-lifted-polyps": 10, "dyed-resection-margins": 10,
            "impacted-stool": 11, "hemorrhoids": 11,
        }

        # finding到子文件夹的映射
        finding_to_subpath = {
            # Upper GI pathological
            "barretts": "upper-gi-tract/pathological-findings/barretts",
            "barretts-short-segment": "upper-gi-tract/pathological-findings/barretts-short-segment",
            "esophagitis-a": "upper-gi-tract/pathological-findings/esophagitis-a",
            "esophagitis-b-d": "upper-gi-tract/pathological-findings/esophagitis-b-d",
            # Lower GI pathological
            "polyps": "lower-gi-tract/pathological-findings/polyps",
            "hemorrhoids": "lower-gi-tract/pathological-findings/hemorrhoids",
            "ulcerative-colitis-grade-0-1": "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-0-1",
            "ulcerative-colitis-grade-1": "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1",
            "ulcerative-colitis-grade-1-2": "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1-2",
            "ulcerative-colitis-grade-2": "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2",
            "ulcerative-colitis-grade-2-3": "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2-3",
            "ulcerative-colitis-grade-3": "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-3",
            # Upper GI anatomical
            "z-line": "upper-gi-tract/anatomical-landmarks/z-line",
            "pylorus": "upper-gi-tract/anatomical-landmarks/pylorus",
            "retroflex-stomach": "upper-gi-tract/anatomical-landmarks/retroflex-stomach",
            # Lower GI anatomical
            "cecum": "lower-gi-tract/anatomical-landmarks/cecum",
            "ileum": "lower-gi-tract/anatomical-landmarks/ileum",
            "retroflex-rectum": "lower-gi-tract/anatomical-landmarks/retroflex-rectum",
            # Lower GI quality-of-mucosal-views
            "bbps-0-1": "lower-gi-tract/quality-of-mucosal-views/bbps-0-1",
            "bbps-2-3": "lower-gi-tract/quality-of-mucosal-views/bbps-2-3",
            "impacted-stool": "lower-gi-tract/quality-of-mucosal-views/impacted-stool",
            # Lower GI therapeutic-interventions
            "dyed-lifted-polyps": "lower-gi-tract/therapeutic-interventions/dyed-lifted-polyps",
            "dyed-resection-margins": "lower-gi-tract/therapeutic-interventions/dyed-resection-margins",
        }

        samples = []
        for _, row in df.iterrows():
            finding = row["Finding"]
            if finding in finding_to_class:
                video_file = row["Video file"]
                if finding in finding_to_subpath:
                    img_path = DATA_ROOT / "HyperKvasir" / finding_to_subpath[finding] / (video_file + ".jpg")
                else:
                    # fallback: 根据organ判断
                    organ = row["Organ"]
                    if organ == "Lower GI":
                        img_path = DATA_ROOT / "HyperKvasir" / "lower-gi-tract" / (video_file + ".jpg")
                    else:
                        img_path = DATA_ROOT / "HyperKvasir" / "upper-gi-tract" / (video_file + ".jpg")
                if img_path.exists():
                    samples.append((str(img_path), finding_to_class[finding]))

        random.shuffle(samples)
        return samples


class KvasirCapsuleDataset(MedicalImageDataset):
    """实验B2: Kvasir-Capsule 12类数据集"""

    CLASS_MAPPING = {
        "normal_clean_mucosa": 0,
        "ileocecal_valve": 1,
        "reduced_mucosal_view": 2,
        "pylorus": 3,
        "angiectasia": 4,
        "ulcer": 5,
        "foreign_body": 6,
        "lymphangiectasia": 7,
        "erosion": 8,
        "blood_fresh": 9,
        "erythema": 10,
        "polyp": 11,
    }

    @staticmethod
    def build_samples(
        normal_max: int = 5000,
        seed: int = 42,
    ) -> List[Tuple[str, int]]:
        """构建Kvasir-Capsule 12类样本列表"""
        random.seed(seed)
        samples = []
        root = DATA_PATHS["kvasir_capsule"]

        for folder_name, class_idx in KvasirCapsuleDataset.CLASS_MAPPING.items():
            folder = root / folder_name
            if folder.exists():
                paths = sorted(folder.glob("*.jpg"))
                if folder_name == "normal_clean_mucosa":
                    random.shuffle(paths)
                    paths = paths[:normal_max]
                samples.extend([(str(p), class_idx) for p in paths])

        random.shuffle(samples)
        return samples


class SIIMDataset(MedicalImageDataset):
    """实验C2: SIIM-ISIC 黑色素瘤二分类数据集"""

    @staticmethod
    def build_samples(seed: int = 42) -> List[Tuple[str, int]]:
        """构建SIIM-ISIC样本列表"""
        import pandas as pd
        random.seed(seed)

        df = pd.read_csv(DATA_PATHS["siim_labels"])
        samples = []

        for _, row in df.iterrows():
            target = row["target"]
            if pd.isna(target):
                continue

            isic_id = row["isic_id"]
            img_path = DATA_PATHS["siim_images"] / f"{isic_id}.jpg"

            if img_path.exists():
                samples.append((str(img_path), int(target)))

        random.shuffle(samples)
        return samples


# OversampledSIIMDataset 的模块级几何变换函数（pickle-friendly，替代原 lambda）
def _oversample_aug_identity(img):
    return img


def _oversample_aug_hflip(img):
    return transforms.functional.hflip(img)


def _oversample_aug_vflip(img):
    return transforms.functional.vflip(img)


def _oversample_aug_rot90(img):
    return transforms.functional.rotate(img, 90)


def _oversample_aug_rot180(img):
    return transforms.functional.rotate(img, 180)


def _oversample_aug_rot270(img):
    return transforms.functional.rotate(img, 270)


def _oversample_aug_hflip_rot90(img):
    return transforms.functional.hflip(transforms.functional.rotate(img, 90))


def _oversample_aug_hflip_rot270(img):
    return transforms.functional.hflip(transforms.functional.rotate(img, 270))


class OversampledSIIMDataset(MedicalImageDataset):
    """
    SIIM-ISIC 过采样数据集 - 解决类别不平衡问题

    通过对正样本(黑色素瘤)进行过采样，使正负样本比例接近1:1
    使用确定性的几何变换来生成多样化的增强样本

    注意：本类当前**未被任何训练脚本调用**。C2 系列（3:1 重平衡）脚本导入它
    但实际使用的是 PerEpochUndersampledSampler（欠采样），本类与配套的
    undersample_samples / _oversample_aug_* 一并处于未启用状态。
    保留是因为它是附录 C.2「Co-design with Re-balancing Strategies」中
    过采样对照臂的现成实现；若该臂不做，可连同 undersample_samples 与
    9 个 _oversample_aug_* 函数一并删除。
    """

    # 确定性几何变换列表（必须用模块级函数以支持 Windows spawn 多进程 pickle）
    GEOMETRIC_AUGMENTATIONS = [
        _oversample_aug_identity,    # 原始
        _oversample_aug_hflip,       # 水平翻转
        _oversample_aug_vflip,       # 垂直翻转
        _oversample_aug_rot90,       # 90度旋转
        _oversample_aug_rot180,      # 180度旋转
        _oversample_aug_rot270,      # 270度旋转
        _oversample_aug_hflip_rot90, # HFlip + 90°
        _oversample_aug_hflip_rot270, # HFlip + 270°
    ]

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        img_size: int = 256,
        transform=None,
        target_pos_ratio: float = 1.0,
    ):
        """
        初始化过采样数据集

        Args:
            samples: 原始样本列表 [(path, label), ...]
            img_size: 图像尺寸
            transform: 基础变换（如归一化）
            target_pos_ratio: 目标正样本比例，1.0表示1:1正负比例
        """
        self.img_size = img_size
        self.transform = transform

        # 分离正负样本
        pos_samples = [(p, l) for p, l in samples if l == 1]
        neg_samples = [(p, l) for p, l in samples if l == 0]

        n_pos = len(pos_samples)
        n_neg = len(neg_samples)

        # 计算需要的正样本数量
        if target_pos_ratio <= 0:
            target_n_pos = n_pos
        else:
            target_n_pos = int(n_neg * target_pos_ratio)

        # 过采样正样本
        oversampled_pos = []
        aug_idx = 0
        pos_idx = 0
        while len(oversampled_pos) < target_n_pos:
            # 循环使用正样本和变换
            path, label = pos_samples[pos_idx % n_pos]
            aug_fn = self.GEOMETRIC_AUGMENTATIONS[aug_idx % len(self.GEOMETRIC_AUGMENTATIONS)]
            oversampled_pos.append((path, label, aug_fn))  # 存储路径、标签和变换函数
            aug_idx += 1
            pos_idx += 1

        # 合并过采样正样本和所有负样本
        self.all_samples = oversampled_pos + [(p, l, None) for p, l in neg_samples]

        # 打乱顺序
        random.shuffle(self.all_samples)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, label, aug_fn = self.all_samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        # 应用几何增强
        if aug_fn is not None:
            img = aug_fn(img)

        # 调整尺寸
        img = transforms.functional.resize(img, (self.img_size, self.img_size))

        # 应用基础变换（归一化等）
        if self.transform:
            img = self.transform(img)

        return img, label

# ===========================
# 下采样 (neg 类)
# ===========================

def undersample_samples(
    samples: List[Tuple[str, int]],
    neg_pos_ratio: float = 5.0,
    seed: int = 42,
) -> List[Tuple[str, int]]:
    """
    负样本下采样到指定 neg:pos 比例 (默认 5:1)

    - 保留全部正样本
    - 随机抽取 n_pos * neg_pos_ratio 个负样本
    - 返回新 list (原始 samples 不变)

    Args:
        samples: 原始样本 [(path, label), ...]
        neg_pos_ratio: 目标 neg:pos 比例, 5.0 = 5:1
        seed: 随机种子 (可复现)

    Returns:
        新的下采样样本 list
    """
    pos = [(p, l) for p, l in samples if l == 1]
    neg = [(p, l) for p, l in samples if l == 0]

    n_pos = len(pos)
    n_neg = len(neg)
    target_n_neg = min(int(n_pos * neg_pos_ratio), n_neg)

    rng = random.Random(seed)
    rng.shuffle(neg)
    return pos + neg[:target_n_neg]


class PerEpochUndersampledSampler(Sampler):
    """
    每 epoch 重新做一次 5:1 neg 下采样的 Sampler

    行为:
        - 每次 __iter__ 调用: 用 (base_seed + epoch) 重新随机抽取 neg 子集
        - 跨 epoch: 模型能见到几乎全部 neg 数据 (n_neg >> target_n_neg)
        - 单 epoch: 仍是 neg:pos = neg_pos_ratio : 1

    用法:
        sampler = PerEpochUndersampledSampler(samples, neg_pos_ratio=5.0, base_seed=42)
        loader = DataLoader(ds, sampler=sampler, shuffle=False, ...)
        # 训练循环里每 epoch 调一次:
        sampler.set_epoch(epoch)
    """
    def __init__(
        self,
        samples: List[Tuple],
        neg_pos_ratio: float = 5.0,
        base_seed: int = 42,
    ):
        self.pos_indices: List[int] = [i for i, s in enumerate(samples) if s[1] == 1]
        self.neg_indices: List[int] = [i for i, s in enumerate(samples) if s[1] == 0]
        self.n_pos = len(self.pos_indices)
        self.n_neg = len(self.neg_indices)
        self.neg_pos_ratio = neg_pos_ratio
        self.target_n_neg = min(int(self.n_pos * neg_pos_ratio), self.n_neg)
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)

        # 重新随机抽 neg (不放回)
        neg_perm = torch.randperm(self.n_neg, generator=g)
        sampled_neg = [self.neg_indices[i] for i in neg_perm[:self.target_n_neg].tolist()]

        # 合并: 全 pos + 该 epoch 抽到的 neg
        indices = self.pos_indices + sampled_neg

        # 最终顺序随机化
        perm = torch.randperm(len(indices), generator=g)
        return iter([indices[i] for i in perm.tolist()])

    def __len__(self):
        return self.n_pos + self.target_n_neg


# ===========================
# 数据增强
# ===========================

def get_train_transforms(img_size: int = 256, use_augmentation: bool = True):
    """获取训练数据增强"""
    if use_augmentation:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_val_transforms(img_size: int = 256):
    """获取验证数据增强"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ===========================
# 数据集划分
# ===========================

def split_dataset(
    samples: List[Tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    _log: bool = True,
) -> Tuple[List, List, List]:
    """划分数据集"""
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    test_samples = [samples[i] for i in test_indices]

    return train_samples, val_samples, test_samples


def split_with_fixed_test(
    samples: List[Tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    test_seed: int = 42,
    trainval_seed: Optional[int] = None,
    stratified: bool = True,
) -> Tuple[List, List, List]:
    """按"test 固定、train/val 随训练 seed"的口径划分数据集。

    设计与用途（对应论文 Table 8 / Appendix A.1）:
      * test 集由 `test_seed` 唯一决定，跨 run、跨模型恒定不变 —— 所有模型在同一批
        测试图像上评估，因此 run i 的两两比较构成配对样本。
      * train / val 由 `trainval_seed`（训练脚本传入 `SEED`）重新抽取 —— 每个 run 的
        train/val 成员不同，故 run 间差异同时包含训练随机性与数据划分差异。
      * 每类的 val 样本数与 test_seed 划分完全一致，只有成员变、数量不变，
        因此 Table 2 的 Train / Val 规模在各 run 之间保持恒定，Table 7 的验证集
        指标也仍然可比。

    Args:
        stratified: True 用 `stratified_split`（保持类别比例），False 用 `split_dataset`
            （整表随机切），与各数据集原有的划分方式保持一致。
        trainval_seed: None 时退化为单一 seed 的固定划分（即 test_seed 决定三段）。
    """
    base = (
        stratified_split(samples, train_ratio, val_ratio, test_ratio, seed=test_seed, _log=False)
        if stratified
        else split_dataset(samples, train_ratio, val_ratio, test_ratio, seed=test_seed, _log=False)
    )
    if trainval_seed is None:
        logging.info("[Split] test/val seed 相同(%d)，三段时间固定" % test_seed)
        return base

    _, val0, test = base
    test_keys = {p for p, _ in test}
    pool = [(p, l) for p, l in samples if p not in test_keys]

    # 每类 val 的目标数量 = test_seed 划分下的每类 val 数量（只换成员，不改数量）
    val_quota = Counter(l for _, l in val0)
    by_class = defaultdict(list)
    for p, l in pool:
        by_class[l].append((p, l))

    rng = random.Random(trainval_seed)   # 独立随机源: 不扰动后续训练使用的全局 RNG
    train_out, val_out = [], []
    for l in sorted(by_class):
        items = list(by_class[l])
        rng.shuffle(items)
        k = val_quota.get(l, 0)
        val_out.extend(items[:k])
        train_out.extend(items[k:])
    rng.shuffle(train_out)
    rng.shuffle(val_out)

    logging.info(
        "[Split] test_seed=%d (跨 run/模型恒定)  trainval_seed=%d (随训练 seed)  "
        "n_train=%d n_val=%d n_test=%d  MD5(train/val/test)=%s/%s/%s"
        % (test_seed, trainval_seed, len(train_out), len(val_out), len(test),
           _partition_fingerprint(train_out),
           _partition_fingerprint(val_out),
           _partition_fingerprint(test))
    )
    return train_out, val_out, test


def _partition_fingerprint(samples: List[Tuple[str, int]]) -> str:
    """划分指纹: 对样本路径集合(排序后)求 md5 前 12 位。

    用于在日志中核验各 run 是否共用同一 train/val/test 划分 —— 样本数相同并不代表
    成员集相同, 只有指纹相同才能确认。
    """
    key = "\n".join(sorted(str(Path(p).as_posix()).lower() for p, _ in samples))
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def stratified_split(
    samples: List[Tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    _log: bool = True,
) -> Tuple[List, List, List]:
    """分层划分数据集（保持类别比例）"""
    random.seed(seed)

    class_samples = {}
    for path, label in samples:
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append((path, label))

    train_samples, val_samples, test_samples = [], [], []

    for label, label_samples in class_samples.items():
        random.shuffle(label_samples)
        n = len(label_samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_samples.extend(label_samples[:train_end])
        val_samples.extend(label_samples[train_end:val_end])
        test_samples.extend(label_samples[val_end:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # 划分指纹: 写进日志, 便于事后核验"各 run 是否共用同一划分"。
    # 只比对 train/val/test 的样本数是不够的 —— 旧版 RESERVE_SEED = SEED + 100 时
    # 三段数量完全相同, 但成员集完全不同; 指纹能把这种情况暴露出来。
    if _log:
        logging.info(
            "[Split] seed=%d  n_train=%d n_val=%d n_test=%d  MD5(train/val/test)=%s/%s/%s"
            % (seed, len(train_samples), len(val_samples), len(test_samples),
               _partition_fingerprint(train_samples),
               _partition_fingerprint(val_samples),
               _partition_fingerprint(test_samples))
        )

    return train_samples, val_samples, test_samples
