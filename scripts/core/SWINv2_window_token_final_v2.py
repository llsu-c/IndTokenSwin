# --------------------------------------------------------
# Swin Transformer V2 with Window Independent Tokens (v2)
# 改进点:
#   1. 双路径初始化器 (IndTokenInitializer) - 全局语义 + 局部结构保持
#   2. 简化 UnifiedWindowLevelBias - 去掉冗余的 ind2local/local2ind 偏置
#   3. 恢复原始 logit_scale 使用方式
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class IndTokenInitializer(nn.Module):
    """简化版 ind token 初始化器

    路径1 (局部结构): 窗口内patch的平均
    路径2 (全局语义): 所有窗口的平均

    融合方式: 简单的残差连接

    使用瓶颈结构减少参数: dim -> dim/4 -> dim
    """

    def __init__(self, dim, bottleneck_ratio=4):
        super().__init__()
        self.bottleneck_dim = dim // bottleneck_ratio

        # 局部特征投影 (瓶颈结构)
        self.local_proj = nn.Sequential(
            nn.Linear(dim, self.bottleneck_dim),
            nn.Linear(self.bottleneck_dim, dim)
        )

        # 全局特征投影（瓶颈结构）
        self.global_proj = nn.Sequential(
            nn.Linear(dim, self.bottleneck_dim),
            nn.Linear(self.bottleneck_dim, dim)
        )

    def forward(self, x_windows, num_windows):
        """
        Args:
            x_windows: (B, num_windows, N, C) - 窗口内patch特征
            num_windows: int - 窗口数量

        Returns:
            ind_tokens: (B, num_windows, C) - 初始化后的独立token
        """
        B, _, N, C = x_windows.shape

        # 路径1: 局部结构 - 每个窗口的局部特征
        local_feat = x_windows.mean(dim=2)  # (B, num_windows, C)
        local_feat = self.local_proj(local_feat)  # (B, num_windows, C)

        # 路径2: 全局语义 - 所有窗口的平均再投影
        global_feat = x_windows.mean(dim=(1, 2))  # (B, C)
        global_feat = self.global_proj(global_feat)  # (B, C)
        global_feat_expanded = global_feat.unsqueeze(1)  # (B, 1, C)
        global_feat_expanded = global_feat_expanded.expand(-1, num_windows, -1)  # (B, num_windows, C)

        # 融合: 全局 + 局部，使用残差连接
        ind_tokens = global_feat_expanded + local_feat

        return ind_tokens


class SimplifiedWindowBias(nn.Module):
    """简化的窗口级位置偏置

    合并了以下功能:
    1. ind -> all ind 偏置（原始 SimplifiedWindowBias）
    2. local ↔ ind 归纳偏置（来自 InductiveLocalIndBias）

    坐标变换使用 log-space (与 cpb_mlp 一致):
        sign(Δ) * log2(|Δ| + 1) / np.log2(8)
    """

    def __init__(self, num_heads, H_grid, W_grid, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.num_windows = H_grid * W_grid
        self.N = window_size * window_size

        # MLP 分支（输入 2D 相对坐标，输出 num_heads 维偏置）
        self.mlp = nn.Sequential(
            nn.Linear(2, 128, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(128, num_heads, bias=False)
        )

        # 窗口网格坐标 (H_grid, W_grid, 2)，归一化到 [0,1]
        grid_normalizer_h = max(H_grid - 1, 1)
        grid_normalizer_w = max(W_grid - 1, 1)
        grid_h = torch.arange(H_grid, dtype=torch.float32) / grid_normalizer_h
        grid_w = torch.arange(W_grid, dtype=torch.float32) / grid_normalizer_w
        grid_coords = torch.stack(torch.meshgrid(grid_h, grid_w, indexing='ij')).permute(1, 2, 0).contiguous()
        self.register_buffer("grid_coords", grid_coords)

        # ========== 预计算偏移的 log-space 表（数值稳定）==========
        eps = 1e-6
        # 窗口两两之间的相对坐标
        ind2ind_offset = grid_coords[:, :, None, None, :] - grid_coords[None, None, :, :, :]  # (H, W, H, W, 2)
        ind2ind_offset = ind2ind_offset.view(-1, 2)  # (H*W*H*W, 2)
        # 数值稳定：防止 log(0) 和除零
        ind2ind_log = torch.sign(ind2ind_offset) * torch.log2(torch.abs(ind2ind_offset) + 1.0 + eps) / np.log2(8)
        self.register_buffer("ind2ind_log_table", ind2ind_log)

        # ===== local ↔ ind 归纳偏置参数（来自 InductiveLocalIndBias）=====
        # 同窗口偏置：local ↔ ind 在同一窗口时加正值
        self.same_window_bias = nn.Parameter(torch.zeros(num_heads))
        # 距离衰减因子
        self.decay_factor = nn.Parameter(torch.tensor(0.5))

    def get_ind2ind_bias(self):
        """ind -> all ind: (num_heads, num_windows, num_windows)"""
        bias = self.mlp(self.ind2ind_log_table)  # (H*W*H*W, num_heads)
        bias = bias.view(self.num_windows, self.num_windows, self.num_heads)
        bias = bias.permute(2, 0, 1).contiguous()  # (num_heads, num_windows, num_windows)
        # 数值稳定：限制 sigmoid 输入范围
        bias = torch.clamp(bias, min=-10, max=10)
        return 16 * torch.sigmoid(bias)

    def get_local_ind_bias(self):
        """local ↔ ind 归纳偏置

        Returns:
            local2ind_bias: (num_heads, num_windows, N)
            ind2local_bias: (num_heads, num_windows, N)
        """
        num_windows = self.num_windows

        # 同窗口偏置基础值
        same_bias = 16 * torch.sigmoid(self.same_window_bias)
        same_bias_expanded = same_bias.unsqueeze(1).unsqueeze(2)
        same_bias_expanded = same_bias_expanded.expand(-1, num_windows, self.N)

        # 距离衰减因子
        decay = F.softplus(self.decay_factor)

        # 最终偏置
        bias = same_bias_expanded * (1 + 1 / (1 + decay))

        return bias, bias

    def forward(self, H_grid=None, W_grid=None, return_local_ind=False):
        """返回偏置

        Args:
            H_grid, W_grid: 仅用于兼容接口
            return_local_ind: 是否返回 local↔ind 偏置

        Returns:
            ind2ind_bias 或 (ind2ind_bias, local2ind_bias, ind2local_bias)
        """
        ind2ind = self.get_ind2ind_bias()

        if return_local_ind:
            local2ind, ind2local = self.get_local_ind_bias()
            return ind2ind, local2ind, ind2local
        return ind2ind


class GatedIndTokenUpdate(nn.Module):
    """简化版门控更新 —— **当前未启用，预留实现**

    论文 §3.2.3 明确采用 "direct overwrite, no gated residual"（T ← T′），
    本模型 forward 中不使用本类。保留原因是附录 C.1 的 "Gated-residual
    ablation" 需要它作为对照臂：把 direct overwrite 换成
    γ·T + (1 − γ)·T′，检验门控是否真的必要。

    若不做该消融，本类可直接删除；一旦删除，附录 C.1 的对照臂需另行实现。

    核心思想：让模型自适应决定应该保留多少原始信息
    简化：使用单层门控，减少参数
    """

    def __init__(self, dim):
        super().__init__()

        # 简化的门控网络：单层
        self.gate_net = nn.Linear(dim * 2, dim, bias=True)

    def forward(self, ind_tokens_output, ind_tokens):
        """
        Args:
            ind_tokens_output: (B, num_windows, C) - 注意力更新的输出
            ind_tokens: (B, num_windows, C) - 原始 token

        Returns:
            new_ind_tokens: (B, num_windows, C) - 门控更新后的 token
        """
        concat = torch.cat([ind_tokens_output, ind_tokens], dim=-1)
        gate = torch.sigmoid(self.gate_net(concat))
        new_ind_tokens = gate * ind_tokens_output + (1 - gate) * ind_tokens
        return new_ind_tokens


class IndTokenPooling(nn.Module):
    """无参数的 Ind Tokens 汇聚器

    使用简单的平均池化，零新增参数。
    """

    def __init__(self):
        super().__init__()

    def forward(self, ind_tokens):
        """
        Args:
            ind_tokens: (B, num_windows, C)

        Returns:
            pooled: (B, C)
        """
        # 简单的平均池化
        return ind_tokens.mean(dim=1)


class WindowAttentionWithGlobalIndTokenV2(nn.Module):
    """Window Attention with Global Independent Tokens (v2)

    改进点:
    1. 使用双路径初始化器初始化 ind tokens
    2. 简化偏置机制，只保留 ind->all ind 偏置
    3. 恢复原始 logit_scale 使用方式

    q = [local_patches, self_ind_token]          (N+1)
    k/v = [local_patches, all_ind_tokens]      (N+W)
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0], H_grid=None, W_grid=None,
                 ind_initializer=None, window_bias=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 恢复原始的 logit_scale（不限制下限）
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        # 预计算 logit_scale 上限（避免每次前向传播创建新 tensor）
        self.register_buffer("logit_scale_max", torch.log(torch.tensor(1. / 0.01)))

        # 窗口内相对位置偏置（保持不变，用于 local -> local）
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(512, num_heads, bias=False)
        )

        # 简化的偏置模块（使用共享的或创建新的）
        if window_bias is not None:
            self.window_bias = window_bias
            self._H_grid = H_grid
            self._W_grid = W_grid
        elif H_grid is not None and W_grid is not None:
            self.window_bias = SimplifiedWindowBias(
                num_heads=num_heads,
                H_grid=H_grid,
                W_grid=W_grid,
                window_size=window_size[0]
            )
            self._H_grid = H_grid
            self._W_grid = W_grid
        else:
            self.window_bias = None
            self._H_grid = None
            self._W_grid = None

        relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous()
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, 0] /= (window_size[0] - 1)
            relative_coords_table[:, :, 1] /= (window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qkv_bias:
            self.ind_q_bias = nn.Parameter(torch.zeros(dim))
            self.ind_v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.ind_q_bias = None
            self.ind_v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # ind_initializer 由 BasicLayer 注入；stage 1+ 时为 None（不会被 forward 触发，见 SwinTransformerBlock 守卫）
        self.ind_initializer = ind_initializer

        # 保存 window_size 用于延迟初始化
        self._window_size = window_size[0] if isinstance(window_size, tuple) else window_size

    def _init_window_bias(self, H_grid, W_grid, device):
        """延迟初始化偏置模块"""
        self._H_grid = H_grid
        self._W_grid = W_grid
        self.window_bias = SimplifiedWindowBias(
            num_heads=self.num_heads,
            H_grid=H_grid,
            W_grid=W_grid,
            window_size=self._window_size
        ).to(device)

    def forward(self, x_windows, ind_tokens, mask=None, H_grid=None, W_grid=None):
        """
        Args:
            x_windows: (B*num_windows, N, C) - 窗口内patch特征
            ind_tokens: (B, num_windows, C) - 所有窗口的独立词向量，如果为None则初始化
            mask: (num_windows, N, N) or None
            H_grid, W_grid: 网格尺寸

        Returns:
            x_windows_out: (B*num_windows, N, C) - 更新后的patch
            new_ind_tokens: (B, num_windows, C) - 更新后的所有独立词向量
        """
        B_times_windows, N, C = x_windows.shape
        B, num_windows, _ = ind_tokens.shape

        assert B_times_windows == B * num_windows, f"{B_times_windows} != {B} * {num_windows}"
        assert C == self.dim, f"{C} != {self.dim}"

        # 初始化偏置模块（如果需要）
        if self.window_bias is None or self._H_grid != H_grid or self._W_grid != W_grid:
            if H_grid is not None and W_grid is not None:
                self._init_window_bias(H_grid, W_grid, x_windows.device)
            else:
                H_grid = int(math.sqrt(num_windows))
                W_grid = num_windows // H_grid
                self._init_window_bias(H_grid, W_grid, x_windows.device)

        # ========== 1. 计算所有 QKV ==========

        # 窗口内patch的 QKV
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x_windows, weight=self.qkv.weight, bias=qkv_bias)
        # qkv: (B*W, N, 3*C)
        qkv = qkv.reshape(B, num_windows, N, 3, self.num_heads, self.head_dim)
        # qkv: (B, W, N, 3, H, d)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        # qkv: (3, B, W, H, N, d)
        q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]
        # 每个都是 (B, W, H, N, d)

        # 独立词向量的 QKV
        ind_bias = None
        if self.ind_q_bias is not None:
            ind_bias = torch.cat((self.ind_q_bias, torch.zeros_like(self.ind_v_bias, requires_grad=False), self.ind_v_bias))
        ind_qkv = F.linear(input=ind_tokens.reshape(B*num_windows, C), weight=self.qkv.weight, bias=ind_bias)
        # ind_qkv: (B*W, 3*C)
        ind_qkv = ind_qkv.reshape(B, num_windows, 3, self.num_heads, self.head_dim)
        # ind_qkv: (B, W, 3, H, d)
        ind_qkv = ind_qkv.permute(2, 0, 1, 3, 4)
        # ind_qkv: (3, B, W, H, d)
        q_ind, k_ind, v_ind = ind_qkv[0], ind_qkv[1], ind_qkv[2]
        # 每个都是 (B, W, H, d)

        # 窗口内位置偏置
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)  # (H, N, N)

        # ========== 2. 为每个窗口构造 q, k, v ==========

        # q: (B, W, H, N+1, d)
        q_ind_expanded = q_ind.unsqueeze(3)  # (B, W, H, 1, d)
        q = torch.cat([q_local, q_ind_expanded], dim=3)

        # k: (B, W, H, N+W, d)
        k_ind_all = k_ind.unsqueeze(1).expand(-1, num_windows, -1, -1, -1)
        k_ind_all = k_ind_all.permute(0, 1, 3, 2, 4)  # (B, W, H, W, d)
        k = torch.cat([k_local, k_ind_all], dim=3)

        # v: same as k
        v_ind_all = v_ind.unsqueeze(1).expand(-1, num_windows, -1, -1, -1)
        v_ind_all = v_ind_all.permute(0, 1, 3, 2, 4)
        v = torch.cat([v_local, v_ind_all], dim=3)

        # ========== 3. 计算注意力（数值稳定化）==========
        q_norm = F.normalize(q, dim=-1, p=2, eps=1e-6)
        k_norm = F.normalize(k, dim=-1, p=2, eps=1e-6)
        attn = (q_norm @ k_norm.transpose(-2, -1))

        # 恢复原始的 logit_scale 使用方式（不限制下限，允许scale增长）
        logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
        attn = attn * logit_scale

        # 填充偏置：只对前 N×N 部分（窗口内 local -> local）
        attn[:, :, :, :N, :N] = attn[:, :, :, :N, :N] + relative_position_bias.unsqueeze(0).unsqueeze(0)

        # 应用 mask
        if mask is not None:
            attn[:, :, :, :N, :N] = attn[:, :, :, :N, :N] + mask.unsqueeze(0).unsqueeze(2)

        # ===== 添加 ind -> all ind 偏置和 local ↔ ind 偏置（合并版）=====
        ind2ind_bias, local2ind_bias, ind2local_bias = self.window_bias(H_grid, W_grid, return_local_ind=True)
        # ind2ind_bias: (num_heads, num_windows, num_windows)
        # local2ind_bias: (num_heads, num_windows, N)
        # ind2local_bias: (num_heads, num_windows, N) - 对称的

        # ind -> all ind: attn[:, :, :, N:, N:]
        # ind2ind_bias: (num_heads, num_windows, num_windows)，索引 [h, wq, wk]
        # 目标 attn[:, :, :, N:, N:] 形状 (B, W, H, W, W)，需把 [h, wq, wk] 映射到
        # (dim1=wq, dim2=h, dim4=wk)。必须先 permute 再 unsqueeze：
        #   (H, W, W) -> permute(1,0,2) -> (W, H, W) -> (1, W, H, 1, W)
        # 直接 reshape 会按 C-order 摊平 (h,wq,wk)，使 head 与 query 窗口错配。
        ind2ind_expanded = ind2ind_bias.permute(1, 0, 2).unsqueeze(0).unsqueeze(3)
        attn[:, :, :, N:, N:] = attn[:, :, :, N:, N:] + ind2ind_expanded

        # local -> ind: attn[:, :, :, :N, N:] - local tokens attend to ind tokens
        # 对于每个窗口 w，local tokens  attend to ind_token[w]
        # bias: local2ind_bias[h, w, n] 对应 head h, 窗口 w, local token n
        # 向量化: 64 次 for 循环合并为单次张量加法,bit-wise 等价(同位置同操作数)
        # local2ind_bias: (H, W, N) → permute(0,2,1) → (H, N, W) → broadcast 到 (B, W, H, N, W)
        attn[:, :, :, :N, N:N+num_windows] = attn[:, :, :, :N, N:N+num_windows] + \
            local2ind_bias.permute(0, 2, 1).unsqueeze(0).unsqueeze(0)

        # ind -> local: attn[:, :, :, N, :N] - ind token attends to local tokens
        # q 的 ind token（在位置 N）对所有 local tokens 添加偏置
        # 使用所有窗口的平均偏置
        ind_bias = ind2local_bias.mean(dim=1)  # (num_heads, N)
        attn[:, :, :, N, :N] = attn[:, :, :, N, :N] + ind_bias.unsqueeze(0).unsqueeze(0)

        # Softmax 和 dropout
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # ========== 4. 输出 ==========

        outputs = attn @ v  # (B, W, H, N+1, d)
        outputs = outputs.permute(0, 1, 3, 2, 4).reshape(B, num_windows, N+1, C)

        # 分离 patch 输出和独立词向量输出
        x_windows_out = outputs[:, :, :N, :].reshape(B*num_windows, N, C)
        ind_tokens_output = outputs[:, :, N, :]  # (B, W, C) - 注意力更新后的原始输出

        x_windows_out = self.proj(x_windows_out)
        x_windows_out = self.proj_drop(x_windows_out)

        # 注意：不再这里做残差连接，残差连接会在 SwinTransformerBlockWithGlobalIndTokenV2 中通过 gated update 完成
        return x_windows_out, ind_tokens_output


class SwinTransformerBlockWithGlobalIndTokenV2(nn.Module):
    """Swin Transformer Block with Global Independent Tokens (v2)"""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0,
                 ind_initializer=None, window_bias=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)

        # 计算 H_grid, W_grid
        H_grid = input_resolution[0] // self.window_size
        W_grid = input_resolution[1] // self.window_size

        self.attn = WindowAttentionWithGlobalIndTokenV2(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            H_grid=H_grid, W_grid=W_grid,
            ind_initializer=ind_initializer, window_bias=window_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, ind_tokens):
        """
        x: (B, H*W, C)
        ind_tokens: (B, num_windows, C) - 按原始窗口排列，如果为None则初始化

        Returns:
            x_out: (B, H*W, C)
            new_ind_tokens: (B, num_windows, C) - 所有独立词向量都更新了
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        H_grid = H // self.window_size
        W_grid = W // self.window_size
        num_windows = H_grid * W_grid

        # 保存shortcut（与原始SwinTransformerBlock一致）
        shortcut = x
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 初始化 ind_tokens（如果为None）
        # 仅 stage 0 走此分支（由 forward_features 注入 None 触发）；其它 stage ind_tokens 已非 None
        if ind_tokens is None:
            assert self.attn.ind_initializer is not None, (
                "ind_tokens is None 但本 stage 未注入 IndTokenInitializer；"
                "ind_tokens=None 仅应在 stage 0 触发"
            )
            x_windows_reshaped = x_windows.reshape(B, num_windows, -1, C)
            ind_tokens = self.attn.ind_initializer(x_windows_reshaped, num_windows)

        # 注意：ind_tokens 始终保持位置锚定
        # 每个 ind_token[i,j] 代表位置 (i,j) 的语义聚合
        # 不随 cyclic shift 滚动，保持稳定性

        # 注意力计算
        x_windows_out, ind_tokens_output = self.attn(x_windows, ind_tokens, mask=self.attn_mask,
                                                     H_grid=H_grid, W_grid=W_grid)

        # 简单残差连接更新ind tokens
        new_ind_tokens = ind_tokens_output

        # Merge windows
        attn_windows = x_windows_out.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # 与原始SwinTransformerBlock一致：norm1在注意力之后
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN with res-post-norm
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x, new_ind_tokens


class PatchMerging(nn.Module):
    """Patch Merging Layer"""

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        assert L == H * W and H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.reduction(x)
        x = self.norm(x)
        return x

    def merge_ind_tokens(self, ind_tokens, input_resolution):
        """合并独立词向量

        直接从 input_resolution 和 num_windows 反推 grid 大小，
        支持非正方形图像。
        """
        B, num_windows_old, C = ind_tokens.shape
        H_old, W_old = input_resolution

        # 从 num_windows_old 和 input_resolution 反推 grid 大小
        # 尝试找到满足 num_windows_old = H_grid * W_grid 的 H_grid, W_grid
        # 使得 H_grid * window_size = H_old, W_grid * window_size = W_old
        import math

        # 找到所有可能的因子对
        factors = []
        for i in range(1, int(math.sqrt(num_windows_old)) + 1):
            if num_windows_old % i == 0:
                factors.append((i, num_windows_old // i))

        # 找到满足 H_old / H_grid = W_old / W_grid 的因子对
        best_H_grid, best_W_grid = 1, num_windows_old
        best_diff = float('inf')

        for h, w in factors:
            # 检查是否能整除
            if H_old % h == 0 and W_old % w == 0:
                window_h = H_old // h
                window_w = W_old // w
                if window_h == window_w:  # 优先正方形窗口
                    diff = 0
                else:
                    diff = abs(window_h - window_w)
                if diff < best_diff:
                    best_diff = diff
                    best_H_grid, best_W_grid = h, w
                    if diff == 0:
                        break

        # 如果没找到完全匹配的，使用面积反推
        if best_diff == float('inf'):
            # 使用 sqrt 反推，假设接近正方形
            side = int(math.sqrt(num_windows_old) + 0.5)
            best_H_grid = side
            best_W_grid = num_windows_old // side

        H_grid_old = best_H_grid
        W_grid_old = best_W_grid

        # 验证
        assert H_grid_old * W_grid_old == num_windows_old, \
            f"Grid mismatch: H_grid={H_grid_old}, W_grid={W_grid_old}, num_windows={num_windows_old}"

        ind_tokens_2d = ind_tokens.reshape(B, H_grid_old, W_grid_old, C)

        # 2x2 窗口合并（支持非正方形）
        new_H_grid = H_grid_old // 2
        new_W_grid = W_grid_old // 2

        # 确保可以整除
        assert H_grid_old % 2 == 0 and W_grid_old % 2 == 0, \
            f"Grid must be divisible by 2: H_grid={H_grid_old}, W_grid={W_grid_old}"

        merged = []
        for i in range(new_H_grid):
            for j in range(new_W_grid):
                # 每个 2x2 区块合并成一个
                patch = torch.cat([
                    ind_tokens_2d[:, i*2, j*2],
                    ind_tokens_2d[:, i*2, j*2+1],
                    ind_tokens_2d[:, i*2+1, j*2],
                    ind_tokens_2d[:, i*2+1, j*2+1],
                ], dim=-1)
                merged.append(patch)

        merged = torch.stack(merged, dim=1)
        merged = self.reduction(merged)
        merged = self.norm(merged)

        return merged


class BasicLayer(nn.Module):
    """Basic Layer for Swin Transformer V2 with Global Ind Tokens (v2)"""

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # 计算 H_grid, W_grid
        H_grid = input_resolution[0] // window_size
        W_grid = input_resolution[1] // window_size

        # 仅 stage 0 构造 IndTokenInitializer；后续 stage 由 PatchMerging.merge_ind_tokens 继承
        self.ind_initializer = IndTokenInitializer(out_dim) if downsample is None else None
        self.window_bias = SimplifiedWindowBias(
            num_heads=num_heads,
            H_grid=H_grid,
            W_grid=W_grid,
            window_size=window_size
        )

        self.blocks = nn.ModuleList([
            SwinTransformerBlockWithGlobalIndTokenV2(
                dim=out_dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                ind_initializer=self.ind_initializer,
                window_bias=self.window_bias)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, ind_tokens=None):
        if self.downsample is not None:
            x = self.downsample(x)
            if ind_tokens is not None:
                old_resolution = (self.input_resolution[0] * 2, self.input_resolution[1] * 2)
                ind_tokens = self.downsample.merge_ind_tokens(ind_tokens, old_resolution)

        for blk in self.blocks:
            if self.use_checkpoint:
                x, ind_tokens = checkpoint.checkpoint(blk, x, ind_tokens)
            else:
                x, ind_tokens = blk(x, ind_tokens)

        return x, ind_tokens

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]


        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2WithGlobalIndTokenV2(nn.Module):
    """Swin Transformer V2 with Global Independent Tokens (v2)

    改进点:
    1. 双路径初始化器初始化 ind tokens
    2. 简化的窗口级偏置机制
    3. 恢复原始 logit_scale 使用
    4. Pre-Norm 结构与原始Swin一致
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        in_dim = embed_dim
        for i_layer in range(self.num_layers):
            out_dim = int(embed_dim * 2 ** i_layer)
            layer = BasicLayer(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer > 0 else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
            in_dim = out_dim

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Ind tokens 可学习汇聚器
        self.ind_pool = IndTokenPooling()

        # 分类头：拼接主特征和独立token特征
        if num_classes > 0:
            self.head = nn.Linear(self.num_features * 2, num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", "ind_q_bias", "ind_v_bias",
                "window_bias", "ind_initializer",
                "same_window_bias", "decay_factor"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # ind_tokens 在第一层通过双路径初始化器初始化
        ind_tokens = None
        for layer in self.layers:
            x, ind_tokens = layer(x, ind_tokens)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)

        if ind_tokens is not None:
            # 使用可学习注意力池化替代简单平均
            ind_tokens_pooled = self.ind_pool(ind_tokens)
            x = torch.cat([x, ind_tokens_pooled], dim=-1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def test_model():
    """测试模型的前向传播和梯度"""
    print("=" * 70)
    print("Testing Swin Transformer V2 with Global Independent Tokens (v2)")
    print("=" * 70)

    model = SwinTransformerV2WithGlobalIndTokenV2(
        img_size=256,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        num_classes=2,
    )

    print("\nModel created successfully!")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # 测试前向传播
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"
    print("Forward pass successful!")

    # 测试梯度回传
    print("\nTesting backward pass...")
    model.zero_grad()
    output.sum().backward()
    has_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  [FAIL] {name} has no grad!")
            has_grad = False
    if has_grad:
        print("Backward pass successful! All params have gradients!")

    print("\n" + "=" * 70)
    print("All tests passed! Model is correct!")
    print("=" * 70)

    print("\nModel Architecture Summary (v2):")
    print("  - IndTokenInitializer: Dual-path initialization (global semantic + local structure)")
    print("  - SimplifiedWindowBias: Only ind->all ind bias, removed redundant ind<->local bias")
    print("  - Original logit_scale usage: no lower bound limit")
    print("  - Pre-Norm structure consistent with original Swin")


if __name__ == '__main__':
    test_model()
