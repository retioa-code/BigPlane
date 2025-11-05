# models/backbone/dinov2.py
# RF-DETR DINOv2 Backbone with 6-channel support

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoBackbone
import types
import math
import json
import os
import warnings

from .dinov2_with_windowed_attn import WindowedDinov2WithRegistersConfig, WindowedDinov2WithRegistersBackbone

size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}

size_to_config = {
    "small": "dinov2_small.json",
    "base": "dinov2_base.json",
    "large": "dinov2_large.json",
}

size_to_config_with_registers = {
    "small": "dinov2_with_registers_small.json",
    "base": "dinov2_with_registers_base.json",
    "large": "dinov2_with_registers_large.json",
}


def get_config(size, use_registers):
    config_dict = size_to_config_with_registers if use_registers else size_to_config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(current_dir, "dinov2_configs")
    config_path = os.path.join(configs_dir, config_dict[size])
    with open(config_path, "r") as f:
        dino_config = json.load(f)
    return dino_config


class DinoV2(nn.Module):
    """DINOv2 Backbone with 6-channel input support for change detection"""

    def __init__(self,
                 shape=(640, 640),
                 out_feature_indexes=[2, 4, 5, 9],
                 size="base",
                 use_registers=True,
                 use_windowed_attn=True,
                 gradient_checkpointing=False,
                 load_dinov2_weights=True,
                 patch_size=14,
                 num_windows=4,
                 positional_encoding_size=37,
                 in_channels=6,  # ✨ 新增：支持6通道输入
                 ):
        super().__init__()

        self.in_channels = in_channels
        name = f"facebook/dinov2-with-registers-{size}" if use_registers else f"facebook/dinov2-{size}"

        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows

        # 创建encoder
        if not use_windowed_attn:
            assert not gradient_checkpointing, "非windowed attention不支持梯度检查点"
            assert load_dinov2_weights, "非windowed attention需要从hub加载预训练权重"
            self.encoder = AutoBackbone.from_pretrained(
                name,
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
            )
        else:
            window_block_indexes = set(range(out_feature_indexes[-1] + 1))
            window_block_indexes.difference_update(out_feature_indexes)
            window_block_indexes = list(window_block_indexes)

            dino_config = get_config(size, use_registers)
            dino_config["return_dict"] = False
            dino_config["out_features"] = [f"stage{i}" for i in out_feature_indexes]

            implied_resolution = positional_encoding_size * patch_size

            if implied_resolution != dino_config["image_size"]:
                print(f"使用不同的位置编码数量，不会加载DINOv2权重")
                dino_config["image_size"] = implied_resolution
                load_dinov2_weights = False

            if patch_size != 14:
                print(f"使用patch_size={patch_size}而非14，不会加载DINOv2权重")
                dino_config["patch_size"] = patch_size
                load_dinov2_weights = False

            if use_registers:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    num_register_tokens=0,
                    gradient_checkpointing=gradient_checkpointing,
                )

            self.encoder = WindowedDinov2WithRegistersBackbone.from_pretrained(
                name,
                config=windowed_dino_config,
            ) if load_dinov2_weights else WindowedDinov2WithRegistersBackbone(windowed_dino_config)

        # ========== 6通道适配 ==========
        if in_channels != 3:
            self._adapt_to_6channel()

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def _adapt_to_6channel(self):
        """扩展第一层卷积从3通道到6通道"""
        # 定位第一层卷积（不同模型结构可能不同）
        if hasattr(self.encoder, 'embeddings'):
            # Huggingface Backbone结构
            if hasattr(self.encoder.embeddings, 'patch_embeddings'):
                conv1 = self.encoder.embeddings.patch_embeddings.projection
            elif hasattr(self.encoder.embeddings, 'convolution'):
                conv1 = self.encoder.embeddings.convolution
            else:
                warnings.warn("未找到第一层卷积，跳过6通道适配")
                return
        else:
            warnings.warn("Backbone结构不明确，跳过6通道适配")
            return

        # 检查是否为3通道
        if conv1.in_channels != 3:
            warnings.warn(f"第一层卷积已非3通道({conv1.in_channels})，跳过")
            return

        # 扩展卷积
        new_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None
        )

        # ========== 权重初始化：复制+缩放 ==========
        with torch.no_grad():
            original_weight = conv1.weight.data  # (out_ch, 3, kh, kw)

            # 复制权重到6通道
            expanded_weight = original_weight.repeat(1, 2, 1, 1)  # (out_ch, 6, kh, kw)
            # expanded_weight = expanded_weight / 2.0  # 缩放保持响应不变
            expanded_weight = expanded_weight   # 缩放保持响应不变

            new_conv.weight.data = expanded_weight

            if conv1.bias is not None:
                new_conv.bias.data = conv1.bias.data.clone()

        # 替换卷积层
        if hasattr(self.encoder.embeddings, 'patch_embeddings'):
            self.encoder.embeddings.patch_embeddings.projection = new_conv
        elif hasattr(self.encoder.embeddings, 'convolution'):
            self.encoder.embeddings.convolution = new_conv

        print(f"✓ 第一层卷积已适配: 3ch -> 6ch")

    def export(self):
        """导出模型（插值位置编码）"""
        if self._export:
            return
        self._export = True

        shape = self.shape

        def make_new_interpolated_pos_encoding(position_embeddings, patch_size, height, width):
            num_positions = position_embeddings.shape[1] - 1
            dim = position_embeddings.shape[-1]
            height = height // patch_size
            width = width // patch_size

            class_pos_embed = position_embeddings[:, 0]
            patch_pos_embed = position_embeddings[:, 1:]

            patch_pos_embed = patch_pos_embed.reshape(
                1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
            )
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        with torch.no_grad():
            new_positions = make_new_interpolated_pos_encoding(
                self.encoder.embeddings.position_embeddings,
                self.encoder.config.patch_size,
                shape[0],
                shape[1],
            )

        old_interpolate_pos_encoding = self.encoder.embeddings.interpolate_pos_encoding

        def new_interpolate_pos_encoding(self_mod, embeddings, height, width):
            num_patches = embeddings.shape[1] - 1
            num_positions = self_mod.position_embeddings.shape[1] - 1
            if num_patches == num_positions and height == width:
                return self_mod.position_embeddings
            return old_interpolate_pos_encoding(embeddings, height, width)

        self.encoder.embeddings.position_embeddings = nn.Parameter(new_positions)
        self.encoder.embeddings.interpolate_pos_encoding = types.MethodType(
            new_interpolate_pos_encoding,
            self.encoder.embeddings
        )

    def forward(self, x):
        """前向传播"""
        assert x.shape[1] == self.in_channels, f"期望{self.in_channels}通道输入，得到{x.shape[1]}"

        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, \
            f"输入必须能被{block_size}整除，得到{x.shape}"

        x = self.encoder(x)
        return list(x[0])


if __name__ == "__main__":
    # 测试6通道输入
    model = DinoV2(in_channels=6)
    x = torch.randn(1, 6, 640, 640)  # 6通道输入
    output = model(x)
    print(f"✓ 6通道输入测试通过")
    print(f"输出数量: {len(output)}")
    for i, feat in enumerate(output):
        print(f"  Feature {i}: {feat.shape}")