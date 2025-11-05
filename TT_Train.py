# TT_Train.py
# 遥感变化检测（6通道）实例分割模型训练 - 修复版本

import torch
from rfdetr import RFDETRSegPreview
import os
import sys
from pathlib import Path

# 自动检测设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ========== 配置参数 ==========
config = {
    # 数据集参数
    'dataset_dir': '/workspace/rfdetr/toy1CD',  # 修改为你的数据集路径
    'dataset_file': 'change_detection',  # ✓ 关键：指定为变化检测数据集
    'output_dir': '/workspace/rfdetr/outputsCD',
    'device': device,

    # 训练参数
    'epochs': 5,
    'batch_size': 4,  # 根据显存调整：RTX4090用16-32，RTX3090用8-16，小显卡用2-4
    'grad_accum_steps': 2,  # 梯度累积步数（有效batch_size = batch_size * grad_accum_steps）
    'lr': 1e-4,
    'lr_encoder': 1.5e-4,
    'weight_decay': 1e-4,

    # 模型参数 - 6通道DINOv2
    'encoder': 'dinov2_windowed_small',  # ✓ 使用windowed attention
    'patch_size': 12,  # ✓ 与backbone适配
    'num_windows': 2,  # ✓ windowed attention的窗口数
    'positional_encoding_size': 36,  # 根据分辨率计算：resolution / patch_size

    # 特征和分割参数
    'out_feature_indexes': [3, 6, 9, 12],
    'segmentation_head': True,  # ✓ 启用分割头支持6通道

    # 分辨率和增强
    'resolution': 384,  # 或 512/576
    'multi_scale': False,
    'square_resize_div_64': True,

    # 其他参数
    'num_classes': 5,  # 根据你的数据集调整
}

# 创建输出目录
os.makedirs(config['output_dir'], exist_ok=True)

print("=" * 60)
print("RF-DETR 6通道变化检测训练")
print("=" * 60)
print(f"数据集: {config['dataset_dir']}")
print(f"输出目录: {config['output_dir']}")
print(f"Batch Size: {config['batch_size']}")
print(f"梯度累积步数: {config['grad_accum_steps']}")
print(f"有效Batch Size: {config['batch_size'] * config['grad_accum_steps']}")
print(f"Epochs: {config['epochs']}")
print(f"分割头: {'启用' if config['segmentation_head'] else '禁用'}")
print(f"数据集类型: {config['dataset_file']}")
print(f"Backbone: {config['encoder']}")
print("=" * 60)

# ========== 加载模型并训练 ==========
try:
    # 加载预训练的RF-DETR Seg Preview模型
    model = RFDETRSegPreview()
    print("✓ 模型加载成功")

    # 开始训练
    # ✓ 重要：现在dataset_file='change_detection'会调用build_change_detection()
    #   该函数会加载ChangeDetectionCOCO类，返回6通道图像
    model.train(
        dataset_dir=config['dataset_dir'],
        dataset_file=config['dataset_file'],  # ✓ 指定数据集类型
        output_dir=config['output_dir'],
        device=config['device'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        grad_accum_steps=config['grad_accum_steps'],
        lr=config['lr'],
        lr_encoder=config['lr_encoder'],
        weight_decay=config['weight_decay'],
        encoder=config['encoder'],
        patch_size=config['patch_size'],
        num_windows=config['num_windows'],
        positional_encoding_size=config['positional_encoding_size'],
        segmentation_head=config['segmentation_head'],
        multi_scale=config['multi_scale'],
        resolution=config['resolution'],
        num_classes=config['num_classes'],
        out_feature_indexes=config['out_feature_indexes'],
        square_resize_div_64=config['square_resize_div_64'],
    )

    print("✓ 训练完成！")

except Exception as e:
    print(f"✗ 训练出错: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)