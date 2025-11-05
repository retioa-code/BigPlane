import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import supervision as sv

# 类别信息（根据你的数据集）
CATEGORIES = {
    0: "Motorcycle",
    1: "Bus",
    2: "Car",
    3: "Motorcycle",
    4: "Motorcycle_2P",
    5: "Motorcycle_Delivery",
    6: "Motorcycle_Freight",
    7: "Motorcycle_NH",
    8: "Pickup",
    9: "Semitrailer",
    10: "Small_Bus",
    11: "Small_Truck",
    12: "Trailer",
    13: "Truck",
    14: "Van",
}

NUM_CLASSES = 15

# 生成随机颜色映射
np.random.seed(42)
COLORS = np.random.randint(0, 255, (NUM_CLASSES, 3))


def load_model(checkpoint_path, device):
    """加载训练好的RF-DETR分割模型"""
    try:
        from rfdetr import RFDETRSegPreview

        # 加载checkpoint，检查类别数
        if not os.path.exists(checkpoint_path):
            print(f"✗ 权重文件不存在: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"✓ 加载checkpoint: {checkpoint_path}")

        # 从checkpoint推断类别数
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            ckpt_model = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            ckpt_model = checkpoint['state_dict']
        else:
            ckpt_model = checkpoint

        # 从class_embed权重推断类别数
        if 'class_embed.weight' in ckpt_model:
            num_classes = ckpt_model['class_embed.weight'].shape[0]
            print(f"✓ 从checkpoint推断类别数: {num_classes}")
        else:
            num_classes = 15
            print(f"⚠ 未能从checkpoint推断类别数，使用默认: {num_classes}")

        # 创建模型实例，指定类别数
        model = RFDETRSegPreview(num_classes=num_classes)
        print(f"✓ 模型初始化成功 (num_classes={num_classes})")

        # 加载权重，使用strict=False忽略形状不匹配
        try:
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.model.model.load_state_dict(checkpoint['model'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.model.model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.model.model.load_state_dict(checkpoint, strict=False)
            else:
                model.model.model.load_state_dict(checkpoint, strict=False)
            print(f"✓ 已加载模型权重")
        except Exception as load_err:
            print(f"⚠ 权重加载警告（使用strict=False）: {load_err}")

        # 设置模型为eval模式
        model.model.model.eval()
        model.model.device = device
        print(f"✓ 模型设置为eval模式，设备: {device}")

        return model

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_image(model, image_path, threshold=0.5):
    """预测单张图片"""
    try:
        # 使用RF-DETR的predict方法
        # 接受文件路径、PIL Image、numpy array 或 torch.Tensor
        detections = model.predict(
            images=str(image_path),
            threshold=threshold
        )

        return detections

    except Exception as e:
        print(f"✗ 预测失败 {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_segmentation(image_path, detections, output_dir):
    """可视化分割结果并保存（掩膜覆盖在原图上）"""
    try:
        # 读取原始图片
        image = cv2.imread(str(image_path))
        image_bgr = image.copy().astype(np.float32)
        height, width = image_bgr.shape[:2]

        # 绘制检测结果
        if detections is not None and len(detections) > 0:
            # 先绘制掩码（半透明）
            if detections.mask is not None:
                for i, mask in enumerate(detections.mask):
                    class_id = int(detections.class_id[i])
                    color_rgb = COLORS[class_id % NUM_CLASSES].astype(np.uint8)
                    color_bgr = color_rgb[::-1]  # RGB转BGR

                    # 处理mask
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()

                    # 二值化mask
                    if mask.dtype == bool:
                        mask_bin = mask.astype(np.uint8)
                    else:
                        mask_bin = (mask > 0.5).astype(np.uint8)

                    # 调整mask到图片大小
                    if mask_bin.shape != (height, width):
                        mask_bin = cv2.resize(mask_bin, (width, height), interpolation=cv2.INTER_NEAREST)

                    # 创建彩色掩码
                    mask_color = np.zeros_like(image_bgr)
                    mask_color[mask_bin > 0] = color_bgr

                    # 半透明叠加（0.4透明度）
                    image_bgr = cv2.addWeighted(image_bgr, 1.0, mask_color, 0.4, 0)

            # 再绘制轮廓和标签
            if detections.mask is not None:
                for i, mask in enumerate(detections.mask):
                    class_id = int(detections.class_id[i])
                    confidence = float(detections.confidence[i])
                    color_rgb = COLORS[class_id % NUM_CLASSES].astype(np.uint8)
                    color_bgr = tuple(map(int, color_rgb[::-1]))  # RGB转BGR，转为int元组
                    class_name = CATEGORIES.get(class_id, f"Class {class_id}")

                    # 处理mask
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()

                    mask_bin = (mask > 0.5).astype(np.uint8)
                    if mask_bin.shape != (height, width):
                        mask_bin = cv2.resize(mask_bin, (width, height), interpolation=cv2.INTER_NEAREST)

                    # 查找轮廓
                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 绘制轮廓
                    for contour in contours:
                        cv2.drawContours(image_bgr, [contour], 0, color_bgr, 2)

                    # 绘制标签（在第一个轮廓附近）
                    if contours:
                        M = cv2.moments(contours[0])
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = contours[0][0][0]

                        label = f"{class_name} {confidence:.2f}"
                        # 绘制背景框
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            image_bgr,
                            (cx - 5, cy - text_height - 10),
                            (cx + text_width + 5, cy),
                            color_bgr,
                            -1
                        )
                        # 绘制文本
                        cv2.putText(
                            image_bgr, label, (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )

        # 转回uint8并保存
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

        # 直接保存为PNG（原图尺寸）
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_seg.png")
        cv2.imwrite(output_path, image_bgr)

        print(f"✓ 已保存: {output_path}")
        return True

    except Exception as e:
        print(f"✗ 可视化失败 {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 配置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")

    checkpoint_path = "/workspace/rfdetr/outputs/checkpoint_best_total.pth"
    test_dir = "/workspace/rfdetr/toy1/test"
    output_dir = "/workspace/rfdetr/predictions"
    threshold = 0.5

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("正在加载模型...")
    # 从checkpoint自动推断类别数
    model = load_model(checkpoint_path, device)
    if model is None:
        print("模型加载失败！")
        return

    print()

    # 获取测试图片列表
    test_path = Path(test_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([f for f in test_path.iterdir()
                          if f.suffix.lower() in image_extensions])

    if not image_files:
        print(f"✗ 在 {test_dir} 中未找到图片")
        return

    print(f"找到 {len(image_files)} 张图片\n")

    # 逐张预测
    success_count = 0
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] 处理: {image_path.name}")

        detections = predict_image(model, image_path, threshold=threshold)
        if detections is not None:
            if visualize_segmentation(image_path, detections, output_dir):
                success_count += 1
                if detections.mask is not None:
                    print(f"   检测到 {len(detections)} 个目标，包含掩码")
                else:
                    print(f"   检测到 {len(detections)} 个目标")
        print()

    print(f"\n{'=' * 50}")
    print(f"预测完成！成功处理: {success_count}/{len(image_files)}")
    print(f"结果已保存到: {output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()