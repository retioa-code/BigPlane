# rfdetr/datasets/__init__.py
# 数据集工厂函数 - 支持3通道和6通道

from .coco import build, build_roboflow, build_change_detection
from pycocotools.coco import COCO


def build_dataset(image_set, args, resolution):
    """构建数据集

    Args:
        image_set: 'train', 'val', 'test'
        args: 配置参数
        resolution: 输入分辨率

    Returns:
        Dataset object
    """

    # 获取数据集类型
    dataset_file = getattr(args, 'dataset_file', 'coco')

    if dataset_file == 'coco':
        return build(image_set, args, resolution)

    elif dataset_file == 'roboflow':
        return build_roboflow(image_set, args, resolution)

    elif dataset_file == 'change_detection':
        # ✓ 6通道变化检测数据集
        return build_change_detection(image_set, args, resolution)

    else:
        raise ValueError(f'dataset {dataset_file} not supported')


def get_coco_api_from_dataset(dataset):
    """从数据集中获取COCO API对象

    Args:
        dataset: Dataset object (CocoDetection, ChangeDetectionCOCO等)

    Returns:
        COCO object for evaluation
    """
    if isinstance(dataset, (ChangeDetectionCOCO,)):
        # 6通道变化检测数据集
        return dataset.coco
    elif hasattr(dataset, 'coco'):
        # 标准COCO数据集（CocoDetection等）
        return dataset.coco
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")


# 导入ChangeDetectionCOCO以便类型检查
from .coco import ChangeDetectionCOCO

__all__ = [
    'build_dataset',
    'get_coco_api_from_dataset',
    'build',
    'build_roboflow',
    'build_change_detection',
]