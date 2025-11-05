# rfdetr/datasets/coco.py
# 完整版本：包含所有原有函数 + 6通道支持

from pathlib import Path
import torch
import torch.utils.data
import torchvision
import pycocotools.mask as coco_mask
import cv2
import numpy as np
import albumentations as A
import math

import rfdetr.datasets.transforms as T


# ========== 原有的辅助函数（必须保留） ==========

def compute_multi_scale_scales(resolution, expanded_scales=False, patch_size=16, num_windows=4):
    """计算多尺度的缩放尺寸"""
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]
    return proposed_scales


def convert_coco_poly_to_mask(segmentations, height, width):
    """将COCO多边形转换为二值掩码张量 [N, H, W]"""
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


# ========== 原有的COCO检测类（保留不改） ==========

class CocoDetection(torchvision.datasets.CocoDetection):
    """标准3通道COCO检测"""

    def __init__(self, img_folder, ann_file, transforms, include_masks=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertCoco(include_masks=include_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


# ========== 新增：6通道变化检测类 ==========
# 重点：确保 ChangeDetectionCOCO 返回6通道张量

class ChangeDetectionCOCO(torchvision.datasets.CocoDetection):
    """
    6通道变化检测数据集（T1前时相 + T2后时相）

    数据结构:
    dataset_root/
    ├── train/
    │   ├── T1/  (前时相图像)
    │   ├── T2/  (后时相图像)
    │   └── _annotations.coco.json
    ├── val/
    │   ├── T1/
    │   ├── T2/
    │   └── _annotations.coco.json
    """

    def __init__(self, root_dir, split='train', transforms=None, include_masks=False, resolution=384):
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        self.t1_dir = self.split_dir / 'T1'
        self.t2_dir = self.split_dir / 'T2'
        self.resolution = resolution
        self.include_masks = include_masks

        # 使用COCO加载标注
        ann_file = str(self.split_dir / '_annotations.coco.json')
        # ✓ 重要：传入split_dir作为img_folder，但实际加载时会从T1/T2读取
        super().__init__(str(self.split_dir), ann_file)

        self._transforms = transforms
        self.prepare = ConvertCocoChangeDetection(include_masks=include_masks)

        # 几何增强（仅训练时）
        if split == 'train':
            self.geometric_aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.RandomCrop(height=resolution, width=resolution, p=0.7),
                A.Resize(height=resolution, width=resolution, p=1.0),
            ], bbox_params=A.BboxParams(
                format='coco',
                label_fields=['class_labels'],
                min_area=1.0,
                min_visibility=0.3
            ))
            # 光度增强（独立应用）
            self.photometric_aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(p=0.3),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ])
        else:
            # 推理时仅resize
            self.geometric_aug = A.Compose([
                A.Resize(height=resolution, width=resolution, p=1.0),
            ], bbox_params=A.BboxParams(
                format='coco',
                label_fields=['class_labels'],
                min_area=1.0,
                min_visibility=0.3
            ))
            self.photometric_aug = None

        # ✓ 标准化（6通道）
        self.normalize = T.Normalize(
            [0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
        )

    def __getitem__(self, idx):
        """返回6通道图像和目标"""

        image_id = self.ids[idx]
        img_info = self.coco.imgs[image_id]
        img_name = img_info['file_name']

        # 加载T1和T2
        img1_path = self.t1_dir / img_name
        img2_path = self.t2_dir / img_name

        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        if img1 is None or img2 is None:
            raise FileNotFoundError(f"图像不存在: {img1_path} 或 {img2_path}")

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # 加载标注
        target = {
            'image_id': image_id,
            'annotations': self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        }

        # 提取bbox和类别
        bboxes, class_labels, anno_list = self._extract_bboxes(target['annotations'])

        # 几何增强
        if len(bboxes) > 0:
            aug1 = self.geometric_aug(image=img1, bboxes=bboxes, class_labels=class_labels)
            img1 = aug1['image']
            bboxes = aug1['bboxes']
            class_labels = aug1['class_labels']

            aug2 = self.geometric_aug(image=img2, bboxes=bboxes, class_labels=class_labels)
            img2 = aug2['image']
        else:
            aug1 = self.geometric_aug(image=img1, bboxes=[], class_labels=[])
            img1 = aug1['image']

            aug2 = self.geometric_aug(image=img2, bboxes=[], class_labels=[])
            img2 = aug2['image']

        # # 光度增强
        # if self.photometric_aug is not None:
        #     img1 = self.photometric_aug(image=img1)['image']
        #     img2 = self.photometric_aug(image=img2)['image']

        # 拼接为6通道
        img6 = np.concatenate([img1, img2], axis=2)

        # 转换为torch张量
        if img6.dtype != np.uint8:
            img6 = (img6 * 255).astype(np.uint8) if img6.max() <= 1 else img6.astype(np.uint8)

        img6_tensor = torch.from_numpy(img6).permute(2, 0, 1).float() / 255.0

        # 应用标准化
        img6_tensor = self.normalize(img6_tensor)

        converted_boxes = []
        H, W = img1.shape[:2]
        if len(bboxes) > 0:
            # bboxes 是一个列表，内部元素是 [x, y, w, h] (float)
            for bbox in bboxes:
                x_min, y_min, w_box, h_box = bbox
                # 1. 转换为 cxcywh 格式 (中心点 + 宽高)
                c_x = x_min + w_box / 2
                c_y = y_min + h_box / 2
                # 2. 归一化 (除以增强后的图像尺寸 W, H)
                c_x_norm = c_x / W
                c_y_norm = c_y / H
                w_norm = w_box / W
                h_norm = h_box / H
                # DETR 模型要求归一化后的中心点和宽高
                converted_boxes.append([c_x_norm, c_y_norm, w_norm, h_norm])
        # 准备目标字典
        target_dict = {
            'image_id': torch.tensor([image_id]),
            'boxes': torch.tensor(converted_boxes, dtype=torch.float32) if converted_boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(class_labels, dtype=torch.int64) if class_labels else torch.zeros((0,),
                                                                                                     dtype=torch.int64),
            'size': torch.tensor(img1.shape[:2], dtype=torch.int32),
            'orig_size': torch.tensor(img1.shape[:2], dtype=torch.int32),
        }

        # 分割掩码
        if self.include_masks and len(anno_list) > 0:
            h, w = img1.shape[:2]
            segmentations = [ann.get('segmentation', []) for ann in anno_list]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            if masks.numel() > 0:
                target_dict['masks'] = masks
            else:
                target_dict['masks'] = torch.zeros((0, h, w), dtype=torch.uint8)

        return img6_tensor[0], target_dict

    @staticmethod
    def _extract_bboxes(annotations):
        """从COCO标注中提取bbox"""
        bboxes = []
        class_labels = []
        anno_list = []

        for ann in annotations:
            if ann.get('iscrowd', 0) == 0:
                bboxes.append(ann['bbox'])
                class_labels.append(ann['category_id'])
                anno_list.append(ann)

        return bboxes, class_labels, anno_list


# ========== 转换类 ==========

class ConvertCoco(object):
    """转换COCO格式标注"""

    def __init__(self, include_masks=False):
        self.include_masks = include_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        if self.include_masks:
            if len(anno) > 0 and 'segmentation' in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                if masks.numel() > 0:
                    target["masks"] = masks[keep]
                else:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            target["masks"] = target["masks"].bool()

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class ConvertCocoChangeDetection(object):
    """转换COCO格式标注（6通道变化检测）"""

    def __init__(self, include_masks=False):
        self.include_masks = include_masks

    def __call__(self, image, target):
        """保持兼容性 - 数据已在ChangeDetectionCOCO.__getitem__中处理"""
        # 此处image已是6通道tensor，target已经过处理
        return image, target


# ========== 变换函数 ==========

def make_coco_transforms(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False,
                         patch_size=16, num_windows=4):
    """原有的3通道变换"""
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [resolution]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([resolution], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_square_div_64(image_set, resolution, multi_scale=False, expanded_scales=False,
                                       skip_random_resize=False, patch_size=16, num_windows=4):
    """原有的3通道变换（方形resize）"""
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [resolution]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.SquareResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.SquareResize(scales),
                ]),
            ),
            normalize,
        ])

    if image_set == 'val' or  image_set == 'test':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# ========== 数据集构建函数 ==========

def build(image_set, args, resolution):
    """原有的COCO数据集构建函数"""
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'image_info_test-dev2017.json'),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ))
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ))
    return dataset


def build_roboflow(image_set, args, resolution):
    """Roboflow格式COCO数据集"""
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root / "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    try:
        include_masks = args.segmentation_head
    except:
        include_masks = False

    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ), include_masks=include_masks)
    return dataset


def build_change_detection(image_set, args, resolution):
    """6通道变化检测数据集

    Args:
        image_set: 'train', 'val', 'test'
        args: 配置参数，需要有 dataset_dir 和 segmentation_head
        resolution: 输入分辨率

    Returns:
        ChangeDetectionCOCO dataset
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f'数据集路径不存在: {root}'

    try:
        include_masks = args.segmentation_head
    except:
        include_masks = False

    # ✓ 关键：传入 transforms=None
    # 因为 ChangeDetectionCOCO 内部已处理所有数据增强
    dataset = ChangeDetectionCOCO(
        root_dir=root,
        split=image_set.split('_')[0],  # 'train' from 'train_val', etc.
        transforms=None,
        include_masks=include_masks,
        resolution=resolution
    )

    return dataset