# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, batch_sigmoid_ce_loss, batch_dice_loss
from rfdetr.models.segmentation_head import point_sample


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25, use_pos_only: bool = False,
                 use_position_modulated_cost: bool = False, mask_point_sample_ratio: int = 16, cost_mask_ce: float = 1, cost_mask_dice: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 "pred_masks": Tensor of dim [batch_size, num_queries, H, W] with predicted mask logits (if segmentation_head=True)
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes]
                 "boxes": Tensor of dim [num_target_boxes, 4]
                 "masks": Tensor of dim [num_target_boxes, H, W] (if segmentation_head=True)
            group_detr: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        masks_present = "masks" in targets[0]

        # 固定随机种子，用于采样点坐标
        generator = torch.Generator(device=outputs["pred_logits"].device)
        generator.manual_seed(42)

        all_indices = []

        # 按批次处理，避免不同批次目标数量不同导致的维度问题
        for batch_idx in range(bs):
            # 获取单个批次的预测和目标
            pred_logits = outputs["pred_logits"][batch_idx]  # [num_queries, num_classes]
            pred_boxes = outputs["pred_boxes"][batch_idx]  # [num_queries, 4]

            tgt_labels = targets[batch_idx]["labels"]  # [num_targets]
            tgt_boxes = targets[batch_idx]["boxes"]  # [num_targets, 4]

            # 处理掩码
            if masks_present:
                pred_masks = outputs["pred_masks"][batch_idx]  # [num_queries, H, W]
                tgt_masks = targets[batch_idx]["masks"]  # [num_targets, H, W]

            num_targets = len(tgt_labels)

            # 计算分类成本
            out_prob = pred_logits.sigmoid()  # [num_queries, num_classes]

            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-F.logsigmoid(-pred_logits))
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(pred_logits))
            cost_class = pos_cost_class[:, tgt_labels] - neg_cost_class[:, tgt_labels]  # [num_queries, num_targets]

            # 计算框成本
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(tgt_boxes)
            )  # [num_queries, num_targets]
            cost_giou = -giou

            # 计算L1框成本
            cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)  # [num_queries, num_targets]

            # 初始化成本矩阵
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

            # 计算掩码成本（如果存在）
            if masks_present and num_targets > 0:
                num_points = pred_masks.shape[-2] * pred_masks.shape[-1] // self.mask_point_sample_ratio

                tgt_masks_dtype = tgt_masks.to(pred_masks.dtype)

                # 为这个批次生成固定的采样点
                point_coords = torch.rand(1, num_points, 2, device=pred_masks.device, generator=generator)

                # 逐个采样预测掩码（避免批大小不匹配）
                pred_masks_logits_list = []
                for q_idx in range(num_queries):
                    pred_mask_q = pred_masks[q_idx:q_idx + 1].unsqueeze(1)  # [1, 1, H, W]
                    sampled = point_sample(
                        pred_mask_q,
                        point_coords,  # [1, num_points, 2]
                        align_corners=False
                    ).squeeze(0).squeeze(0)  # [num_points]
                    pred_masks_logits_list.append(sampled)
                pred_masks_logits = torch.stack(pred_masks_logits_list, dim=0)  # [num_queries, num_points]

                # 逐个采样目标掩码（避免批大小不匹配）
                tgt_masks_flat_list = []
                for t_idx in range(num_targets):
                    tgt_mask_t = tgt_masks_dtype[t_idx:t_idx + 1].unsqueeze(1)  # [1, 1, H, W]
                    sampled = point_sample(
                        tgt_mask_t,
                        point_coords,  # [1, num_points, 2]
                        align_corners=False,
                        mode="nearest"
                    ).squeeze(0).squeeze(0)  # [num_points]
                    tgt_masks_flat_list.append(sampled)
                tgt_masks_flat = torch.stack(tgt_masks_flat_list, dim=0)  # [num_targets, num_points]

                # 计算掩码损失成本
                cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)  # [num_queries, num_targets]
                cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)  # [num_queries, num_targets]

                # 添加掩码成本到总成本
                C = C + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice

            # 处理 group_detr（多组查询）
            C = C.float().cpu()  # [num_queries, num_targets]

            # 处理 NaN 和 Inf
            max_cost = C.max() if C.numel() > 0 else 0
            C[C.isinf() | C.isnan()] = max_cost * 2

            # 执行 Hungarian 匹配
            if num_targets == 0:
                # 没有目标的情况
                indices_batch = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
            else:
                # 处理 group_detr
                if group_detr == 1:
                    # 标准情况：单个查询组
                    i, j = linear_sum_assignment(C.numpy())
                    indices_batch = (i, j)
                else:
                    # 多查询组情况
                    g_num_queries = num_queries // group_detr
                    indices_list = []

                    for g_i in range(group_detr):
                        start_idx = g_i * g_num_queries
                        if g_i == group_detr - 1:
                            # 最后一组可能包含剩余查询
                            end_idx = num_queries
                        else:
                            end_idx = start_idx + g_num_queries

                        C_g = C[start_idx:end_idx, :]  # [g_num_queries, num_targets]
                        i_g, j_g = linear_sum_assignment(C_g.numpy())

                        # 调整预测索引偏移
                        i_g = i_g + start_idx
                        indices_list.append((i_g, j_g))

                    # 合并所有组的匹配结果
                    all_i = np.concatenate([idx[0] for idx in indices_list])
                    all_j = np.concatenate([idx[1] for idx in indices_list])
                    indices_batch = (all_i, all_j)

            all_indices.append(indices_batch)

        # 转换为张量
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in all_indices]


def build_matcher(args):
    if args.segmentation_head:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,)
    else:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
        )