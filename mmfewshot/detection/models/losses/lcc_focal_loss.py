import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.cross_entropy_loss import cross_entropy


class MultiCEFocalLoss(torch.nn.Module):

    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        if alpha is None:
            alpha = torch.ones(class_num, 1)
        else:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha.ndim == 0:
                alpha = alpha.repeat(class_num).view(class_num, 1)
            elif alpha.ndim == 1:
                alpha = alpha.view(class_num, 1)
        self.register_buffer('alpha', alpha.float())
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.class_num).type_as(predict)
        ids = target.view(-1, 1)
        alpha = self.alpha.to(predict.device)[ids.data.view(-1)]
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


@LOSSES.register_module()
class LCCFocalLoss(nn.Module):

    def __init__(self,
                 num_base_classes,
                 num_novel_classes,
                 base_label_ids=None,
                 novel_label_ids=None,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        super().__init__()
        self.num_base_classes = num_base_classes
        self.num_novel_classes = num_novel_classes
        self.num_classes = self.num_base_classes + self.num_novel_classes + 1
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

        if base_label_ids is None:
            self.base_label_ids = list(range(num_base_classes))
        else:
            self.base_label_ids = list(base_label_ids)
        if novel_label_ids is None:
            self.novel_label_ids = list(
                range(num_base_classes, num_base_classes + num_novel_classes))
        else:
            self.novel_label_ids = list(novel_label_ids)

        self.alpha = alpha

        self.cls_criterion = cross_entropy
        self.cls_criterion1 = MultiCEFocalLoss(
            num_novel_classes + 1, reduction=self.reduction)

        self._label_maps_built = False

    def _build_label_maps(self, device):
        """Build label remapping tensors on the given device."""
        if (self._label_maps_built and hasattr(self, '_stage0_map')
                and self._stage0_map.device == device):
            return
        label_ids = self.base_label_ids + self.novel_label_ids
        max_label = max(label_ids)
        # stage0: map each label to [0, num_base_classes]
        # base label -> its position in base_label_ids; novel/bg -> num_base_classes
        self._stage0_map = torch.full(
            (max_label + 1,), self.num_base_classes, dtype=torch.long, device=device)
        for i, lbl in enumerate(self.base_label_ids):
            self._stage0_map[lbl] = i

        # stage1: identify novel labels and map to [0, num_novel-1]
        self._is_novel = torch.zeros(
            max_label + 1, dtype=torch.bool, device=device)
        self._novel_map = torch.zeros(
            max_label + 1, dtype=torch.long, device=device)
        for i, lbl in enumerate(self.novel_label_ids):
            self._is_novel[lbl] = True
            self._novel_map[lbl] = i

        self._label_maps_built = True

    def forward(self,
                predict,
                predict_novel,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        self._build_label_maps(target.device)

        bg_label = self.num_base_classes + self.num_novel_classes

        # stage0: map targets to [0, num_base_classes]
        # base labels -> their position in base_label_ids
        # novel/bg labels -> num_base_classes (the "other" bin)
        valid_mask = target < len(self._stage0_map)
        label_stage0 = target.clone()
        label_stage0[valid_mask] = self._stage0_map[target[valid_mask]]
        label_stage0 = torch.clamp(label_stage0, 0, self.num_base_classes)
        loss_cls_stage0 = self.loss_weight * self.cls_criterion(
            predict,
            label_stage0,
            weight,
            class_weight=None,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=None,
            avg_non_ignore=False,
            **kwargs)

        # stage1: novel targets -> their novel index; BG targets -> novel_bg
        valid_mask = target < len(self._is_novel)
        is_novel_target = torch.zeros_like(target, dtype=torch.bool)
        is_novel_target[valid_mask] = self._is_novel[target[valid_mask]]

        # include novel targets and BG targets (BG mapped to num_novel_classes)
        is_stage1 = is_novel_target | (target == bg_label)

        cls_score_stage1 = predict_novel[is_stage1]
        if cls_score_stage1.numel() > 0:
            stage1_targets = target[is_stage1]
            # default: novel_bg (for BG samples)
            label_stage1 = torch.full_like(
                stage1_targets, self.num_novel_classes)
            # novel targets: map to their internal novel index
            novel_in_stage1 = is_novel_target[is_stage1]
            if novel_in_stage1.any():
                label_stage1[novel_in_stage1] = self._novel_map[
                    stage1_targets[novel_in_stage1]]
            loss_cls_stage1 = self.loss_weight * \
                self.cls_criterion1(cls_score_stage1, label_stage1)
        else:
            loss_cls_stage1 = predict_novel.sum() * 0.0

        return dict(loss_cls_s0=loss_cls_stage0, loss_cls_s1=loss_cls_stage1)
