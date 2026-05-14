# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

from mmdet.models.builder import DETECTORS

from .meta_rcnn import MetaRCNN
from .vfa_utils import VFATestMixin


@DETECTORS.register_module()
class VFA(MetaRCNN, VFATestMixin):
    """VFA detector built on top of Meta R-CNN."""

    def __init__(self, *args, with_refine: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The CLIP-based refinement is used by the original VFA COCO configs.
        self.with_refine = with_refine

    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training."""
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)

        # VFA stops the gradient at RPN and trains ROI features separately.
        query_feats_rpn = [x.detach() for x in query_feats]
        query_feats_rcnn = query_feats
        support_feats = self.extract_support_feat(support_img)

        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats_rpn,
                    support_feats,
                    query_img_metas=query_data['img_metas'],
                    query_gt_bboxes=query_data['gt_bboxes'],
                    query_gt_labels=None,
                    query_gt_bboxes_ignore=query_data.get(
                        'gt_bboxes_ignore', None),
                    support_img_metas=support_data['img_metas'],
                    support_gt_bboxes=support_data['gt_bboxes'],
                    support_gt_labels=support_data['gt_labels'],
                    support_gt_bboxes_ignore=support_data.get(
                        'gt_bboxes_ignore', None),
                    proposal_cfg=proposal_cfg)
            else:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats_rpn,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            query_feats_rcnn,
            support_feats,
            proposals=proposal_list,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        bbox_results = super().simple_test(img, img_metas, proposals, rescale)
        if self.with_refine:
            return self.refine_test(bbox_results, img_metas)
        return bbox_results
