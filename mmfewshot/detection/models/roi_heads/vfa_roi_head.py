# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from torch import Tensor

from .meta_rcnn_roi_head import MetaRCNNRoIHead


class VAE(nn.Module):

    def __init__(self, in_channels: int, latent_dim: int,
                 hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_channels), nn.BatchNorm1d(in_channels),
            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        z = self.decoder_input(z)
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tuple[Tensor,
                                                                  Tensor]:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu, std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z, z_inv = self.reparameterize(mu, log_var)
        z_out = self.decode(z)
        return [z_out, z_inv, input, mu, log_var]

    def loss_function(self,
                      input: Tensor,
                      rec: Tensor,
                      mu: Tensor,
                      log_var: Tensor,
                      kld_weight: float = 0.00025) -> dict:
        recons_loss = F.mse_loss(rec, input)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0)
        return {'loss_vae': recons_loss + kld_weight * kld_loss}


@HEADS.register_module()
class VFARoIHead(MetaRCNNRoIHead):
    """VFA RoI head with VAE-transformed support features."""

    def __init__(self, vae_dim: int = 2048, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vae = VAE(vae_dim, vae_dim, vae_dim)

    def _bbox_forward_train(self, query_feats: List[Tensor],
                            support_feats: List[Tensor],
                            sampling_results: object,
                            query_img_metas: List[Dict],
                            query_gt_bboxes: List[Tensor],
                            query_gt_labels: List[Tensor],
                            support_gt_labels: List[Tensor]) -> Dict:
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        support_feat = self.extract_support_feats(support_feats)[0]
        support_feat_rec, support_feat_inv, _, mu, log_var = self.vae(
            support_feat)

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  query_gt_bboxes,
                                                  query_gt_labels,
                                                  self.train_cfg)
        labels, label_weights, bbox_targets, bbox_weights = bbox_targets
        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': []}
        batch_size = len(query_img_metas)
        num_sample_per_image = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_image
            end = (img_id + 1) * num_sample_per_image
            random_index = np.random.choice(range(len(support_gt_labels)))
            random_query_label = support_gt_labels[random_index]
            for i in range(support_feat.size(0)):
                if support_gt_labels[i] == random_query_label:
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat_inv[i].sigmoid().unsqueeze(0))
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], bbox_results['bbox_pred'],
                        query_rois[start:end], labels[start:end],
                        label_weights[start:end], bbox_targets[start:end],
                        bbox_weights[start:end])
                    for key in single_loss_bbox.keys():
                        loss_bbox[key].append(single_loss_bbox[key])

        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc':
                    loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                else:
                    loss_bbox[key] = torch.stack(
                        loss_bbox[key]).sum() / batch_size

        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_rec)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        loss_bbox.update(
            self.vae.loss_function(support_feat, support_feat_rec, mu,
                                   log_var))
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1), query_roi_feats)
        return dict(cls_score=cls_score, bbox_pred=bbox_pred)

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)
        rois = bbox2roi(proposals)

        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]
            _, support_feat_inv, _, _, _ = self.vae(support_feat)
            bbox_results = self._bbox_forward(query_roi_feats,
                                              support_feat_inv.sigmoid())
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]
        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())
        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
