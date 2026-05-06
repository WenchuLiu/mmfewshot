import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from mmdet.models.losses import accuracy
from mmcv.ops.nms import batched_nms


@HEADS.register_module()
class IncreaseBBoxHead(ConvFCBBoxHead):

    def __init__(self,
                 fc_out_channels=1024,
                 scale=20.,
                 base_cpt=None,
                 base_alpha=0.5,
                 novel_class_ids=None,  # <--- [修改1] 新增参数：新类ID列表
                 *args,
                 **kwargs):
        super(IncreaseBBoxHead,
              self).__init__(*args,
                             **kwargs)
        # 清理父类中不必要的层，因为我们会有自定义的分支
        if hasattr(self, 'shared_fcs'):
            del self.shared_fcs
        if hasattr(self, 'cls_fcs'):
            del self.cls_fcs
        if hasattr(self, 'reg_fcs'):
            del self.reg_fcs
            
        assert base_cpt is not None
        self.base_cpt = base_cpt
        self.base_alpha = base_alpha

        # --- [修改2]：逻辑区分新旧类 ---
        if novel_class_ids is not None:
            if isinstance(novel_class_ids, int):
                novel_class_ids = [novel_class_ids]
            self.novel_class_ids = set(novel_class_ids)
            
            # 自动计算旧类 ID (Base IDs)
            # 假设 ID 范围是 0 到 num_classes-1 (不含背景)
            all_indices = set(range(self.num_classes))
            self.base_class_ids = sorted(list(all_indices - self.novel_class_ids))
            
            self.num_base = len(self.base_class_ids)
            self.num_novel = len(self.novel_class_ids)
            
            print(f'[IncreaseBBoxHead] Configured Novel IDs: {sorted(list(self.novel_class_ids))}')
            print(f'[IncreaseBBoxHead] Configured Base IDs: {self.base_class_ids}')
        else:
            # 默认回退逻辑：后 1/4 为新类
            self.num_base = self.num_classes // 4 * 3
            self.num_novel = self.num_classes // 4
            self.base_class_ids = list(range(self.num_base))
            self.novel_class_ids = set(range(self.num_base, self.num_classes))
            print(f'[IncreaseBBoxHead] Using Default Split (3:1). Base: {self.num_base}, Novel: {self.num_novel}')

        # 构建 Base 分支
        base_shared_fcs = nn.ModuleList()
        # 假设输入特征展平后是 49 * 256 (7x7x256)
        base_shared_fcs.append(nn.Linear(49 * 256, fc_out_channels))
        base_shared_fcs.append(nn.Linear(fc_out_channels, fc_out_channels))
        self.base_shared_fcs = base_shared_fcs
        
        # 构建 Novel 分支
        novel_shared_fcs = nn.ModuleList()
        novel_shared_fcs.append(nn.Linear(49 * 256, fc_out_channels))
        novel_shared_fcs.append(nn.Linear(fc_out_channels, fc_out_channels))
        self.novel_shared_fcs = novel_shared_fcs

        # 重建分类层
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)
        
        # 重建回归层 (父类中如果删除了，这里需要重新定义，或者依赖父类如果没删)
        # ConvFCBBoxHead 通常在 init 最后会定义 fc_reg，如果我们在上面删了，这里需要加回来
        # 或者上面的 del self.reg_fcs 是指 shared 部分。
        # 这里假设 self.fc_reg 已经由父类初始化好了，或者我们需要自己加：
        if self.with_reg and not hasattr(self, 'fc_reg'):
             out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
             self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        # temperature scaling
        self.scale = scale

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        torch.manual_seed(0)
        processing = True
        
        # 如果 state_dict 中已经包含了 novel_shared_fcs，说明是测试或Resume，不需要特殊处理
        for param_name in state_dict:
            if 'novel_shared_fcs' in param_name:
                processing = False
                break
        print('-------Loading Weights: Processing Mode-----', processing)

        if processing:
            # 加载 Base Training 的权重文件
            base_weights = torch.load(self.base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']

            # --- [修改3]：直接加载模式 ---
            # 因为权重ID已对齐，直接加载，不做 Re-indexing
            
            # 1. 加载非 head 部分的权重 (如果有)
            for n, p in base_weights.items():
                if 'bbox_head' not in n:
                    state_dict[n] = copy.deepcopy(p)

            # 2. 初始化 bbox_head 的权重
            for n, p in base_weights.items():
                # 处理共享全连接层的分裂 (Shared FCs splitting)
                if 'shared_fcs' in n:
                    # 复制给 base branch
                    new_n_base = n.replace('shared_fcs', 'base_shared_fcs')
                    state_dict[new_n_base] = copy.deepcopy(p)
                    # 复制给 novel branch
                    new_n_novel = n.replace('shared_fcs', 'novel_shared_fcs')
                    state_dict[new_n_novel] = copy.deepcopy(p)

                # 处理分类层 (fc_cls)
                elif 'fc_cls' in n:
                    # 直接深拷贝，假设 ID 已对齐
                    state_dict[n] = copy.deepcopy(p)

                # 处理回归层 (fc_reg)
                elif not self.reg_class_agnostic and 'fc_reg' in n:
                    state_dict[n] = copy.deepcopy(p)
                
                # 其他层直接拷贝
                elif 'fc_reg' in n and self.reg_class_agnostic:
                     state_dict[n] = copy.deepcopy(p)

            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
        else:
            # 如果不是 processing 模式，直接调用父类加载
            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, x, return_fc_feat=False):
        x = x.flatten(1)
        alpha = self.base_alpha

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        
        # 并行通过 base 和 novel 分支，然后加权融合
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1-alpha) * novel_x
            x = self.relu(x)

        bbox_preds = self.fc_reg(x)
        
        # Cosine Similarity Classifier Logic
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        
        # Normalize weights logic
        if self.training:
             # 训练时动态归一化权重
             with torch.no_grad():
                temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                                   dim=1).unsqueeze(1).expand_as(
                                       self.fc_cls.weight.data)
                self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                    temp_norm + 1e-5)
        
        # 注意：测试时通常不应该在 forward 里修改 weight.data，但为了保持和你原代码一致，保留上面的逻辑
        # 如果是测试，建议把上面的 with torch.no_grad 块放到 if self.training 里，
        # 或者依赖 hook 进行归一化。但在 Simple Shot 论文实现中常这么写。

        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist
        
        return scores, bbox_preds


@HEADS.register_module()
class KDBBoxHead(IncreaseBBoxHead):
    def __init__(self,
                 loss_kd_weight=0.001,
                 base_alpha = 0.5,
                 *args,
                 **kwargs):
        # kwargs 包含 novel_class_ids，透传给父类
        super().__init__(base_alpha=base_alpha, *args, **kwargs)

        self.loss_kd = dict()
        self.loss_kd_weight = loss_kd_weight
        self.base_alpha = base_alpha
    
    def forward(self, x, return_fc_feat=False):
        loss_feature_kd = 0
        x = x.flatten(1)
        alpha = self.base_alpha

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            
            # 融合特征
            fused_x = alpha * base_x + (1-alpha) * novel_x
            
            # 计算特征蒸馏损失 (Feature Distillation Loss)
            # 这里的逻辑是让融合后的特征 接近 Base 分支的特征
            if self.training:
                loss_feature_kd += torch.dist(fused_x, base_x, 2)
            
            x = self.relu(fused_x)
        
        if self.training:
            loss_kd = loss_feature_kd / 2.0 * self.loss_kd_weight
            self.loss_kd['loss_kd'] = loss_kd

        bbox_preds = self.fc_reg(x)
        
        # 下面是分类逻辑，与父类一致
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                               dim=1).unsqueeze(1).expand_as(
                                   self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
             
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        
        # 加上 Distillation Loss
        if cls_score is not None:
            losses.update(self.loss_kd)

        return losses


@HEADS.register_module()
class DisKDBBoxHead(KDBBoxHead):

    def __init__(self,
                 dis_loss=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if dis_loss is not None:
            self.dis_loss = build_loss(copy.deepcopy(dis_loss))
        else:
            self.dis_loss = None

    def forward(self, x, return_fc_feat=False):
        
        kd_loss_list = []
        loss_feature_kd = 0 # 也可以保留这个
        
        x = x.flatten(1)
        alpha = self.base_alpha
        
        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            
            # 计算融合特征
            fused_x = alpha * base_x + (1-alpha) * novel_x
            
            # DisKDBBoxHead 特有的 Loss 计算逻辑：
            # 这里是计算 Frobenius Norm (也就是 base_x 和 fused_x 之间的差异)
            if self.training:
                kd_loss_list.append(torch.norm(base_x - fused_x, p='fro', dim=-1))
            
            x = self.relu(fused_x)
            
        if self.training:
            # 聚合每一层的 loss
            kd_loss = torch.cat([l.unsqueeze(0) for l in kd_loss_list], dim=0)
            kd_loss = torch.mean(kd_loss)
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] = kd_loss

        bbox_preds = self.fc_reg(x)
        
        # 分类逻辑
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                               dim=1).unsqueeze(1).expand_as(
                                   self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             cos_dis=None,
             reduction_override=None):
        
        # 调用父类 (KDBBoxHead) 的 loss 方法，它会计算 cls, bbox 和 loss_kd
        losses = super().loss(cls_score, bbox_pred, rois, labels, 
                              label_weights, bbox_targets, bbox_weights, 
                              reduction_override)
        
        # 如果还有额外的 dis_loss (Distillation Loss 的另一种形式，如 RKD 等)，再加进去
        if self.dis_loss is not None and cls_score is not None:
             losses.update(self.dis_loss(cls_score, labels, label_weights))
             
        return losses