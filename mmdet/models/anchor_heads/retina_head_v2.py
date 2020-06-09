import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from .anchor_head import AnchorHead
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, bbox_overlaps, bbox2delta,
                        multi_apply, multiclass_nms, force_fp32)
from ..builder import build_loss
from ..registry import HEADS
import torch
from collections import ChainMap


class UncertaintyWithLossFeature(nn.Module):
    def __init__(self):
        super(UncertaintyWithLossFeature, self).__init__()
        self.iou_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.prob_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.cls_loss_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.reg_loss_net = nn.Sequential(
            nn.Linear(in_features=4, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=64 * 4, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
        )
        self.init_weight()

    def init_weight(self):
        for m in [self.iou_net, self.prob_net, self.cls_loss_net, self.reg_loss_net, self.predictor]:
            normal_init(m[0], mean=0.0, std=0.0001, bias=0)
            normal_init(m[2], mean=0.0, std=0.0001, bias=0)

    def forward(self, ious, probs, cls_loss, reg_loss):
        iou_feature = self.iou_net(ious.view(ious.shape[0], -1))
        probs_feature = self.prob_net(probs.view(probs.shape[0], -1))
        cls_loss_feature = self.cls_loss_net(cls_loss.view(cls_loss.shape[0], -1))
        reg_loss_feature = self.reg_loss_net(reg_loss.view(reg_loss.shape[0], 4))
        non_visual_input = torch.cat((iou_feature, probs_feature, cls_loss_feature, reg_loss_feature), dim=1)
        prediction = self.predictor(non_visual_input)
        return prediction


@HEADS.register_module
class RetinaHeadV2(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 transformer_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # configs for predictor related infos
        self.transformer_cfg = transformer_cfg
        self.cls_prediction_min = self.transformer_cfg.cls_prediction_min
        self.cls_prediction_max = self.transformer_cfg.cls_prediction_max
        self.reg_prediction_min = self.transformer_cfg.reg_prediction_min
        self.reg_prediction_max = self.transformer_cfg.reg_prediction_max
        self.uncertainty_cls_weight = self.transformer_cfg.uncertainty_cls_weight
        self.uncertainty_reg_weight = self.transformer_cfg.uncertainty_reg_weight
        # configs for predictor related infos

        octave_scales = np.array(
            [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaHeadV2, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)
        self.uncertainty_predictor = UncertaintyWithLossFeature()
        self.global_step = 0

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        self.global_step = self.global_step + 1
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        original_anchors = [[k for k in m] for m in anchor_list]
        original_anchors = list(map(list, zip(*original_anchors)))
        original_anchors = [torch.cat(m) for m in original_anchors]
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg,
            reduction_override="none")

        cls_scores_flatten = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in
                              cls_scores]
        bbox_preds_flatten = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        labels_list_flatten = [label.reshape(-1, 1) for label in labels_list]
        label_weights_list_flatten = [label_weights.reshape(-1, 1) for label_weights in label_weights_list]
        bbox_targets_list_flatten = [bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list]
        bbox_weights_list_flatten = [bbox_weights.reshape(-1, 4) for bbox_weights in bbox_weights_list]
        original_anchors_flatten = [anchors.reshape(-1, 4) for anchors in original_anchors]
        losses_cls_flatten = [loss_cls.reshape(-1, self.cls_out_channels) for loss_cls in losses_cls]
        losses_bbox_flatten = [loss_bbox.reshape(-1, 4) for loss_bbox in losses_bbox]
        cls_scores_flatten = torch.cat(cls_scores_flatten, dim=0)
        bbox_preds_flatten = torch.cat(bbox_preds_flatten, dim=0)
        labels_list_flatten = torch.cat(labels_list_flatten, dim=0)
        label_weights_list_flatten = torch.cat(label_weights_list_flatten, dim=0)
        bbox_targets_list_flatten = torch.cat(bbox_targets_list_flatten, dim=0)
        bbox_weights_list_flatten = torch.cat(bbox_weights_list_flatten, dim=0)
        original_anchors_flatten = torch.cat(original_anchors_flatten, dim=0)
        losses_cls_flatten = torch.cat(losses_cls_flatten, dim=0)
        losses_bbox_flatten = torch.cat(losses_bbox_flatten, dim=0)
        split_point = [m.shape[0] * m.shape[1] for m in label_weights_list]

        label_weights, bbox_weights, losses = self.predict_weights(
            cls_score=cls_scores_flatten, bbox_pred=bbox_preds_flatten, labels=labels_list_flatten,
            label_weights=label_weights_list_flatten, bbox_targets=bbox_targets_list_flatten,
            bbox_weights=bbox_weights_list_flatten, anchors=original_anchors_flatten, loss_cls=losses_cls_flatten,
            loss_bbox=losses_bbox_flatten)
        label_weights_list_new = torch.split(label_weights, split_point)
        bbox_weights_list_new = torch.split(bbox_weights, split_point)
        label_weights_list_new = [m.reshape(2, -1) for m in label_weights_list_new]
        bbox_weights_list_new = [m.reshape(2, -1, 4) for m in bbox_weights_list_new]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list_new,
            bbox_targets_list,
            bbox_weights_list_new,
            num_total_samples=num_total_samples,
            cfg=cfg,
            reduction_override=None)
        losses.update(
            dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        )
        return losses

    def predict_weights(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, anchors,
                        loss_cls, loss_bbox):
        labels = labels.reshape(-1, )
        pos_inds = labels > 0
        postive_score = cls_score[pos_inds, labels[pos_inds] - 1].sigmoid()
        pos_pred = bbox_pred[pos_inds]
        pos_proposals = anchors[pos_inds]
        pos_bbox = delta2bbox(pos_proposals, pos_pred, means=self.target_means, stds=self.target_stds)
        pos_targets = bbox_targets[pos_inds]
        gt_bboxes = delta2bbox(pos_proposals, pos_targets, means=self.target_means, stds=self.target_stds)
        ious = bbox_overlaps(gt_bboxes, pos_bbox, is_aligned=True).view(-1, )
        total_ious = ious.new_full((pos_inds.numel(),), 0.0)
        total_ious[pos_inds] = ious
        total_scores = postive_score.new_full((pos_inds.numel(),), 0.0)
        total_scores[pos_inds] = postive_score
        uncertainty_prediction = self.uncertainty_predictor(
            total_ious,
            total_scores,
            loss_cls.sum(dim=1).detach().data,
            loss_bbox.detach().data
        )
        losses = dict()
        uncertainty_prediction_cls = uncertainty_prediction[:, 0]
        uncertainty_prediction_reg = uncertainty_prediction[:, 1]
        uncertainty_prediction_cls = torch.clamp(uncertainty_prediction_cls, min=self.cls_prediction_min,
                                                 max=self.cls_prediction_max)
        uncertainty_prediction_reg = torch.clamp(uncertainty_prediction_reg, min=self.reg_prediction_min,
                                                 max=self.reg_prediction_max)
        uncertainty_prediction_cls = torch.ones_like(
                uncertainty_prediction_cls) * uncertainty_prediction_cls.mean()
        losses.update({
                        "loss_uncertainty_cls": uncertainty_prediction_cls.sum() / uncertainty_prediction_cls.numel() * self.uncertainty_cls_weight})
        losses.update({
                "loss_uncertainty_reg": uncertainty_prediction_reg[
                                                pos_inds].mean() * self.uncertainty_reg_weight})

        uncertainty_prediction_reg = torch.exp(-1. * uncertainty_prediction_reg)
        uncertainty_prediction_cls = torch.exp(-1. * uncertainty_prediction_cls)
        losses.update({
            "cls_prediction_pos": uncertainty_prediction_cls[pos_inds].mean(),
            "cls_prediction_neg": uncertainty_prediction_cls[~pos_inds].mean(),
            "cls_prediction_reg": uncertainty_prediction_reg[pos_inds].mean(),
        })
        bbox_weights = bbox_weights.detach().data * uncertainty_prediction_reg.view(-1, 1)
        label_weights = label_weights.detach().data * uncertainty_prediction_cls.view(-1, 1)
        return label_weights, bbox_weights, losses

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg, reduction_override=None):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples, reduction_override=reduction_override)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples, reduction_override=reduction_override)
        return loss_cls, loss_bbox

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
