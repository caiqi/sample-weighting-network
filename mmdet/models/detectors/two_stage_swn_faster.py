import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler, delta2bbox, bbox_overlaps, eval_map_coco
from mmcv.cnn import constant_init, normal_init
import numpy as np

class UncertaintyWithLossFeature(nn.Module):
    def __init__(self, class_num, visual_feature_dim):
        super(UncertaintyWithLossFeature, self).__init__()
        self.class_num = class_num
        self.visual_feature_dim = visual_feature_dim
        self.embedding_dim = 64

        self.iou_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.prob_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.cls_loss_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.reg_loss_net = nn.Sequential(
            nn.Linear(in_features=4, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim * 4, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=2),
        )
        self.init_weight()

    def init_weight(self):
        for m in [self.iou_net, self.prob_net, self.cls_loss_net, self.reg_loss_net, self.predictor]:
            normal_init(m[0], mean=0.0, std=0.0001, bias=0)
            normal_init(m[2], mean=0.0, std=0.0001, bias=0)

    def forward(self, ious, probs, cls_loss, reg_loss):
        iou_feature = self.iou_net(ious.view(ious.shape[0], -1))
        probs_feature = self.prob_net(probs.view(probs.shape[0], -1))
        cls_loss_feature = self.cls_loss_net(cls_loss.view(cls_loss.shape[0], 1))
        reg_loss_feature = self.reg_loss_net(reg_loss.view(reg_loss.shape[0], 4))
        non_visual_input = torch.cat((iou_feature, probs_feature, cls_loss_feature, reg_loss_feature), dim=1)
        prediction = self.predictor(non_visual_input)
        return prediction

@DETECTORS.register_module
class TwoStageDetectorSWNFaster(BaseDetector, RPNTestMixin, BBoxTestMixin,
                                                              MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 transformer=None):
        super(TwoStageDetectorSWNFaster, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)
        self.global_step = 0
        self.transformer_cfg = transformer

        self.cls_prediction_min = self.transformer_cfg.cls_prediction_min
        self.cls_prediction_max = self.transformer_cfg.cls_prediction_max
        self.reg_prediction_min = self.transformer_cfg.reg_prediction_min
        self.reg_prediction_max = self.transformer_cfg.reg_prediction_max

        self.uncertainty_cls_weight = self.transformer_cfg.uncertainty_cls_weight
        self.uncertainty_reg_weight = self.transformer_cfg.uncertainty_reg_weight

        self.class_num = self.transformer_cfg.class_num
        self.visual_feature_dim = self.transformer_cfg.visual_feature_dim
        self.negative_regularization = self.transformer_cfg.negative_regularization

        self.uncertainty_predictor = UncertaintyWithLossFeature(class_num=self.class_num,
                                                                visual_feature_dim=self.visual_feature_dim)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def print_stats(self, name, tensor):
        if self.global_step % 10 != 0:
            return
        print("{}: max: {} min: {} : mean: {} std: {}".format(
            name,
            tensor.max().item(),
            tensor.min().item(),
            tensor.mean().item(),
            tensor.std().item(),
        ))

    def init_weights(self, pretrained=None):
        super(TwoStageDetectorSWNFaster, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        self.global_step += 1
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)

            # start creating features for input
            pos_inds = bbox_targets[0] > 0
            cls_score_post_softmax = cls_score.softmax(dim=1)
            pos_probs_single_item = cls_score_post_softmax[pos_inds, bbox_targets[0][pos_inds]]
            pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                           4)[pos_inds, bbox_targets[0][pos_inds]]
            # simple trick to remove NaN values
            pos_bbox_pred[pos_bbox_pred != pos_bbox_pred] = 0
            pos_gts = torch.cat([k.pos_gt_bboxes for k in sampling_results], dim=0)
            pos_proposal = torch.cat([k.pos_bboxes for k in sampling_results], dim=0)
            target_means = self.bbox_head.target_means
            target_stds = self.bbox_head.target_stds
            pos_bbox = delta2bbox(pos_proposal, pos_bbox_pred, means=target_means, stds=target_stds)
            pos_ious = bbox_overlaps(pos_gts, pos_bbox, is_aligned=True)
            pos_ious = pos_ious.view(-1, )

            total_ious = pos_ious.new_full((pos_inds.numel(),), 0.0)
            total_ious[pos_inds] = pos_ious
            total_probs_single_item = pos_probs_single_item.new_full((pos_inds.numel(),), 0.0)
            total_probs_single_item[pos_inds] = pos_probs_single_item

            with torch.no_grad():
                loss_bbox_as_features = self.bbox_head.loss(cls_score, bbox_pred,
                                                            *bbox_targets, reduction_override="none")
                cls_loss_as_feature = loss_bbox_as_features["loss_cls"].detach().data
                bbox_loss_as_feature = loss_bbox_as_features["loss_bbox"].detach().data
                losses.update({
                    "pos_cls_loose_value": cls_loss_as_feature[pos_inds].mean(),
                    "neg_cls_loose_value": cls_loss_as_feature[~pos_inds].mean(),
                    "reg_loose_value": bbox_loss_as_feature.sum(dim=1).mean(),
                })
                bbox_loss_as_feature_full = torch.zeros((pos_inds.numel(), bbox_loss_as_feature.shape[1]),
                                                        device=bbox_loss_as_feature.device).type(
                    bbox_loss_as_feature.type())
                bbox_loss_as_feature_full[pos_inds] = bbox_loss_as_feature

            uncertainty_prediction = self.uncertainty_predictor(total_ious.detach().data,
                                                                        total_probs_single_item.detach().data,
                                                                        cls_loss_as_feature,
                                                                        bbox_loss_as_feature_full)
            uncertainty_prediction_cls = uncertainty_prediction[:, 0]
            uncertainty_prediction_reg = uncertainty_prediction[:, 1]

            uncertainty_prediction_cls = torch.clamp(uncertainty_prediction_cls, min=self.cls_prediction_min,
                                                     max=self.cls_prediction_max)
            uncertainty_prediction_reg = torch.clamp(uncertainty_prediction_reg, min=self.reg_prediction_min,
                                                     max=self.reg_prediction_max)
            negative_avg = uncertainty_prediction_cls[~pos_inds].mean()
            uncertainty_prediction_cls[~pos_inds] = torch.ones_like(
                uncertainty_prediction_cls[~pos_inds]) * negative_avg
            positive_avg = uncertainty_prediction_cls[pos_inds].mean()
            uncertainty_prediction_cls[pos_inds] = torch.ones_like(
                uncertainty_prediction_cls[pos_inds]) * positive_avg
            uncertainty_prediction_cls_for_regularization_pos = (uncertainty_prediction_cls[
                                                                     pos_inds].mean() * self.uncertainty_cls_weight)
            uncertainty_prediction_cls_for_regularization_neg = (uncertainty_prediction_cls[
                                                                     ~pos_inds].mean() * self.negative_regularization)
            losses.update({
                "loss_uncertainty_cls_pos": uncertainty_prediction_cls_for_regularization_pos})
            losses.update({
                "loss_uncertainty_cls_neg": uncertainty_prediction_cls_for_regularization_neg})
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
            bbox_targets_weighted = [m.detach().data for m in bbox_targets]
            bbox_targets_weighted[3] = bbox_targets_weighted[3] * uncertainty_prediction_reg.view(-1, 1)
            bbox_targets_weighted[1] = bbox_targets_weighted[1] * uncertainty_prediction_cls.view(-1, )
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                                *bbox_targets_weighted)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
