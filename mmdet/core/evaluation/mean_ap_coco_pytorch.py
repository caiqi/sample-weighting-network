import numpy as np
from .bbox_overlaps import bbox_overlaps


def average_precision(recalls, precisions):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls.view(1, -1)
        precisions = precisions.view(1, -1)
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = torch.zeros(num_scales, dtype=torch.float)
    zeros = torch.zeros((num_scales, 1), dtype=recalls.dtype)
    ones = torch.ones((num_scales, 1), dtype=recalls.dtype)
    mrec = torch.cat((zeros, recalls, ones), dim=1)
    mpre = torch.cat((zeros, precisions, zeros), dim=1)
    for i in range(mpre.shape[1] - 1, 0, -1):
        mpre[:, i - 1] = torch.max(mpre[:, i - 1:i + 1], dim=1)[0]
    for i in range(num_scales):
        ind = torch.where(mrec[i, 1:] != mrec[i, :-1])[0]
        ap[i] = torch.sum(
            (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(det_bboxes, gt_bboxes, iou_thr):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): the detected bbox
        gt_bboxes (ndarray): ground truth bboxes of this image
        gt_ignore (ndarray): indicate if gts are ignored for evaluation or not
        iou_thr (list): list of iou thresholds

    Returns:
        tuple: (tp, fp), two arrays whose elements are 0 and 1
    """
    num_dets = det_bboxes.shape[0]
    num_ious = len(iou_thr)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = torch.zeros(num_ious, num_dets, dtype=torch.float)
    fp = torch.zeros(num_ious, num_dets, dtype=torch.float)
    if gt_bboxes.shape[0] == 0:
        fp[...] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    ious_max = ious.max(axis=1)
    iou_thr = torch.from_numpy(iou_thr).view(-1, 1)
    tp = ious_max.view(1, -1) >= iou_thr
    fp = 1 - tp
    return tp, fp


import torch


def get_cls_results(det_results, gt_bboxes, gt_labels, class_id):
    """Get det results and gt information of a certain class."""
    cls_dets = [det[class_id]
                for det in det_results]  # det bboxes of this class
    cls_gts = []  # gt bboxes of this class
    for j in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[j]
        cls_inds = (gt_labels[j] == class_id + 1)
        cls_gt = gt_bbox[cls_inds, :] if gt_bbox.shape[0] > 0 else gt_bbox
        cls_gts.append(cls_gt)
    return cls_dets, cls_gts


def eval_map_coco_pytorch(det_results,
                  gt_bboxes,
                  gt_labels,
                  gt_ignore=None,
                  iou_thr=(0.5, 0.75)):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list): a list of list, [[cls1_det, cls2_det, ...], ...]
        gt_bboxes (list): ground truth bboxes of each image, a list of K*4
            array.
        gt_labels (list): ground truth labels of each image, a list of K array
        gt_ignore (list): gt ignore indicators of each image, a list of K array
        iou_thr (list): IoU threshold
        dataset (None or str or list): dataset name or dataset classes, there
            are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc.
        print_summary (bool): whether to print the mAP summary

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(gt_bboxes) == len(gt_labels)
    eval_results = []
    num_classes = len(det_results[0])  # positive class num
    gt_labels = [
        label if label.ndim == 1 else label[:, 0] for label in gt_labels
    ]
    num_ious = len(iou_thr)
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts = get_cls_results(
            det_results, gt_bboxes, gt_labels, i)
        # calculate tp and fp for each image
        tpfp = [
            tpfp_default(cls_dets[j], cls_gts[j], iou_thr) for j in range(len(cls_dets))
        ]
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale, gts ignored or beyond scale
        # are not counted
        num_gts = 0
        for j, bbox in enumerate(cls_gts):
            num_gts += bbox.shape[0]

        num_gts = num_gts * torch.ones(num_ious, dtype=torch.int)

        # sort all det bboxes by score, also sort tp and fp
        cls_dets = torch.cat(cls_dets, dim=0)

        num_dets = cls_dets.shape[0]
        sort_inds = torch.argsort(-cls_dets[:, -1])
        tp = torch.cat(tp, dim=1)[:, sort_inds]
        fp = torch.cat(fp, dim=1)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = torch.cumsum(tp, dim=1)
        fp = torch.cumsum(fp, dim=1)
        eps = 1e-6
        recalls = tp / torch.clamp(num_gts.view(-1, 1), min=eps)
        precisions = tp / torch.clamp((tp + fp), min=eps)
        # calculate AP
        ap = average_precision(recalls, precisions)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    # shape (num_classes, num_ious)
    all_ap = torch.cat([cls_result['ap'] for cls_result in eval_results], dim=0)
    all_num_gts = torch.cat(
        [cls_result['num_gts'] for cls_result in eval_results], dim=0)

    mean_ap = [
        all_ap[all_num_gts[:, i] > 0, i].mean()
        for i in range(num_ious)
    ]

    return mean_ap, eval_results
