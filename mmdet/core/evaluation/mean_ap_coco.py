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
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
    ones = np.ones((num_scales, 1), dtype=recalls.dtype)
    mrec = np.hstack((zeros, recalls, ones))
    mpre = np.hstack((zeros, precisions, zeros))
    for i in range(mpre.shape[1] - 1, 0, -1):
        mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
    for i in range(num_scales):
        ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
        ap[i] = np.sum(
            (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(det_bboxes, gt_bboxes, gt_ignore, iou_thr):
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
    num_gts = gt_bboxes.shape[0]
    num_ious = len(iou_thr)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_ious, num_dets), dtype=np.float32)
    fp = np.zeros((num_ious, num_dets), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        fp[...] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])

    # if no area range is specified, gt_area_ignore is all False
    for k in range(num_ious):
        gt_covered = np.zeros(num_gts, dtype=bool)
        for i in sort_inds:
            if ious_max[i] >= iou_thr[k]:
                matched_gt = ious_argmax[i]
                if not gt_ignore[matched_gt]:
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        # print("Duplicate detection")
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            else:
                fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, gt_bboxes, gt_labels, gt_ignore, class_id):
    """Get det results and gt information of a certain class."""
    cls_dets = [det[class_id]
                for det in det_results]  # det bboxes of this class
    cls_gts = []  # gt bboxes of this class
    cls_gt_ignore = []
    for j in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[j]
        cls_inds = (gt_labels[j] == class_id + 1)
        cls_gt = gt_bbox[cls_inds, :] if gt_bbox.shape[0] > 0 else gt_bbox
        cls_gts.append(cls_gt)
        if gt_ignore is None:
            cls_gt_ignore.append(np.zeros(cls_gt.shape[0], dtype=np.int32))
        else:
            cls_gt_ignore.append(gt_ignore[j][cls_inds])
    return cls_dets, cls_gts, cls_gt_ignore


def eval_map_coco(det_results,
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
    if gt_ignore is not None:
        assert len(gt_ignore) == len(gt_labels)
        for i in range(len(gt_ignore)):
            assert len(gt_labels[i]) == len(gt_ignore[i])
    eval_results = []
    num_classes = len(det_results[0])  # positive class num
    gt_labels = [
        label if label.ndim == 1 else label[:, 0] for label in gt_labels
    ]
    num_ious = len(iou_thr)
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gt_ignore = get_cls_results(
            det_results, gt_bboxes, gt_labels, gt_ignore, i)
        # calculate tp and fp for each image
        tpfp = [
            tpfp_default(cls_dets[j], cls_gts[j], cls_gt_ignore[j], iou_thr) for j in range(len(cls_dets))
        ]
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale, gts ignored or beyond scale
        # are not counted
        num_gts = 0
        for j, bbox in enumerate(cls_gts):
            num_gts += np.sum(np.logical_not(cls_gt_ignore[j]))
        num_gts = num_gts * np.ones(num_ious, dtype=int)

        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
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
    all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
    all_num_gts = np.vstack(
        [cls_result['num_gts'] for cls_result in eval_results])
    mean_ap = [
        all_ap[all_num_gts[:, i] > 0, i].mean()
        if np.any(all_num_gts[:, i] > 0) else 0.0
        for i in range(num_ious)
    ]

    return mean_ap, eval_results
