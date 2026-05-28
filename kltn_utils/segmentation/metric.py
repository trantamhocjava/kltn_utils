import numpy as np
from scipy.spatial import cKDTree


def _get_confusion_matrix(preds, targets):
    """
    preds, targets: numpy array binary, ví dụ (B, 1, H, W) hoặc (H, W)
    """
    preds = preds.astype(np.float32).reshape(-1)
    targets = targets.astype(np.float32).reshape(-1)

    TP = (preds * targets).sum()
    TN = ((1 - preds) * (1 - targets)).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()

    return TP, TN, FP, FN


def dice_score(preds, targets, smooth=1e-6):
    TP, _, FP, FN = _get_confusion_matrix(preds, targets)
    return float((2 * TP + smooth) / (2 * TP + FP + FN + smooth))


def iou_score(preds, targets, smooth=1e-6):
    TP, _, FP, FN = _get_confusion_matrix(preds, targets)
    return float((TP + smooth) / (TP + FP + FN + smooth))


def accuracy(preds, targets, smooth=1e-6):
    TP, TN, FP, FN = _get_confusion_matrix(preds, targets)
    return float((TP + TN + smooth) / (TP + TN + FP + FN + smooth))


def specificity(preds, targets, smooth=1e-6):
    _, TN, FP, _ = _get_confusion_matrix(preds, targets)
    return float((TN + smooth) / (TN + FP + smooth))


def precision(preds, targets, smooth=1e-6):
    TP, _, FP, _ = _get_confusion_matrix(preds, targets)
    return float((TP + smooth) / (TP + FP + smooth))


def hausdorff_distance_95(preds, targets):
    """
    preds, targets: numpy array binary, thường là (H, W)
    """
    preds = preds.astype(bool)
    targets = targets.astype(bool)

    pred_points = np.argwhere(preds)
    target_points = np.argwhere(targets)

    if len(pred_points) == 0 or len(target_points) == 0:
        return np.nan

    tree_pred = cKDTree(pred_points)
    tree_target = cKDTree(target_points)

    d_pred_to_target, _ = tree_target.query(pred_points)
    d_target_to_pred, _ = tree_pred.query(target_points)

    hd95_1 = np.percentile(d_pred_to_target, 95)
    hd95_2 = np.percentile(d_target_to_pred, 95)

    return float(max(hd95_1, hd95_2))


def cal_metric(preds, targets, mode):
    if mode == "dice":
        result = dice_score(preds, targets)
    elif mode == "iou":
        result = iou_score(preds, targets)
    elif mode == "accuracy":
        result = accuracy(preds, targets)
    elif mode == "specificity":
        result = specificity(preds, targets)
    elif mode == "precision":
        result = precision(preds, targets)
    elif mode == "hausdorff_distance_95":
        result = hausdorff_distance_95(preds, targets)

    return result
