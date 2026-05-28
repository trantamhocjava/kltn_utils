import numpy as np
from scipy.spatial import cKDTree


def _threshold(preds, threshold=0.5):
    return (preds > threshold).float()


def _get_confusion_matrix(preds, targets):
    """
    preds, targets: (B, 1, H, W)
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    TP = (preds * targets).sum()
    TN = ((1 - preds) * (1 - targets)).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()

    return TP, TN, FP, FN


def dice_score(preds, targets, smooth=1e-6):
    preds = _threshold(preds)
    targets = targets.float()

    TP, _, FP, FN = _get_confusion_matrix(preds, targets)

    return ((2 * TP + smooth) / (2 * TP + FP + FN + smooth)).item()


def iou_score(preds, targets, smooth=1e-6):
    preds = _threshold(preds)
    targets = targets.float()

    TP, _, FP, FN = _get_confusion_matrix(preds, targets)

    return ((TP + smooth) / (TP + FP + FN + smooth)).item()


def accuracy(preds, targets, smooth=1e-6):
    preds = _threshold(preds)
    targets = targets.float()

    TP, TN, FP, FN = _get_confusion_matrix(preds, targets)

    return ((TP + TN + smooth) / (TP + TN + FP + FN + smooth)).item()


def specificity(preds, targets, smooth=1e-6):
    preds = _threshold(preds)
    targets = targets.float()

    _, TN, FP, _ = _get_confusion_matrix(preds, targets)

    return ((TN + smooth) / (TN + FP + smooth)).item()


def precision(preds, targets, smooth=1e-6):
    preds = _threshold(preds)
    targets = targets.float()

    TP, _, FP, _ = _get_confusion_matrix(preds, targets)

    return ((TP + smooth) / (TP + FP + smooth)).item()


def hausdorff_distance_95(preds, targets):
    """
    preds, targets: (H, W) numpy array (binary) - Xử lý từng ảnh một
    """
    preds = preds.astype(bool)
    targets = targets.astype(bool)

    pred_points = np.argwhere(preds)
    target_points = np.argwhere(targets)

    # Nếu mô hình đoán trống trơn hoặc Ground Truth trống
    if len(pred_points) == 0 or len(target_points) == 0:
        return np.nan

    # Dùng cKDTree để tìm điểm lân cận gần nhất (cực kỳ nhanh)
    tree_pred = cKDTree(pred_points)
    tree_target = cKDTree(target_points)

    # Khoảng cách từ mỗi điểm Pred tới điểm Target gần nhất
    d_pred_to_target, _ = tree_target.query(pred_points)
    # Khoảng cách từ mỗi điểm Target tới điểm Pred gần nhất
    d_target_to_pred, _ = tree_pred.query(target_points)

    # Lấy bách phân vị thứ 95 thay vì lấy max()
    hd95_1 = np.percentile(d_pred_to_target, 95)
    hd95_2 = np.percentile(d_target_to_pred, 95)

    return max(hd95_1, hd95_2)


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
