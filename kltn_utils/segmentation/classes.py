import numpy as np
import torch

from . import metric


class MetricCalculator:
    def reset(self):
        self.mask_pred = []
        self.mask_true = []

        self.loss_dict = self.get_loss_dict()

    def update(self, result):
        mask_pred = (torch.sigmoid(result["mask_logits"].detach()) > 0.5).int()
        self.mask_pred.append(mask_pred.cpu())
        self.mask_true.append(result["mask"].cpu())

        self.update_loss_dict(result)

    def return_metrics(self):
        mask_true = torch.cat(self.mask_true, dim=0).numpy()
        mask_pred = torch.cat(self.mask_pred, dim=0).numpy()

        dice = metric.dice_score(mask_pred, mask_true)
        iou = metric.iou_score(mask_pred, mask_true)
        accuracy = metric.accuracy(mask_pred, mask_true)
        specificity = metric.specificity(mask_pred, mask_true)
        precision = metric.precision(mask_pred, mask_true)
        hausdorff_distance_95 = metric.hausdorff_distance_95(mask_pred, mask_true)

        return {
            "dice": dice,
            "iou": iou,
            "accuracy": accuracy,
            "specificity": specificity,
            "precision": precision,
            "hausdorff_distance_95": hausdorff_distance_95,
            **self.return_loss_dict(),
        }

    def update_loss_dict(self, result):
        for key, value in self.loss_dict.items():
            value.append(result[key].item())

    def return_loss_dict(self):
        result = {}

        for key, value in self.loss_dict.items():
            result[key] = np.array(value).mean()

        return result

    def get_loss_dict(self):
        pass
