import numpy as np
import torch

from . import kltn_utils


class MetricCalculator:
    def reset(self, loss_dict):
        self.y_pred = []
        self.y_true = []
        self.c_pred = []
        self.c_true = []

        self.loss_dict = loss_dict

    def update(self, result):
        y_pred = torch.argmax(result["y_logits"].detach(), dim=1)
        self.y_pred.append(y_pred.cpu())
        self.y_true.append(result["y"].cpu())

        concept_pred = (torch.sigmoid(result["c_logits"].detach()) >= 0.5).long()
        self.c_pred.append(concept_pred.cpu())
        self.c_true.append(result["c"].cpu())

        self.update_loss_dict(result)

    def return_metrics(self):
        y_true = torch.cat(self.y_true, dim=0).numpy()
        y_pred = torch.cat(self.y_pred, dim=0).numpy()
        concept_true = torch.cat(self.c_true, dim=0).numpy()
        concept_pred = torch.cat(self.c_pred, dim=0).numpy()

        y_acc = kltn_utils.cal_label_accuracy(y_true, y_pred, "acc")
        y_bmac = kltn_utils.cal_label_accuracy(y_true, y_pred, "bmac")
        c_acc = kltn_utils.cal_concept_accuracy(concept_true, concept_pred, "acc")
        c_overall_acc = kltn_utils.cal_concept_accuracy(
            concept_true, concept_pred, "overall_acc"
        )

        return {
            "y_acc": y_acc,
            "y_bmac": y_bmac,
            "c_acc": c_acc,
            "c_overall_acc": c_overall_acc,
            **self.return_loss_dict(),
        }

    def update_loss_dict(self, result):
        for key, value in result.items():
            self.loss_dict[key].append(value.item())

    def return_loss_dict(self):
        result = {}

        for key, value in self.loss_dict.items():
            result[key] = np.array(value).mean()

        return result
