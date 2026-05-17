import time

import numpy as np
import pytorch_lightning as pl
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


class BaseTrain(pl.LightningModule):
    def __init__(self, config, train_metric, val_metric, test_metric, cp_path):
        super().__init__()

        self.config = config

        self.train_metric = train_metric
        self.val_metric = val_metric
        self.test_metric = test_metric

        self.cp_path = cp_path

    # define optimizers
    def configure_optimizers(self):
        res = {
            "optimizer": kltn_utils.build_optimizer(self.model, self.config),
        }

        return res

    def get_loss(self, batch):
        pass

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.val_metric.reset()
        self.start_time = time.time()

    def training_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update loss and metric
        self.train_metric.update(result)

        return result["loss"]

    def on_validation_epoch_end(self):
        metric = {
            **kltn_utils.add_prefix_in_dict(
                self.train_metric.return_metrics(), "train"
            ),
            **kltn_utils.add_prefix_in_dict(self.test_metric.return_metrics(), "val"),
            "epoch_time": time.time() - self.start_time,
        }

        self.log_result(metric)

    def validation_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update loss and metric
        self.val_metric.update(result)

    def on_test_epoch_start(self):
        self.test_metric.reset()
        self.start_time = time.time()

    def on_test_epoch_end(self):
        test_result = {
            **kltn_utils.add_prefix_in_dict(self.test_metric.return_metrics(), "test"),
            "test_time": time.time() - self.start_time,
        }

        kltn_utils.save_dict_to_json(test_result, f"{self.cp_path}/test_result.json")

    def test_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update loss and metric
        self.test_metric.update(result)

    def log_result(self, metric):
        for key, value in metric.items():
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True)
