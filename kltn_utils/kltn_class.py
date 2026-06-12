import os
import shutil
import time

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset

from . import kltn_utils


class MetricCalculator:
    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.c_pred = []
        self.c_true = []

        self.loss_dict = self.get_loss_dict()

    def update(self, result):
        y_pred = torch.argmax(result["y_logits"].detach(), dim=1)
        concept_pred = (torch.sigmoid(result["c_logits"].detach()) >= 0.5).long()

        self.y_pred.append(y_pred.cpu())
        self.c_pred.append(concept_pred.cpu())

        self.y_true.append(result["y"].cpu())
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

    def get_loss_dict(self):
        pass

    def update_loss_dict(self, result):
        for key, value in self.loss_dict.items():
            value.append(result[key].item())

    def return_loss_dict(self):
        result = {}

        for key, value in self.loss_dict.items():
            result[key] = np.array(value).mean()

        return result


class BaseTrain(pl.LightningModule):
    def __init__(self, CustomMetric, cp_path):
        super().__init__()

        self.train_metric = CustomMetric()
        self.val_metric = CustomMetric()
        self.test_metric = CustomMetric()

        self.cp_path = cp_path

    def get_loss(self, batch):
        pass

    def update_optimizer_manually(self, result):
        pass

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.val_metric.reset()
        self.start_time = time.time()

    def training_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update optimizer manually if automatic_optimization = false
        if not self.automatic_optimization:
            self.update_optimizer_manually(result)

        # Update loss and metric
        self.train_metric.update(result)

        return result["loss"]

    def on_validation_epoch_end(self):
        epoch_time = time.time() - self.start_time
        metric = {
            **kltn_utils.add_prefix_in_dict(
                self.train_metric.return_metrics(), "train"
            ),
            **kltn_utils.add_prefix_in_dict(self.val_metric.return_metrics(), "val"),
            "epoch_time": epoch_time,
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
        test_time = time.time() - self.start_time
        test_result = {
            **kltn_utils.add_prefix_in_dict(self.test_metric.return_metrics(), "test"),
            "test_time": test_time,
        }

        kltn_utils.save_dict_to_json(test_result, f"{self.cp_path}/test_result.json")

    def test_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update loss and metric
        self.test_metric.update(result)

    def log_result(self, metric):
        for key, value in metric.items():
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True)


class BaseTrainer:
    def __init__(self, config) -> None:
        self.config = config
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def next(self):
        if self.config.mode == "train":
            self.train_model(
                cp_path=self.config.cp_path,
                last_state=self.config.last_state,
                monitor=self.config.monitor,
                end_epoch=self.config.end_epoch,
                amp=self.config.amp,
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
            )
        else:
            self.test_model(
                model=self.model,
                best_model_path=self.config.best_model,
                test_loader=self.test_loader,
            )

    def train_model(
        self,
        cp_path,
        last_state,
        monitor,
        end_epoch,
        amp,
        model,
        train_loader,
        val_loader,
    ):
        if last_state is not None:
            kltn_utils.rank_zero_info_newline(f"Restore last state from {last_state}")
            ckpt_path = f"{last_state}/last.ckpt"
            shutil.copy(f"{last_state}/best.ckpt", f"{cp_path}/best.ckpt")
        else:
            ckpt_path = None

        model_ckpt = ModelCheckpoint(
            dirpath=cp_path,
            save_top_k=1,
            save_last=True,
            monitor=monitor,
            mode=kltn_utils.get_mode(monitor),
            filename="best",
        )
        csv_logger = CSVLogger(save_dir=cp_path, name="", version="")

        trainer = Trainer(
            accelerator="gpu",
            devices=2,
            max_epochs=end_epoch,
            precision="16-mixed" if amp else 32,
            strategy=DDPStrategy(find_unused_parameters=True),
            default_root_dir=cp_path,
            num_sanity_val_steps=0,
            logger=[csv_logger],
            callbacks=[model_ckpt],
        )

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )

    def test_model(self, model, best_model_path, test_loader):
        tester = Trainer(
            accelerator="gpu",
            devices=1,
            precision=32,
        )

        tester.test(model=model, ckpt_path=best_model_path, dataloaders=test_loader)


class BaseKFoldTrainer:
    def __init__(self, config) -> None:
        self.config = config
        self.kfold_obj = None
        self.train_dataset = None
        self.test_dataset = None

    def build_model_fn(self):
        pass

    def get_mode(self, monitor):
        pass

    def next(self):
        if self.config.mode == "train":
            self.train_model(
                kfold_obj=self.kfold_obj,
                cp_path=self.config.cp_path,
                last_state=self.config.last_state,
                monitor=self.config.monitor,
                end_epoch=self.config.end_epoch,
                amp=self.config.amp,
                train_dataset=self.train_dataset,
            )
        else:
            self.test_model(
                best_model_path=self.config.best_model,
                test_dataset=self.test_dataset,
            )

    def train_model(
        self,
        kfold_obj,
        cp_path,
        last_state,
        monitor,
        end_epoch,
        amp,
        train_dataset,
    ):
        num_fold = kfold_obj.n_splits

        list_best_score = []
        mode = self.get_mode(monitor)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold_obj.split(train_dataset)):
            kltn_utils.rank_zero_info_newline(f"run fold {fold_idx+1}/{num_fold}")
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2,
            )

            model = self.build_model_fn()

            if last_state is not None:
                kltn_utils.rank_zero_info_newline(
                    f"Restore last state from {last_state}"
                )
                ckpt_path = f"{last_state}/last.ckpt"
                shutil.copy(f"{last_state}/best.ckpt", f"{cp_path}/best.ckpt")
            else:
                ckpt_path = None

            fold_cp_path = f"{cp_path}/fold_{fold_idx}"
            os.makedirs(fold_cp_path, exist_ok=True)

            model_ckpt = ModelCheckpoint(
                dirpath=fold_cp_path,
                save_top_k=1,
                save_last=True,
                monitor=monitor,
                mode=mode,
                filename="best",
            )
            csv_logger = CSVLogger(save_dir=fold_cp_path, name="", version="")

            trainer = Trainer(
                accelerator="gpu",
                devices=2,
                max_epochs=end_epoch,
                precision="16-mixed" if amp else 32,
                strategy=DDPStrategy(find_unused_parameters=True),
                default_root_dir=fold_cp_path,
                num_sanity_val_steps=0,
                logger=[csv_logger],
                callbacks=[model_ckpt],
            )

            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path,
            )

            best_score = model_ckpt.best_model_score.item()
            # TODO: DEBUG
            kltn_utils.rank_zero_info_newline(f"best_score: {best_score}")
            # END DEBUG
            list_best_score.append(best_score)

        best_idx = kltn_utils.find_best_idx_in_list(list_best_score, mode)
        best_folder_path = f"{cp_path}/fold_{best_idx}"

        shutil.copy(f"{best_folder_path}/last.ckpt", f"{cp_path}/last.ckpt")
        shutil.copy(f"{best_folder_path}/best.ckpt", f"{cp_path}/best.ckpt")

    def test_model(self, best_model_path, test_dataset):
        model = self.build_model_fn()

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
        )

        tester = Trainer(
            accelerator="gpu",
            devices=1,
            precision=32,
        )

        tester.test(model=model, ckpt_path=best_model_path, dataloaders=test_loader)
