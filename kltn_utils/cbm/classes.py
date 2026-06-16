import os
import shutil

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from .. import kltn_utils
from ..path import utils as path_utils


class BaseStratifiedKFoldTrainer:
    def __init__(self, config) -> None:
        self.config = config
        self.train_dataset = None
        self.test_dataset = None

    def build_model_fn(self):
        pass

    def get_mode(self, monitor):
        return kltn_utils.get_mode(monitor)

    def next(self):
        if self.config.mode == "train":
            self.train_model()
        else:
            self.test_model()

    def train_model(
        self,
    ):
        kfold_obj = StratifiedKFold(
            n_splits=self.config.num_fold,
            shuffle=True,
            random_state=42,
        )

        list_best_score = []
        mode = self.get_mode(self.config.monitor)

        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold_obj.split(
                X=self.train_dataset.file_paths,
                y=self.train_dataset.labels,
            )
        ):
            kltn_utils.rank_zero_info_newline(
                f"run fold {fold_idx+1}/{self.config.num_fold}"
            )
            train_subset = Subset(self.train_dataset, train_idx)
            val_subset = Subset(self.train_dataset, val_idx)

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
            model.setup_grad()

            if self.config.last_state is not None:
                kltn_utils.rank_zero_info_newline(
                    f"Restore last state from {self.config.last_state}"
                )
                ckpt_path = f"{self.config.last_state}/last.ckpt"
                shutil.copy(
                    f"{self.config.last_state}/best.ckpt",
                    f"{self.config.cp_path}/best.ckpt",
                )
            else:
                ckpt_path = None

            fold_cp_path = f"{self.config.cp_path}/fold_{fold_idx}"
            os.makedirs(fold_cp_path, exist_ok=True)

            model_ckpt = ModelCheckpoint(
                dirpath=fold_cp_path,
                save_top_k=1,
                save_last=True,
                monitor=self.config.monitor,
                mode=mode,
                filename="best",
            )
            csv_logger = CSVLogger(save_dir=fold_cp_path, name="", version="")

            trainer = Trainer(
                accelerator="gpu",
                devices=2,
                max_epochs=self.config.end_epoch,
                precision="16-mixed" if self.config.amp else 32,
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
            list_best_score.append(best_score)

        best_idx = kltn_utils.find_best_idx_in_list(list_best_score, mode)
        best_folder_path = f"{self.config.cp_path}/fold_{best_idx}"

        path_utils.copy_files(
            src_folder_path=best_folder_path,
            des_folder_path=self.config.cp_path,
            file_names=["last.ckpt", "best.ckpt", "metrics.csv"],
        )

    def test_model(self):
        model = self.build_model_fn()

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
        )

        tester = Trainer(
            accelerator="gpu",
            devices=1,
            precision=32,
        )

        tester.test(
            model=model, ckpt_path=self.config.best_model, dataloaders=test_loader
        )


class BaseStratifiedKFoldTrainerV1(BaseStratifiedKFoldTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def train_model(
        self,
    ):
        kfold_obj = StratifiedKFold(
            n_splits=self.config.num_fold,
            shuffle=True,
            random_state=42,
        )

        folds = list(
            kfold_obj.split(
                X=self.train_dataset.file_paths,
                y=self.train_dataset.labels,
            )
        )[: self.config.num_run_fold]

        list_best_score = []
        mode = self.get_mode(self.config.monitor)

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_subset = Subset(self.train_dataset, train_idx)
            val_subset = Subset(self.train_dataset, val_idx)

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
            model.setup_grad()

            if self.config.last_state is not None:
                kltn_utils.rank_zero_info_newline(
                    f"Restore last state from {self.config.last_state}"
                )
                ckpt_path = f"{self.config.last_state}/last.ckpt"
                shutil.copy(
                    f"{self.config.last_state}/best.ckpt",
                    f"{self.config.cp_path}/best.ckpt",
                )
            else:
                ckpt_path = None

            fold_cp_path = f"{self.config.cp_path}/fold_{fold_idx}"
            os.makedirs(fold_cp_path, exist_ok=True)

            model_ckpt = ModelCheckpoint(
                dirpath=fold_cp_path,
                save_top_k=1,
                save_last=True,
                monitor=self.config.monitor,
                mode=mode,
                filename="best",
            )
            csv_logger = CSVLogger(save_dir=fold_cp_path, name="", version="")

            trainer = Trainer(
                accelerator="gpu",
                devices=2,
                max_epochs=self.config.end_epoch,
                precision="16-mixed" if self.config.amp else 32,
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
            list_best_score.append(best_score)

        best_idx = kltn_utils.find_best_idx_in_list(list_best_score, mode)
        best_folder_path = f"{self.config.cp_path}/fold_{best_idx}"

        path_utils.copy_files(
            src_folder_path=best_folder_path,
            des_folder_path=self.config.cp_path,
            file_names=["last.ckpt", "best.ckpt", "metrics.csv"],
        )
