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
        shutil.move(f"{best_folder_path}/last.ckpt", f"{cp_path}/last.ckpt")
        shutil.move(f"{best_folder_path}/best.ckpt", f"{cp_path}/best.ckpt")

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
