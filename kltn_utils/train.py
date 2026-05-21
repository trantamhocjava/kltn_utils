import shutil

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from kltn_utils import kltn_utils


def train_model(
    cp_path, last_state, monitor, end_epoch, amp, model, train_loader, val_loader
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
        strategy="ddp_notebook",
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


def test_model(model, best_model_path, test_loader):
    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    tester.test(model=model, ckpt_path=best_model_path, dataloaders=test_loader)
