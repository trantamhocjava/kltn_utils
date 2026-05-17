import shutil

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from kltn_utils import kltn_utils

from . import kltn_const


def train_model(last_state, monitor, end_epoch, amp, model, train_loader, val_loader):
    if last_state is not None:
        kltn_utils.rank_zero_info_newline(f"Restore last state from {last_state}")
        ckpt_path = f"{last_state}/last.ckpt"
        shutil.copy(f"{last_state}/best.ckpt", f"{kltn_const.CP_PATH}/best.ckpt")
    else:
        ckpt_path = None

    model_ckpt = ModelCheckpoint(
        dirpath=kltn_const.CP_PATH,
        save_top_k=1,
        save_last=True,
        monitor=monitor,
        mode=kltn_utils.get_mode(monitor),
        filename="best",
    )
    csv_logger = CSVLogger(save_dir=kltn_const.CP_PATH, name="", version="")

    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        max_epochs=end_epoch,
        precision="16-mixed" if amp else 32,
        strategy="ddp",
        default_root_dir=kltn_const.CP_PATH,
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
