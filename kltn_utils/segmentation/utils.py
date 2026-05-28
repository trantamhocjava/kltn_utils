import torch
from pytorch_lightning import seed_everything

from .. import kltn_const


def get_mode(monitor):
    if monitor in const.METRIC_MAX:
        return "max"
    else:
        return "min"


def seed_everything_in_pl():
    seed_everything(kltn_const.SEEDING, workers=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
