from . import const


def get_mode(monitor):
    if monitor in const.METRIC_MAX:
        return "max"
    else:
        return "min"
