from ._register import MODELS
from .lpd import get_lpd_dict

lpd_dict = get_lpd_dict()


def get_model(cfg):
    model_struct = cfg.model.struct.copy()
    model_name = model_struct.pop('name')

    return MODELS.get_module(model_name)(**model_struct, lpd_dict=lpd_dict)
