from ._register import TRAINERS


def get_trainer(cfg):
    return TRAINERS.get_module(cfg.trainer.name)(cfg)
