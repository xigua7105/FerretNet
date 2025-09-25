from ._register import TESTER


def get_tester(cfg):
    return TESTER.get_module(cfg.tester.name)(cfg)
