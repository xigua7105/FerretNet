from ._register import LOSS_FN


def get_loss_fn(cfg):
    loss_terms = cfg.loss.loss_terms.copy()
    loss_name = loss_terms.pop('name')
    return LOSS_FN.get_module(loss_name)(**loss_terms)
