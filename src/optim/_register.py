import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from util.register import REGISTER
from .scheduler import WarmUpLR

default_optimizers = ['Adam', 'SGD', 'AdamW', 'ASGD']
OPTIMIZERS = REGISTER("Optimizers")
for opt_name in default_optimizers:
    opt_name = getattr(optim, opt_name, None)
    OPTIMIZERS.register_module(opt_name)


SCHEDULERS = REGISTER("Schedulers")
SCHEDULERS.register_module(WarmUpLR)
SCHEDULERS.register_module(MultiStepLR)
SCHEDULERS.register_module(CosineAnnealingLR, "Cosine")
