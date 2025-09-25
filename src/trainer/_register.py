from util.register import REGISTER
from .cls_trainer import CLSTrainer
from .syn_trainer import SynTrainer

TRAINERS = REGISTER("trainers")
TRAINERS.register_module(CLSTrainer)
TRAINERS.register_module(SynTrainer)
