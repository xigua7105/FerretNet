from util.register import REGISTER
from .resnet import ResNet
from .xception import Xception
from .ferretnet import Ferret

MODELS = REGISTER("models")
MODELS.register_module(ResNet)
MODELS.register_module(Ferret)
MODELS.register_module(Xception)
