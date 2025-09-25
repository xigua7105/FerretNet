from torchvision import transforms
from util.register import REGISTER
from .dataset import ImageFolderDefault, SynDataset


# register torchvision transforms
TV_TRANSFORMS = REGISTER("transforms")
tv_transforms = [
    "Compose", "ToTensor", "PILToTensor", "ToPILImage", "Normalize", "Resize", "CenterCrop", "RandomApply",
    "RandomChoice", "RandomOrder", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop",
    "ColorJitter", "RandomRotation", "InterpolationMode"
]
for tv_tran_name in tv_transforms:
    tv_transform = getattr(transforms, tv_tran_name, None)
    TV_TRANSFORMS.register_module(tv_transform, tv_tran_name)


# register datasets
DATASETS = REGISTER("Datasets")
DATASETS.register_module(ImageFolderDefault)
DATASETS.register_module(SynDataset)
