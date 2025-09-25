import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS


def pil_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolderDefault(ImageFolder):
    def __init__(self, root, cfg=None, is_train: bool = False, transform=None, target_transform=None):
        img_loader = pil_loader

        self.root = root
        self.is_train = is_train
        super(ImageFolderDefault, self).__init__(
            root=self.root,
            loader=img_loader,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index: int) -> dict:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'img': sample, 'target': target}

    def __len__(self) -> int:
        return len(self.samples)


class SynDataset(Dataset):
    def __init__(self, root, cfg, is_train: bool = False, transform=None, target_transform=None):
        self.root = root
        self.is_train = is_train
        self.class_to_idx = cfg.data.class_to_idx if cfg.data.class_to_idx is not None else {"0_real": 0, "1_fake": 1}
        cfg.data.class_to_idx = self.class_to_idx

        self.loader = pil_loader
        self.transform = transform
        self.target_transform = target_transform

        self.samples = self.make_dataset()

    def make_dataset(self):
        instance = []

        for root, _, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(IMG_EXTENSIONS):
                    path = os.path.join(root, file)
                    # To adapt to the window system
                    cls_name = path.replace("\\", "/").split('/')[-2]
                    assert cls_name in self.class_to_idx.keys()
                    label = self.class_to_idx[cls_name]

                    item = path, label
                    instance.append(item)

        return instance

    def __getitem__(self, index: int) -> dict:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'img': sample, 'target': int(target)}

    def __len__(self) -> int:
        return len(self.samples)


class ImageFolderLMDB(Dataset):
    def __init__(self, cfg, is_train: bool = False, transform=None, target_transform=None):
        super().__init__()

