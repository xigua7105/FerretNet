import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS


def pil_loader(path):
    return Image.open(path).convert('RGB')


class OpenDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()

        self.root = root
        self.loader = pil_loader
        self.transform = transform
        self.samples = self.make_dataset()

    def make_dataset(self):
        instance = []
        for root, _, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(IMG_EXTENSIONS):
                    path = os.path.join(root, file)
                    instance.append(path)
        return instance

    def __getitem__(self, index: int):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.samples)
