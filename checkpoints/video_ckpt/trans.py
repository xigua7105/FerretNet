import torch
from glob import glob


pth_path = glob('*.pth')

for pth in pth_path:
    x = torch.load(pth, map_location='cpu')
    y = {
        "model": x
    }
    torch.save(y, pth)
