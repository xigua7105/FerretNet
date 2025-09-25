import argparse
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from torchvision.transforms import ToPILImage, ToTensor
from src.model.lpd import get_lpd_dict


_lpd_dict = get_lpd_dict()


def _generated_images_v1(path, selection_method):
    raw_image = Image.open(path).convert("RGB")

    img = ToTensor()(raw_image)
    img = img.unsqueeze(0)

    reconstruction = selection_method(img)
    feature = img - reconstruction

    rec_image = ToPILImage()(reconstruction.squeeze(0))
    lpd_image = ToPILImage()(feature.squeeze(0))

    return [raw_image, rec_image, lpd_image]


def show_title(idx):
    if (idx + 1) % 3 == 1:
        plt.title('Raw Image')
    elif (idx + 1) % 3 == 2:
        plt.title('Reconstruction')
    elif (idx + 1) % 3 == 0:
        plt.title('Feature Map')


def plt_show(imgs, column=3):
    assert column % 3 == 0
    row = len(imgs) // column
    rows = row if len(imgs) % column == 0 else row + 1

    plt.figure(figsize=(column * 3, rows * 3))
    for idx, img in enumerate(imgs):
        plt.subplot(rows, column, idx + 1)
        if idx + 1 <= column:
            show_title(idx)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    # plt.savefig("lpd.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lpd_func", type=str, default="median")
    parser.add_argument("--size", type=int, default=3, help="Local neighborhood size, int odd")
    parser.add_argument("--dir_root", type=str, default="analysis/example/images/cam")
    args = parser.parse_args()

    img_paths = glob(args.dir_root + "/*")[:6]
    assert args.lpd_func in _lpd_dict.keys() and args.lpd_func != "origin"
    selection_func = _lpd_dict[args.lpd_func](kernel_size=args.size)
    return_img = []
    for img_path in img_paths:
        return_img.extend(_generated_images_v1(img_path, selection_func))

    plt_show(return_img, column=6)
