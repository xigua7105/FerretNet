import glob
import argparse
import os.path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from src.model.lpd import MaskMedianValues
from src.model.module.gradcam import GradCAM
from src.model import get_model
from util import get_cfg, load_state_dict


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    return image, input_tensor


def overlay_heatmap(image, cam, alpha=0.5):
    heatmap = plt.get_cmap('jet')(cam)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)

    superimposed_image = Image.blend(image, heatmap, alpha=alpha)
    return superimposed_image


# An intuitive analysis that aligns better with human visual perception
def _generate_images_v1(img_path):
    ori_image = Image.open(img_path)

    img = transforms.ToTensor()(ori_image)
    img = img.unsqueeze(0)

    reconstruction = MaskMedianValues(kernel_size=3)(img)

    lpd_map = img - reconstruction

    reconstruction = transforms.ToPILImage()(reconstruction.squeeze(0))

    lpd_map = transforms.ToPILImage()(lpd_map.squeeze(0))

    return ori_image, reconstruction, lpd_map


def _generate_images_v2(img_path):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    ori_image = Image.open(img_path)

    img = transforms.ToTensor()(ori_image)
    img = transforms.Normalize(mean=mean, std=std)(img)
    img = img.unsqueeze(0)

    reconstruction = MaskMedianValues(kernel_size=3)(img)
    lpd_map = img - reconstruction

    for c in range(3):
        reconstruction[:, c, :, :] = reconstruction[:, c, :, :] * std[c] + mean[c]
        lpd_map[:, c, :, :] = lpd_map[:, c, :, :] * std[c] + mean[c]

    reconstruction = transforms.ToPILImage()(reconstruction.squeeze(0))
    lpd_map = transforms.ToPILImage()(lpd_map.squeeze(0))

    return ori_image, reconstruction, lpd_map


# What Neural Networks "See"
def _generate_images_v3(img_path):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    ori_image = Image.open(img_path)

    img = transforms.ToTensor()(ori_image)
    img = transforms.Normalize(mean=mean, std=std)(img)
    img = img.unsqueeze(0)

    reconstruction = MaskMedianValues(kernel_size=3)(img)
    lpd_map = img - reconstruction

    image = transforms.ToPILImage()(img.squeeze(0))
    reconstruction = transforms.ToPILImage()(reconstruction.squeeze(0))
    lpd_map = transforms.ToPILImage()(lpd_map.squeeze(0))

    return image, reconstruction, lpd_map


def get_heatmap(root):
    original_image, input_tensor = preprocess_image(root)

    grad_cam = GradCAM(model, model.feature)
    output = model(input_tensor.cuda())
    predicted_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, predicted_class].backward()
    cam = grad_cam.generate_cam(predicted_class)
    heatmap = overlay_heatmap(original_image, cam, alpha=0.5)
    return heatmap


def generate_images(source_path, target_path: str = "paper/", gen_mode: str = "v1", suffix: str = "png", quality: int = 100, dpi: int = 300):
    assert suffix in ["png", "jpeg", "webp"]
    assert gen_mode in ["v1", "v2", "v3"]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print('create output dir:{}'.format(target_dir))
    base_name = os.path.basename(source_path).split('.')[0]
    if gen_mode == 'v3':
        original_image, reconstruction, lpd_map = _generate_images_v3(source_path)
    elif gen_mode == 'v2':
        original_image, reconstruction, lpd_map = _generate_images_v2(source_path)
    else:
        original_image, reconstruction, lpd_map = _generate_images_v1(source_path)

    heatmap = get_heatmap(source_path)

    if suffix == "png":
        save_path = os.path.join(target_path, base_name + '.png')
        original_image.save(save_path, format=suffix)

        save_path = os.path.join(target_path, base_name + '_reconstruction' + f'_{gen_mode}' + '.png')
        reconstruction.save(save_path, format=suffix)

        save_path = os.path.join(target_path, base_name + '_lpd' + f'_{gen_mode}' + '.png')
        lpd_map.save(save_path, format=suffix)

        save_path = os.path.join(target_path, base_name + '_heatmap' + '.png')
        heatmap.save(save_path, format=suffix)
    else:
        _suffix = suffix if suffix == "webp" else "jpg"

        save_path = os.path.join(target_path, base_name + f'.{_suffix}')
        original_image.save(save_path, format=suffix, quality=quality, dpi=(dpi, dpi))

        save_path = os.path.join(target_path, base_name + '_reconstruction' + f'_{gen_mode}' + f'.{_suffix}')
        reconstruction.save(save_path, format=suffix, quality=quality, dpi=(dpi, dpi))

        save_path = os.path.join(target_path, base_name + '_lpd' + f'_{gen_mode}' + f'.{_suffix}')
        lpd_map.save(save_path, format=suffix, quality=quality, dpi=(dpi, dpi))

        save_path = os.path.join(target_path, base_name + '_heatmap' + f'.{_suffix}')
        heatmap.save(save_path, format=suffix, quality=quality, dpi=(dpi, dpi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg_path', type=str, default='configs/Test.yaml')
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default='analysis/example/images/cam')
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    parser.add_argument('--gen_mode', type=str, default='v1', help='v1, v2, v3')
    args = parser.parse_args()

    cfg = get_cfg(**parser.parse_args().__dict__)

    model = get_model(cfg)
    default_ckpt_path = f"checkpoints/4cls_ckpt/{cfg.model.name.lower()}.pth"
    resume_ckpt = args.resume_ckpt if args.resume_ckpt is not None else default_ckpt_path
    load_state_dict(model, resume_ckpt)
    model.cuda().eval()

    image_paths = glob.glob(f"{args.image_dir}/*")

    # image_paths = [
    #     r"fake_biggan.png"
    # ]

    target_dir = f"analysis/outputs/{args.gen_mode}"

    for path in tqdm(image_paths):
        generate_images(path, target_dir, gen_mode=args.gen_mode, suffix="png")
