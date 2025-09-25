import os
import argparse
from src.model import get_model
from src.model.lpd import get_lpd_dict
from torch.nn import functional as F
from src.model.module.gradcam import GradCAM, overlay_heatmap
from util.demo_tools import plt_show, load_images, image_transforms
from util import get_cfg, load_state_dict


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg_path', type=str, default='configs/Test.yaml')
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    cfg = get_cfg(**parser.parse_args().__dict__)

    img_transform = image_transforms()

    model = get_model(cfg)
    default_ckpt_path = f"checkpoints/4cls_ckpt/{cfg.model.name.lower()}.pth"
    resume_ckpt = args.resume_ckpt if args.resume_ckpt is not None else default_ckpt_path
    load_state_dict(model, resume_ckpt)
    model = model.cuda()
    model.eval()

    lpd_dict = get_lpd_dict()
    lpd_func = lpd_dict[cfg.model.struct["lpd_func"]](kernel_size=cfg.model.struct["window_size"])

    if cfg.model.name.lower().startswith("xception"):
        grad_cam = GradCAM(model=model, target_layer=model.exit_flow.conv)
    elif cfg.model.name.lower().startswith("resnet"):
        grad_cam = GradCAM(model=model, target_layer=model.layer4)
    else:
        grad_cam = GradCAM(model=model, target_layer=model.feature)

    while True:
        img_path = input("please input the path of image: ->") if args.image_path is None else args.image_path
        while not os.path.exists(img_path):
            img_path = input("The input image path does not exist! Enter again: ->")

        raw_image, rec_image, lpd_image = load_images(img_path, lpd_func)

        image = img_transform(raw_image)
        image = image.unsqueeze(0).cuda()

        output = model(image)

        model.zero_grad()
        output[0, 0].backward()
        cam = grad_cam.generate_cam(0)

        cam_image = overlay_heatmap(raw_image, cam, alpha=0.5)

        score = F.sigmoid(output).item()
        if score > 0.5:
            print("This is a synthetic image. prob:{:.4f}\n".format(score))
        else:
            print("This is a real image. prob:{:.4f}\n".format(1.0 - score))

        if args.verbose:
            plt_show(raw_image, cam_image, rec_image, lpd_image)


if __name__ == "__main__":
    main()
