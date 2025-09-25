import os
import cv2
import torch
import argparse
from util import get_cfg, load_state_dict
from src.model import get_model
from torch.nn import functional as F
from src.model.module.gradcam import GradCAM, overlay_heatmap
from util.demo_tools import image_transforms, cv2_to_pil, pil_to_cv2
from util.tools import get_timepc


def main():
    _dtype = torch.half

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg_path', type=str, default='configs/Test.yaml')
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--resume_ckpt', type=str, default='checkpoints/video_ckpt/ferret-video-1.pth')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    cfg = get_cfg(**parser.parse_args().__dict__)

    img_transform = image_transforms()

    model = get_model(cfg)
    load_state_dict(model, args.resume_ckpt)
    model = model.cuda().to(_dtype)
    model.eval()

    if cfg.model.name.lower().startswith("xception"):
        grad_cam = GradCAM(model=model, target_layer=model.exit_flow.conv)
    elif cfg.model.name.lower().startswith("resnet"):
        grad_cam = GradCAM(model=model, target_layer=model.layer4)
    else:
        grad_cam = GradCAM(model=model, target_layer=model.feature)

    print(
        '----------------------------------------------------------------------------------\n'
        'example image path are as fellows:\n'
    )
    for root, _, files in os.walk("analysis/example"):
        for file in files:
            if file.lower().endswith(".mp4"):
                sample_path = os.path.join(root, file)
                print(sample_path)
    print('----------------------------------------------------------------------------------\n')

    while True:
        # 输入视频地址
        video_path = input("please input the path of video: ->")

        while not os.path.exists(video_path):
            video_path = input("The input video path does not exist! Enter again: ->")

        t_s_video = get_timepc()
        print("Start detect video!")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        real_count = 0
        synthetic_count = 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            t_s_cost = get_timepc()
            frame_index += 1

            t_s_data = get_timepc()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2_to_pil(frame)
            image = img_transform(frame)
            image = image.unsqueeze(0).cuda().to(_dtype)
            t_e_data = get_timepc()
            data_time = t_e_data - t_s_data

            t_s_inference = get_timepc(cuda_synchronize=True)
            output = model(image)
            model.zero_grad()
            t_e_inference = get_timepc(cuda_synchronize=True)
            inference_time = t_e_inference - t_s_inference

            score = F.sigmoid(output).item()
            if score > 0.5:
                text = ("| Frame index:[{}/{}] | Synthetic prob:{:.4f}".format(frame_index, frame_count, score))
                synthetic_count += 1
            else:
                text = ("| Frame index:[{}/{}] | Real prob:{:.4f}".format(frame_index, frame_count, 1.0 - score))
                real_count += 1

            t_s_verbose = get_timepc()
            if args.verbose:
                # GRAD CAM
                output[0, 0].backward()
                cam = grad_cam.generate_cam(0)
                cam_image = overlay_heatmap(frame, cam, alpha=0.5)
                cam_image = pil_to_cv2(cam_image)

                cv2.putText(cam_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Processed Frame', cam_image)
                cv2.waitKey(1)
            t_e_verbose = get_timepc()
            verbose_time = t_e_verbose - t_s_verbose

            t_e_cost = get_timepc()
            cost_time = t_e_cost - t_s_cost
            print("| Frame index:[{}/{}] | [resolution:{}*{}]\t[time cost:{:.4f}s | data time:{:.4f}s | inference time:{:.4f}s | verbose time:{:.4f}s]".format(frame_index, frame_count, width, height, cost_time, data_time, inference_time, verbose_time))
            print(text)
            print("---> | Real Frames:{} | Synthetic Frames:{}|\n".format(real_count, synthetic_count))

        cap.release()
        cv2.destroyAllWindows()

        t_e_video = get_timepc()
        video_time = t_e_video - t_s_video
        print("Finish detect video! Time Cost:{:.4f}s".format(video_time))
        real_ratio = 100 * real_count / frame_count
        synthetic_ratio = 100 * synthetic_count / frame_count
        print("The real frames account for {:.2f}%, and the synthetic frames account for {:.2f}%.".format(real_ratio, synthetic_ratio))


if __name__ == "__main__":
    main()
