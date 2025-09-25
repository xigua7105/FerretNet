import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Normalize, Compose, ToTensor, ToPILImage, CenterCrop, RandomCrop


def cv2_to_pil(raw_image):
    pil_image = Image.fromarray(raw_image)
    return pil_image


def pil_to_cv2(pil_image):
    cv2_img = np.array(pil_image)
    return cv2_img


def _generated_images(path, lpd_func):
    raw_image = Image.open(path).convert("RGB")

    img = ToTensor()(raw_image)
    img = img.unsqueeze(0)

    reconstruction = lpd_func(img)
    feature = img - reconstruction

    rec_image = ToPILImage()(reconstruction.squeeze(0))
    lpd_image = ToPILImage()(feature.squeeze(0))

    return [raw_image, rec_image, lpd_image]


def load_images(path, lpd_func=None):
    if lpd_func is not None:
        return _generated_images(path, lpd_func)
    else:
        return Image.open(path).convert("RGB"), None, None


def image_transforms():
    return Compose([
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])


def plt_show(raw_image, cam_image, rec_image=None, lpd_image=None):
    plt.figure(figsize=(16, 5))

    plt.subplot(141)
    plt.imshow(raw_image)
    plt.title('Raw Image')
    plt.axis('off')

    if rec_image is not None:
        plt.subplot(142)
        plt.imshow(rec_image)
        plt.title('Rec Image')
        plt.axis('off')

    if lpd_image is not None:
        plt.subplot(143)
        plt.imshow(lpd_image)
        plt.title('LPD')
        plt.axis('off')

    plt.subplot(144)
    plt.imshow(cam_image, cmap="rainbow")
    plt.title('GradCAM')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

