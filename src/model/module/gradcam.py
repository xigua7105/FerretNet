import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image


def preprocess_image(path):
    raw_image = Image.open(path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    norm_input = preprocess(raw_image).unsqueeze(0)
    return raw_image, norm_input


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, class_idx):
        if self.features is None or self.gradients is None:
            raise ValueError("Gradients or features not available. Ensure backward() has been called.")

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.features, dim=1).squeeze(0)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.detach().cpu().numpy()

    def clear(self):
        """Clears stored features and gradients to avoid memory issues."""
        self.features = None
        self.gradients = None


def overlay_heatmap(image, cam, alpha=0.5):
    heatmap = plt.get_cmap('jet')(cam)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)

    return Image.blend(image, heatmap, alpha=alpha)


def generate_cam(img_path, model, grad_cam):
    raw_image, input_tensor = preprocess_image(img_path)

    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, predicted_class].backward()
    cam = grad_cam.generate_cam(predicted_class)

    return raw_image, overlay_heatmap(raw_image, cam, alpha=0.5)