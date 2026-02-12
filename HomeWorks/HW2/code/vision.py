"""Computer-vision interpretability utilities for HW2 (Grad-CAM, Guided Backprop, SmoothGrad).

This file provides compact implementations suitable for demo and experiments with a
pretrained VGG16 from torchvision.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms


def get_vgg16(device="cpu"):
    vgg = models.vgg16(pretrained=True).to(device).eval()
    return vgg


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        loss = out[0, class_idx]
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# Guided Backpropagation (modifies ReLU backward)
class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                # backward hook to allow only positive gradients
                self.handlers.append(module.register_backward_hook(self._relu_backward_hook))

    def _relu_backward_hook(self, module, grad_in, grad_out):
        return (F.relu(grad_in[0]),)

    def generate(self, input_tensor, target_class=None):
        input_tensor.requires_grad = True
        out = self.model(input_tensor)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        loss = out[0, target_class]
        self.model.zero_grad()
        loss.backward()
        grad = input_tensor.grad.detach().cpu().squeeze().numpy()
        return grad


def preprocess_image(pil_image):
    preproc = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preproc(pil_image).unsqueeze(0)


def smoothgrad(model, input_tensor, target_class=None, n_samples=25, stdev=0.15):
    model.zero_grad()
    avg_grad = 0
    for i in range(n_samples):
        noise = torch.randn_like(input_tensor) * stdev
        noisy = (input_tensor + noise).clamp(0, 1).detach()
        noisy.requires_grad = True
        out = model(noisy)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        loss = out[0, target_class]
        loss.backward()
        avg_grad += noisy.grad.detach()
    avg_grad = (avg_grad / n_samples).cpu().squeeze().numpy()
    return avg_grad


# Activation maximization (simple gradient ascent on input)
def activation_maximization(model, target_class, steps=200, lr=1.0, tv_weight=1e-5, device="cpu"):
    x = torch.randn(1, 3, 224, 224, device=device, requires_grad=True) * 0.1
    opt = torch.optim.Adam([x], lr=lr)

    def tv_loss(img):
        return torch.mean(torch.abs(img[:, :, :-1] - img[:, :, 1:])) + torch.mean(
            torch.abs(img[:, :-1, :] - img[:, 1:, :])
        )

    for i in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = -out[0, target_class] + tv_weight * tv_loss(x.squeeze(0))
        loss.backward()
        opt.step()
        # small jitter
        x.data = x.data + (torch.randn_like(x) * 0.001)
    img = x.detach().cpu().squeeze()
    # clamp to [0,1] after denormalizing a common normalization is not applied here
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.numpy()
