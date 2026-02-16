"""Computer-vision interpretability utilities for HW2 (Grad-CAM, Guided Backprop, SmoothGrad).

This file provides compact implementations suitable for demo and experiments with a
pretrained VGG16 from torchvision.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms


def get_vgg16(device="cpu", prefer_pretrained=True):
    """Load VGG16 with graceful fallback when pretrained weights are unavailable."""
    weights_enum = getattr(models, "VGG16_Weights", None)
    try:
        if prefer_pretrained:
            if weights_enum is not None:
                vgg = models.vgg16(weights=weights_enum.IMAGENET1K_V1)
            else:  # older torchvision API
                vgg = models.vgg16(pretrained=True)
        else:
            if weights_enum is not None:
                vgg = models.vgg16(weights=None)
            else:
                vgg = models.vgg16(pretrained=False)
    except Exception as exc:  # pragma: no cover - offline/pretrained fallback
        print(
            f"[vision] Could not load pretrained VGG16 ({exc.__class__.__name__}). "
            "Falling back to randomly initialized weights."
        )
        if weights_enum is not None:
            vgg = models.vgg16(weights=None)
        else:
            vgg = models.vgg16(pretrained=False)
    vgg = vgg.to(device).eval()
    return vgg


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()
        if out.requires_grad:
            out.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def close(self):
        if getattr(self, "forward_handle", None) is not None:
            self.forward_handle.remove()
            self.forward_handle = None

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
                # Guided backprop is incompatible with in-place ReLUs.
                module.inplace = False
                # backward hook to allow only positive gradients
                if hasattr(module, "register_full_backward_hook"):
                    handle = module.register_full_backward_hook(
                        self._relu_backward_hook
                    )
                else:  # pragma: no cover - old torch fallback
                    handle = module.register_backward_hook(self._relu_backward_hook)
                self.handlers.append(handle)

    def _relu_backward_hook(self, module, grad_in, grad_out):
        if not grad_in or grad_in[0] is None:
            return grad_in
        return (F.relu(grad_in[0]),)

    def generate(self, input_tensor, target_class=None):
        x = input_tensor.detach().clone().requires_grad_(True)
        out = self.model(x)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        loss = out[0, target_class]
        self.model.zero_grad()
        loss.backward()
        grad = x.grad.detach().cpu().squeeze().numpy()
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
    model.eval()
    if target_class is None:
        with torch.no_grad():
            target_class = model(input_tensor).argmax(dim=1).item()
    avg_grad = torch.zeros_like(input_tensor)
    for i in range(n_samples):
        noise = torch.randn_like(input_tensor) * stdev
        noisy = (input_tensor + noise).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(noisy)
        loss = out[0, target_class]
        loss.backward()
        avg_grad += noisy.grad.detach()
    avg_grad = (avg_grad / n_samples).cpu().squeeze().numpy()
    return avg_grad


def smoothgrad_guided_backprop(
    model,
    input_tensor,
    target_class=None,
    n_samples=25,
    stdev=0.15,
):
    """SmoothGrad with Guided Backpropagation: average Guided Backprop maps over K noisy inputs."""
    model.eval()
    if target_class is None:
        with torch.no_grad():
            target_class = model(input_tensor).argmax(dim=1).item()
    gb = GuidedBackprop(model)
    accum = None
    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor, device=input_tensor.device) * stdev
        noisy = (input_tensor + noise).detach().clone()
        noisy.requires_grad_(True)
        grad_map = gb.generate(noisy, target_class=target_class)
        if accum is None:
            accum = np.zeros_like(grad_map)
        accum += grad_map
    accum = accum / n_samples
    return accum


def smoothgrad_guided_gradcam(
    model,
    target_layer,
    input_tensor,
    target_class=None,
    n_samples=25,
    stdev=0.15,
):
    """SmoothGrad + Guided Grad-CAM: average Guided Grad-CAM maps over K noisy inputs."""
    model.eval()
    if target_class is None:
        with torch.no_grad():
            target_class = model(input_tensor).argmax(dim=1).item()
    accum = None
    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor, device=input_tensor.device) * stdev
        noisy = (input_tensor + noise).detach().clone()
        cam = GradCAM(model, target_layer)
        heat = cam(noisy, class_idx=target_class)
        cam.close()
        gb = GuidedBackprop(model)
        gb_grad = gb.generate(noisy, target_class=target_class)
        guided_cam = heat * np.abs(gb_grad).max(axis=0)
        if accum is None:
            accum = np.zeros_like(guided_cam)
        accum += guided_cam
    accum = accum / n_samples
    return accum


# Activation maximization (simple gradient ascent on input)
def activation_maximization(
    model,
    target_class,
    steps=200,
    lr=1.0,
    tv_weight=1e-5,
    device="cpu",
    use_random_shifts=True,
    shift_max=8,
    image_size=224,
):
    x = torch.nn.Parameter(
        torch.randn(1, 3, image_size, image_size, device=device) * 0.1
    )
    opt = torch.optim.Adam([x], lr=lr)

    def tv_loss(img):
        return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        )

    for i in range(steps):
        opt.zero_grad()
        x_forward = x
        if use_random_shifts and shift_max > 0:
            dy = int(torch.randint(-shift_max, shift_max + 1, (1,), device=device).item())
            dx = int(torch.randint(-shift_max, shift_max + 1, (1,), device=device).item())
            x_forward = torch.roll(x_forward, shifts=(dy, dx), dims=(2, 3))
        out = model(x_forward)
        loss = -out[0, target_class] + tv_weight * tv_loss(x)
        loss.backward()
        opt.step()
        # small jitter
        x.data = x.data + (torch.randn_like(x) * 0.001)
    img = x.detach().cpu().squeeze(0)
    # clamp to [0,1] after denormalizing a common normalization is not applied here
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.numpy()
