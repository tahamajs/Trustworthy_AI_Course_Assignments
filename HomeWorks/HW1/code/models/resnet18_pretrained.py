"""
ResNet18 pre-trained on ImageNet as feature extractor for downstream tasks (e.g. SVHN, CIFAR10).
Replaces the final linear layer for num_classes and optionally freezes the backbone.
"""
import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18 as tv_resnet18
    try:
        from torchvision.models import ResNet18_Weights
    except ImportError:
        ResNet18_Weights = None
except ImportError:
    ResNet18_Weights = None
    tv_resnet18 = None


def resnet18_imagenet(num_classes=10, in_channels=3, freeze_backbone=False, pretrained=True):
    """
    Load torchvision ResNet18 with ImageNet weights, replace fc for num_classes.
    in_channels must be 3 (ImageNet is RGB). return_features in forward returns (logits, 512-d feat).
    """
    if tv_resnet18 is None:
        raise ImportError("torchvision.models.resnet18 is required for pretrained ResNet18.")

    # Prefer weights API (torchvision 0.13+), fallback to pretrained=True
    if pretrained and ResNet18_Weights is not None:
        try:
            model = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            model = tv_resnet18(pretrained=True)
    elif pretrained:
        model = tv_resnet18(pretrained=True)
    else:
        model = tv_resnet18(weights=None) if ResNet18_Weights is not None else tv_resnet18(pretrained=False)

    # Replace final layer for our num_classes
    model.fc = nn.Linear(512, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    # Wrap so we have the same interface as custom resnet (forward with return_features)
    class _Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x, return_features=False):
            feat = backbone_forward(self.backbone, x)
            out = self.backbone.fc(feat)
            if return_features:
                return out, feat
            return out

    return _Wrapper(model)


def backbone_forward(model, x):
    """Forward through ResNet up to (and including) avgpool, return flattened 512-d vector."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x
