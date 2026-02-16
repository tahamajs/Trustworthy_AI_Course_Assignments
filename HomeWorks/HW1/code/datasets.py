import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import FakeData

try:
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError:
    StratifiedShuffleSplit = None


# Augmentation presets: "default" = full aug; "digits_safe" = no horizontal flip (can swap 6/9), no heavy rotation
def get_transforms(dataset_name, train=True, image_size=32, augment=False, to_rgb3=True, augmentation_preset='default'):
    if dataset_name.lower() in ("svhn",):
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        base = [transforms.Resize((image_size, image_size))]
    elif dataset_name.lower() in ("mnist",):
        mean = (0.1307,)
        std = (0.3081,)
        base = [transforms.Resize((image_size, image_size))]
    elif dataset_name.lower() in ("cifar10",):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        base = [transforms.Resize((image_size, image_size))]
    else:
        mean = (0.5,)
        std = (0.5,)
        base = [transforms.Resize((image_size, image_size))]

    if train and augment:
        if augmentation_preset == 'digits_safe':
            # No RandomHorizontalFlip (can swap 6/9); mild crop and jitter only
            aug = [
                transforms.RandomCrop(image_size, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        else:
            # default: current behavior
            aug = [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
    else:
        aug = []

    # convert grayscale to RGB if requested (for MNIST -> make 3-channels)
    conv = [transforms.Lambda(lambda img: img.convert('RGB'))] if to_rgb3 else []
    if to_rgb3 and len(mean) == 1:
        mean = (mean[0], mean[0], mean[0])
        std = (std[0], std[0], std[0])

    transform = transforms.Compose(base + conv + aug + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform


def _stratified_subset(dataset, n_samples, seed=42):
    """Return a stratified Subset of dataset with n_samples (class-balanced)."""
    if StratifiedShuffleSplit is None:
        # fallback: take first n_samples
        n_samples = min(n_samples, len(dataset))
        return Subset(dataset, list(range(n_samples)))
    try:
        labels = np.array(dataset.labels if hasattr(dataset, 'labels') else dataset.targets)
    except Exception:
        labels = np.array([dataset[i][1] for i in range(min(len(dataset), 50000))])
        if len(dataset) > len(labels):
            labels = np.concatenate([labels, np.array([dataset[i][1] for i in range(len(labels), len(dataset))])])
    n_samples = min(n_samples, len(dataset))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    train_idx, _ = next(sss.split(np.arange(len(dataset)), labels))
    return Subset(dataset, train_idx.tolist())


def get_dataloaders(dataset_name, batch_size=128, image_size=32, augment=False, num_workers=4, demo=False,
                    augmentation_preset='default', cifar10_train_ratio=None, max_train_samples=None, seed=42):
    """
    cifar10_train_ratio: if float (e.g. 0.2), use 20%% of CIFAR10 train for training and 80%% for validation (stratified).
    max_train_samples: if int (e.g. 750), cap training set to this many samples with stratified sampling (for fine-tuning).
    """
    try:
        if dataset_name.lower() == 'svhn':
            train_set = datasets.SVHN(root='data/svhn', split='train', download=True,
                                      transform=get_transforms('svhn', train=True, image_size=image_size, augment=augment, to_rgb3=True, augmentation_preset=augmentation_preset))
            test_set = datasets.SVHN(root='data/svhn', split='test', download=True,
                                     transform=get_transforms('svhn', train=False, image_size=image_size, augment=False, to_rgb3=True))
            num_classes = 10
            in_channels = 3
            if max_train_samples is not None and max_train_samples < len(train_set):
                train_set = _stratified_subset(train_set, max_train_samples, seed=seed)
        elif dataset_name.lower() == 'mnist':
            train_set = datasets.MNIST(root='data/mnist', train=True, download=True,
                                       transform=get_transforms('mnist', train=True, image_size=image_size, augment=augment, to_rgb3=True, augmentation_preset=augmentation_preset))
            test_set = datasets.MNIST(root='data/mnist', train=False, download=True,
                                      transform=get_transforms('mnist', train=False, image_size=image_size, augment=False, to_rgb3=True))
            num_classes = 10
            in_channels = 3
        elif dataset_name.lower() == 'cifar10':
            full_train = datasets.CIFAR10(root='data/cifar10', train=True, download=True,
                                         transform=get_transforms('cifar10', train=True, image_size=image_size, augment=augment, to_rgb3=True, augmentation_preset=augmentation_preset))
            test_set = datasets.CIFAR10(root='data/cifar10', train=False, download=True,
                                        transform=get_transforms('cifar10', train=False, image_size=image_size, augment=False, to_rgb3=True))
            num_classes = 10
            in_channels = 3
            if cifar10_train_ratio is not None and 0 < cifar10_train_ratio < 1 and StratifiedShuffleSplit is not None:
                labels = np.array(full_train.targets)
                sss = StratifiedShuffleSplit(n_splits=1, train_size=cifar10_train_ratio, random_state=seed)
                train_idx, val_idx = next(sss.split(np.arange(len(full_train)), labels))
                train_set = Subset(full_train, train_idx.tolist())
                # Use 80% portion as "validation" (fitting) set for early stopping / reporting
                val_transform = get_transforms('cifar10', train=False, image_size=image_size, augment=False, to_rgb3=True)
                val_full = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=val_transform)
                val_set = Subset(val_full, val_idx.tolist())
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                return train_loader, test_loader, num_classes, in_channels
            train_set = full_train
        else:
            raise ValueError('Unknown dataset: ' + dataset_name)
    except Exception:
        # Offline/demo fallback using synthetic data to keep pipelines runnable
        channels = 3 if dataset_name.lower() in ('svhn', 'cifar10') else 1
        num_classes = 10
        in_channels = 3
        transform = get_transforms('svhn' if channels == 3 else 'mnist', train=True, image_size=image_size, augment=augment, to_rgb3=True, augmentation_preset=augmentation_preset)
        train_size = 2048 if demo else 8192
        test_size = 512 if demo else 2048
        train_set = FakeData(size=train_size, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform)
        test_set = FakeData(size=test_size, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform)

    if demo:
        # shrink dataset sizes for quick debugging / smoke checks
        train_n = min(1024, len(train_set))
        test_n = min(256, len(test_set))
        train_set = Subset(train_set, list(range(train_n)))
        test_set = Subset(test_set, list(range(test_n)))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, test_loader, num_classes, in_channels

    # Avoid multiprocessing pickling issues for lambda-based FakeData transforms
    if isinstance(train_set, FakeData):
        num_workers = 0

    # standard loaders for full runs
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, num_classes, in_channels
