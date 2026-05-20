import os

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


def loader_to_tensors(loader):
    X, Y = zip(*loader)
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)


class SubsetWithAttrs(Subset):
    """Subset that forwards unknown attributes (e.g., classes, targets) to the base dataset."""

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def make_train_val_split(train_base, val_base, val_ratio, seed, save_index=None):
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed,
    )

    targets = np.array(train_base.targets)
    indices = np.arange(len(train_base))

    train_idx, val_idx = next(splitter.split(indices, targets))

    if save_index is not None:
        np.savez(
            save_index,
            train_idx=train_idx,
            val_idx=val_idx,
        )

    train_ds = SubsetWithAttrs(train_base, train_idx)
    val_ds = SubsetWithAttrs(val_base, val_idx)

    return train_ds, val_ds


class CombinedImageFolder(Dataset):
    """
    Combine ImageFolder datasets from root/train and root/test into one dataset.
    Keeps ImageFolder-style attributes like classes, class_to_idx, targets, samples.
    """

    def __init__(self, data_dir, transform=None):
        self.transform = transform

        train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"))
        test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"))

        if train_ds.classes != test_ds.classes:
            raise ValueError(
                "Class mismatch between train and test folders:\n"
                f"train classes: {train_ds.classes}\n"
                f"test classes:  {test_ds.classes}"
            )

        self.classes = train_ds.classes
        self.class_to_idx = train_ds.class_to_idx

        self.samples = train_ds.samples + test_ds.samples
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = datasets.folder.default_loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def get_datasets(data_dir, seed, train_tfms=None, val_tfms=None):

    train_base = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_tfms,
    )

    val_base = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=val_tfms,
    )

    train_ds, val_ds = make_train_val_split(
        train_base, val_base, val_ratio=0.1, seed=seed
    )

    test_ds = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=val_tfms,
    )

    return train_ds, val_ds, test_ds


def get_dataloaders(
    data_dir: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    train_tfms,
    val_tfms,
    pin_memory: bool = True,
):
    """
    Build datasets + dataloaders for one seed.
    """

    if seed is None:
        train_ds = datasets.ImageFolder(
            os.path.join(data_dir, "train"),
            transform=train_tfms,
        )
        test_ds = datasets.ImageFolder(
            os.path.join(data_dir, "test"),
            transform=val_tfms,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, test_loader, train_ds, test_ds

    train_ds, val_ds, test_ds = get_datasets(
        data_dir,
        seed,
        train_tfms,
        val_tfms,
    )

    _check_classes(train_ds, val_ds, test_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def _check_classes(train_ds, val_ds, test_ds):
    """
    Internal consistency check.
    """
    same = train_ds.classes == val_ds.classes and train_ds.classes == test_ds.classes

    if not same:
        raise ValueError("Dataset class mismatch!")


def get_tfms(img_size):

    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    return train_tfms, val_tfms
