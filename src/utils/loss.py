from collections import Counter
import torch
from torch.utils.data import Subset

def get_class_weights(dataset):
    # Handle Subset
    if isinstance(dataset, Subset):
        targets = []
        for idx in dataset.indices:
            _, label = dataset.dataset[idx]
            targets.append(label)
    else:
        # If dataset has .targets (ImageFolder style)
        if hasattr(dataset, "targets"):
            targets = dataset.targets
        else:
            # Generic fallback (TensorDataset / custom dataset)
            targets = []
            for i in range(len(dataset)):
                _, label = dataset[i]
                targets.append(label)

    targets = torch.tensor(targets)
    num_classes = len(torch.unique(targets))
    counts = Counter(targets.tolist())
    total = len(targets)

    weights = torch.zeros(num_classes)

    for cls in range(num_classes):
        weights[cls] = total / (num_classes * counts[cls])

    return weights