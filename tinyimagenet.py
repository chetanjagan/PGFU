"""
datasets/tinyimagenet.py
========================
PyTorch Dataset class for TinyImageNet-200.

Expected directory structure:
    root/
      train/
        classname/
          images/
            *.JPEG
      val/
        classname/
          images/
            *.JPEG
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetDataset(Dataset):
    """
    TinyImageNet-200 dataset loader.

    Args:
        root      : path to tiny-imagenet-200 directory
        split     : 'train' or 'val'
        transform : torchvision transforms
    """

    def __init__(self, root, split="train", transform=None):
        self.root      = root
        self.split     = split
        self.transform = transform
        self.samples   = []

        # Build class mapping from train directory
        train_dir  = os.path.join(root, "train")
        class_dirs = sorted(os.listdir(train_dir))
        self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        split_dir = os.path.join(
            root, "train" if split == "train" else "val")

        for cls in sorted(os.listdir(split_dir)):
            if cls not in self.class_to_idx:
                continue
            # Try images/ subdirectory first, then direct
            img_dir = os.path.join(split_dir, cls, "images")
            if not os.path.isdir(img_dir):
                img_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append(
                        (os.path.join(img_dir, fname),
                         self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_tinyimagenet_transforms():
    """Standard TinyImageNet transforms."""
    import torchvision.transforms as T
    mean = (0.4802, 0.4481, 0.3975)
    std  = (0.2770, 0.2691, 0.2821)

    train_transform = T.Compose([
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_transform, test_transform
