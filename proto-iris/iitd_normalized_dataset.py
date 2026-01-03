import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class IITDNormalizedDataset(Dataset):
    def __init__(self, root, allowed_classes, min_samples_per_class=5):
        """
        root:
            path to normalized npz files
        allowed_classes:
            list of class names (e.g. ['1_L', '1_R', ...])
        """
        self.samples = []
        self.labels = []

        # Collect files per class
        class_files = defaultdict(list)

        for cls in allowed_classes:
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue

            for f in os.listdir(cls_path):
                if f.endswith("_norm.npz"):
                    class_files[cls].append(os.path.join(cls_path, f))

        # Safety check (should already be filtered)
        valid_classes = {
            cls: files
            for cls, files in class_files.items()
            if len(files) >= min_samples_per_class
        }

        self.class_to_idx = {
            cls: i for i, cls in enumerate(sorted(valid_classes.keys()))
        }

        for cls, files in valid_classes.items():
            for f in files:
                self.samples.append(f)
                self.labels.append(self.class_to_idx[cls])

        self.y = np.array(self.labels)

        print(
            f"== Dataset split: {len(valid_classes)} classes, "
            f"{len(self.samples)} images"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=False, mmap_mode=None)

        img  = np.array(data["normalized_image"], dtype=np.float32, copy=True)
        mask = np.array(data["normalized_mask"], dtype=np.float32, copy=True)

        data.close()  # IMPORTANT: close zip file explicitly
        
        img = img * mask
        if img.max() > 1:
            img /= 255.0

        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        label = self.labels[idx]

        return img, label
