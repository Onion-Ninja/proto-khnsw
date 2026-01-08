import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class IITDGaborTemplateDataset(Dataset):
    """
    Loads Gabor iris codes + masks from .npz files
    (ProtoNet-compatible, identity-disjoint, IO-safe)
    """

    def __init__(self, root, allowed_classes, min_samples_per_class=5):
        """
        root:
            path to templates npz files
            e.g. ~/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/templates

        allowed_classes:
            list of class names (e.g. ['1_L', '1_R', ...])

        min_samples_per_class:
            safety threshold (≥ 5 samples)
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
                if f.endswith(".npz"):
                    class_files[cls].append(os.path.join(cls_path, f))

        # Safety check (mirrors normalized dataset)
        valid_classes = {
            cls: files
            for cls, files in class_files.items()
            if len(files) >= min_samples_per_class
        }

        print(f"Total classes: {len(class_files)}")
        print(f"Valid classes: {len(valid_classes)}")

        # Assign class indices deterministically
        self.class_to_idx = {
            cls: i for i, cls in enumerate(sorted(valid_classes.keys()))
        }

        # Flatten samples
        for cls, files in valid_classes.items():
            for f in files:
                self.samples.append(f)
                self.labels.append(self.class_to_idx[cls])

        # REQUIRED by PrototypicalBatchSampler
        self.y = np.array(self.labels)

        print(
            f"== Gabor Dataset split: {len(valid_classes)} classes, "
            f"{len(self.samples)} templates"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        # Safe, eager npz loading (same as normalized dataset)
        data = np.load(path, allow_pickle=False, mmap_mode=None)

        iris_code = np.array(
            data["iris_code"], dtype=np.float32, copy=True
        )
        mask_code = np.array(
            data["mask_code"], dtype=np.float32, copy=True
        )

        data.close()  # IMPORTANT

        # Apply mask (domain-consistent)
        iris_code = iris_code * mask_code

        # Flatten to 1D vector for ProtoNet
        iris_code = torch.from_numpy(iris_code).view(-1)

        label = self.labels[idx]
        return iris_code, label
