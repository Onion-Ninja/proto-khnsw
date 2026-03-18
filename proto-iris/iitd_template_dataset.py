import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class IITDTemplateDataset(Dataset):
    """
    Dataset for IITD iris Gabor templates (iris_code + mask_code)
    """

    def __init__(self, root, allowed_classes, min_samples_per_class=5):
        self.samples = []
        self.labels = []

        class_files = defaultdict(list)

        # Collect template files
        for cls in allowed_classes:
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue

            for f in os.listdir(cls_path):
                if f.endswith(".npz"):
                    class_files[cls].append(os.path.join(cls_path, f))

        # Filter classes with enough samples
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

        print(f"== Template Dataset: {len(valid_classes)} classes, "
              f"{len(self.samples)} templates")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=False)

        iris_code = np.array(data["iris_code"], dtype=np.float32, copy=True)
        mask_code = np.array(data["mask_code"], dtype=np.float32, copy=True)
        data.close()

        # Apply mask
        iris_code = iris_code * mask_code

        # (16, 256, 2) → (2, 16, 256)
        iris_code = np.transpose(iris_code, (2, 0, 1))

        iris_code = torch.from_numpy(iris_code)

        label = self.labels[idx]

        return iris_code, label