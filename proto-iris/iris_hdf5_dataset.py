import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class IrisTemplateHDF5Dataset(Dataset):
    """
    Dataset for iris templates stored in HDF5 format
    """

    def __init__(self, hdf5_path, allowed_classes=None, min_samples_per_class=5):

        self.file = h5py.File(hdf5_path, "r")

        self.iris_codes = self.file["iris_codes"]
        self.mask_codes = self.file["mask_codes"]
        self.labels = self.file["labels"]

        self.samples = []
        self.targets = []

        class_files = defaultdict(list)

        # Collect class indices
        for idx, label in enumerate(self.labels):

            cls = label.decode()

            class_files[cls].append(idx)

        # Filter classes
        valid_classes = {
            cls: indices
            for cls, indices in class_files.items()
            if len(indices) >= min_samples_per_class
        }

        if allowed_classes is not None:

            valid_classes = {
                cls: valid_classes[cls]
                for cls in allowed_classes
                if cls in valid_classes
            }

        self.class_to_idx = {
            cls: i for i, cls in enumerate(sorted(valid_classes.keys()))
        }

        for cls, indices in valid_classes.items():

            for idx in indices:

                self.samples.append(idx)
                self.targets.append(self.class_to_idx[cls])

        self.y = np.array(self.targets)

        print(
            f"== HDF5 Dataset: {len(valid_classes)} classes, "
            f"{len(self.samples)} templates"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        real_idx = self.samples[idx]

        iris_code = self.iris_codes[real_idx]
        mask_code = self.mask_codes[real_idx]

        iris_code = iris_code.astype(np.float32) * mask_code.astype(np.float32)

        # (16,256,2) → (2,16,256)
        iris_code = np.transpose(iris_code, (2, 0, 1))

        iris_code = torch.from_numpy(iris_code)

        label = self.targets[idx]

        return iris_code, label