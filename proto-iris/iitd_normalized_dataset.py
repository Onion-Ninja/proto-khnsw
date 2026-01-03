import os
import numpy as np
import torch
from torch.utils.data import Dataset

class IITDNormalizedDataset(Dataset):
    """
    Loads normalized iris images from .npz files
    """

    def __init__(self, root):
        """
        root:
        ~/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/normalized
        """
        self.samples = []
        self.labels = []

        class_dirs = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

        for cls in class_dirs:
            cls_path = os.path.join(root, cls)
            for f in os.listdir(cls_path):
                if f.endswith("_norm.npz"):
                    self.samples.append(os.path.join(cls_path, f))
                    self.labels.append(self.class_to_idx[cls])

        self.y = np.array(self.labels)

        print(f"== Normalized Dataset: {len(self.samples)} images")
        print(f"== Normalized Dataset: {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=False)

        img = data["normalized_image"].astype(np.float32)
        mask = data["normalized_mask"].astype(np.float32)

        # Apply mask (important for iris!)
        img = img * mask

        # Normalize
        img = img / 255.0 if img.max() > 1 else img

        # [1, H, W]
        img = torch.from_numpy(img).unsqueeze(0)

        label = self.labels[idx]
        return img, label
