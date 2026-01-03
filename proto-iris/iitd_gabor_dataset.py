import os
import numpy as np
import torch
from torch.utils.data import Dataset

class IITDGaborTemplateDataset(Dataset):
    """
    Loads Gabor iris codes + masks from .npz
    """

    def __init__(self, root):
        """
        root:
        ~/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/templates
        """
        self.samples = []
        self.labels = []

        class_dirs = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

        for cls in class_dirs:
            cls_path = os.path.join(root, cls)
            for f in os.listdir(cls_path):
                if f.endswith(".npz"):
                    self.samples.append(os.path.join(cls_path, f))
                    self.labels.append(self.class_to_idx[cls])

        self.y = np.array(self.labels)

        print(f"== Gabor Dataset: {len(self.samples)} templates")
        print(f"== Gabor Dataset: {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=False)

        iris_code = data["iris_code"].astype(np.float32)
        mask_code = data["mask_code"].astype(np.float32)

        # Mask invalid bits
        iris_code = iris_code * mask_code

        # Flatten for ProtoNet embedding
        iris_code = torch.from_numpy(iris_code).view(-1)

        label = self.labels[idx]
        return iris_code, label
