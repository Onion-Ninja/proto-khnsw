import os
import numpy as np
import h5py
from tqdm import tqdm

# Inside h5 files, we will store as this where N is the number of templates:
# iris_codes  (N, 16, 256, 2)
# mask_codes  (N, 16, 256, 2)
# labels      (N)

DATASET_ROOT = os.path.expanduser(
    "~/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/templates"
)

OUTPUT_FILE = os.path.expanduser(
    "~/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/templates.h5"
)


def collect_files(root):

    files = []
    labels = []

    for cls in sorted(os.listdir(root)):

        cls_path = os.path.join(root, cls)

        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):

            if f.endswith(".npz"):

                files.append(os.path.join(cls_path, f))
                labels.append(cls)

    return files, labels


def convert():

    files, labels = collect_files(DATASET_ROOT)

    print("Total templates:", len(files))

    iris_codes = []
    mask_codes = []
    class_labels = []

    for path, label in tqdm(zip(files, labels), total=len(files)):

        try:
            with np.load(path, allow_pickle=False) as data:

                iris = data["iris_code"].astype(np.uint8)
                mask = data["mask_code"].astype(np.uint8)

            iris_codes.append(iris)
            mask_codes.append(mask)
            class_labels.append(label)

        except Exception as e:
            print("Skipping:", path)
            print(e)

    iris_codes = np.stack(iris_codes)
    mask_codes = np.stack(mask_codes)

    print("Saving HDF5 dataset...")

    with h5py.File(OUTPUT_FILE, "w") as f:

        f.create_dataset(
            "iris_codes",
            data=iris_codes,
            compression="gzip"
        )

        f.create_dataset(
            "mask_codes",
            data=mask_codes,
            compression="gzip"
        )

        f.create_dataset(
            "labels",
            data=np.array(class_labels, dtype="S")
        )

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    convert()