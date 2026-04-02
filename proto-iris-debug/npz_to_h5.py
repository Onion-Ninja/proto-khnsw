import os
import numpy as np
import h5py
from tqdm import tqdm
import cv2


# ===================== CONFIG =====================

ROOT = os.path.expanduser("~/datasets/iris_db/CASIA_iris_thousand/worldcoin_outputs_npz")

PATHS = {
    "template": os.path.join(ROOT, "templates"),
    "normalized": os.path.join(ROOT, "normalized"),
    "segmentation": os.path.join(ROOT, "segmentation"),
}

OUTPUTS = {
    "template": os.path.join(ROOT, "templates.h5"),
    "normalized": os.path.join(ROOT, "normalized.h5"),
    "segmentation": os.path.join(ROOT, "segmentation.h5"),
}

# ===================== COMMON =====================

def collect_files(root, suffix):
    files, labels = [], []

    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)

        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            if f.endswith(suffix):
                files.append(os.path.join(cls_path, f))
                labels.append(cls)

    return files, labels


# ===================== TEMPLATE =====================

def convert_templates():
    root = PATHS["template"]
    out = OUTPUTS["template"]

    files, labels = collect_files(root, ".npz")
    print("Total templates:", len(files))

    iris_codes, mask_codes, class_labels = [], [], []

    for path, label in tqdm(zip(files, labels), total=len(files)):
        try:
            data = np.load(path, allow_pickle=False)

            iris = data["iris_code"].astype(np.uint8)
            mask = data["mask_code"].astype(np.uint8)

            iris_codes.append(iris)
            mask_codes.append(mask)
            class_labels.append(label)

        except Exception as e:
            print("Skipping:", path, e)

    iris_codes = np.stack(iris_codes)
    mask_codes = np.stack(mask_codes)

    with h5py.File(out, "w") as f:
        f.create_dataset("iris_codes", data=iris_codes, compression="gzip")
        f.create_dataset("mask_codes", data=mask_codes, compression="gzip")
        f.create_dataset("labels", data=np.array(class_labels, dtype="S"))

    print("Saved:", out)


# ===================== NORMALIZED =====================

def convert_normalized():
    root = PATHS["normalized"]
    out = OUTPUTS["normalized"]

    files, labels = collect_files(root, "_norm.npz")
    print("Total normalized:", len(files))

    images, masks, class_labels = [], [], []
    k =0
    for path, label in tqdm(zip(files, labels), total=len(files)):
        try:
            
            # if "normalized_image" not in data:
            #     print("Keys present:", data.files)
            #     print("File:", path)
            #     continue

            data = np.load(path, allow_pickle=True)

            # Case 1: Standard format
            if "normalized_image" in data:
                img = data["normalized_image"].astype(np.uint8)
                mask = data["normalized_mask"].astype(bool)

            # Case 2: Nested format (CASIA issue)
            elif "normalization" in data:
                norm = data["normalization"].item()  # extract dict

                img = norm["normalized_image"].astype(np.uint8)
                mask = norm["normalized_mask"].astype(bool)

            else:
                print("Unknown format:", path, data.files)
                continue
            # img = data["normalized_image"].astype(np.uint8)
            # mask = data["normalized_mask"].astype(bool)

            if img.shape != (128, 512):
                continue

            images.append(img)
            masks.append(mask)
            class_labels.append(label)

        except Exception as e:
            print("Skipping:", path, e)

    images = np.stack(images)
    masks = np.stack(masks)

    with h5py.File(out, "w") as f:
        f.create_dataset("images", data=images, compression="gzip")
        f.create_dataset("masks", data=masks, compression="gzip")
        f.create_dataset("labels", data=np.array(class_labels, dtype="S"))

    print("Saved:", out)


# ===================== SEGMENTATION =====================

def convert_segmentation():
    root = PATHS["segmentation"]
    out = OUTPUTS["segmentation"]

    files, labels = collect_files(root, "_seg.npz")
    print("Total segmentation:", len(files))

    seg_maps, class_labels = [], []

    valid_count = 0

    for path, label in tqdm(zip(files, labels), total=len(files)):
        try:
            data = np.load(path, allow_pickle=False)

            if "predictions" not in data:
                print("Missing predictions:", path)
                continue

            seg = data["predictions"].astype(np.float32)

            # DEBUG (only first few)
            if valid_count < 5:
                print("Shape:", seg.shape)

            # Flexible shape handling
            if len(seg.shape) != 3:
                continue

            # Optional: resize if needed
            if seg.shape != (240, 320, 4):
                # skip OR resize
                seg_resized = np.zeros((240, 320, 4), dtype=np.float32)

                for i in range(4):
                    seg_resized[:, :, i] = cv2.resize(seg[:, :, i], (320, 240))

                seg = seg_resized


            seg_maps.append(seg)
            class_labels.append(label)
            valid_count += 1

        except Exception as e:
            print("Skipping:", path, e)

    seg_maps = np.stack(seg_maps)

    with h5py.File(out, "w") as f:
        f.create_dataset("segmentation_maps", data=seg_maps, compression="gzip")
        f.create_dataset("labels", data=np.array(class_labels, dtype="S"))

    print("Saved:", out)


# ===================== MAIN =====================

if __name__ == "__main__":

    MODE = "normalized"  
    # OPTIONS: "template", "normalized", "segmentation"

    if MODE == "template":
        convert_templates()

    elif MODE == "normalized":
        convert_normalized()

    elif MODE == "segmentation":
        convert_segmentation()

    else:
        raise ValueError("Invalid MODE")