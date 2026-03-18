"""
Convert NPZ iris template files to JSON format.

Usage:
    python npz_to_json.py --dataset <dataset_name>

Valid dataset names:
    CASIA_iris_thousand (default)
    CASIA_v1
    IITD_v1
    #   UBIRIS_v2 (not available yet)
"""

import os
import json
import numpy as np
import argparse
import gc


# ----------------------------- #
# Convert a single NPZ → JSON
# ----------------------------- #

def convert_npz_to_json(npz_path, json_path):

    try:
        with np.load(npz_path, allow_pickle=False) as data:

            # Copy arrays immediately (avoids numpy zip issues)
            iris_code = np.array(data["iris_code"], dtype=np.uint8)
            mask_code = np.array(data["mask_code"], dtype=np.uint8)

        output = {
            "iris_code": iris_code.tolist(),
            "mask_code": mask_code.tolist(),
            "shape": list(iris_code.shape)
        }

        with open(json_path, "w") as f:
            json.dump(output, f, separators=(",", ":"))

    except Exception as e:
        print(f"[ERROR] Failed to convert: {npz_path}")
        print(e)


# ----------------------------- #
# Convert an entire dataset
# ----------------------------- #

def convert_dataset(dataset_name):

    base_path = os.path.expanduser(
        f"~/datasets/iris_db/{dataset_name}/worldcoin_outputs_npz"
    )

    input_root = os.path.join(base_path, "templates")
    output_root = os.path.join(base_path, "templates_json")

    print("Input directory :", input_root)
    print("Output directory:", output_root)

    if not os.path.exists(input_root):
        raise ValueError(f"Dataset path not found: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    total_files = 0

    for root, dirs, files in os.walk(input_root):

        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)

        os.makedirs(output_dir, exist_ok=True)

        for file in files:

            if not file.endswith(".npz"):
                continue

            npz_path = os.path.join(root, file)

            json_name = file.replace(".npz", ".json")
            json_path = os.path.join(output_dir, json_name)

            # Skip already converted files
            if os.path.exists(json_path):
                continue

            convert_npz_to_json(npz_path, json_path)

            total_files += 1

            if total_files % 100 == 0:
                print(f"Converted {total_files} files...")
                gc.collect()

    print("\nConversion complete.")
    print(f"Total files converted: {total_files}")


# ----------------------------- #
# Main
# ----------------------------- #

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="CASIA_iris_thousand",
        choices=[
            "CASIA_iris_thousand",
            "CASIA_v1",
            "IITD_v1",
            "UBIRIS_v2"
        ],
        help="Dataset name"
    )

    args = parser.parse_args()

    convert_dataset(args.dataset)


if __name__ == "__main__":
    main()