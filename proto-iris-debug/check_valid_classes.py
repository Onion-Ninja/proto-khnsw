# valid_classes_check.py

import os
from collections import defaultdict


def check_valid_classes(root, min_samples=5):
    """
    Safety check for IITD normalized dataset.
    Counts samples per class and reports how many
    classes satisfy the minimum sample requirement.
    """

    class_counts = defaultdict(int)

    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            if f.endswith("_norm.npz"):
                class_counts[cls] += 1

    total_classes = len(class_counts)

    valid_classes = {
        cls: cnt for cls, cnt in class_counts.items()
        if cnt >= min_samples
    }

    invalid_classes = {
        cls: cnt for cls, cnt in class_counts.items()
        if cnt < min_samples
    }

    print("====== IITD Normalized Dataset Check ======")
    print(f"Dataset root          : {root}")
    print(f"Total classes found   : {total_classes}")
    print(f"Min samples required  : {min_samples}")
    print(f"Valid classes (>= {min_samples}) : {len(valid_classes)}")
    print(f"Rejected classes (< {min_samples}) : {len(invalid_classes)}")
    print()

    print("---- Sample count distribution ----")
    dist = defaultdict(int)
    for cnt in class_counts.values():
        dist[cnt] += 1

    for cnt in sorted(dist.keys()):
        print(f"{cnt} samples : {dist[cnt]} classes")

    print("\n---- Example valid classes ----")
    for cls, cnt in list(valid_classes.items())[:10]:
        print(f"{cls} : {cnt}")

    print("\n---- Example rejected classes ----")
    for cls, cnt in list(invalid_classes.items())[:10]:
        print(f"{cls} : {cnt}")

    return valid_classes, invalid_classes


if __name__ == "__main__":
    ROOT = os.path.expanduser(
        "/home/nishkal/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/normalized"
    )

    check_valid_classes(ROOT, min_samples=5)
