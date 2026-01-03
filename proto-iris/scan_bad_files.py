import os
import numpy as np

ROOT = "/home/nishkal/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/normalized"

bad_files = []

for cls in os.listdir(ROOT):
    cls_path = os.path.join(ROOT, cls)
    if not os.path.isdir(cls_path):
        continue

    for f in os.listdir(cls_path):
        if f.endswith(".npz"):
            path = os.path.join(cls_path, f)
            try:
                data = np.load(path, allow_pickle=False)
                _ = data["normalized_image"]
                _ = data["normalized_mask"]
            except Exception as e:
                bad_files.append(path)

print(f"Found {len(bad_files)} corrupted files")

i =1
for bf in bad_files:
    print(f"{i} : {bf}")
# # Optional: delete them
# for f in bad_files:
#     print("Removing:", f)
#     os.remove(f)
