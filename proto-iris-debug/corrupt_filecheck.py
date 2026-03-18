import numpy as np
import os

root = "/home/nishkal/datasets/iris_db/CASIA_iris_thousand/worldcoin_outputs_npz/templates"

count = 0

for dirpath, _, files in os.walk(root):
    for f in files:
        if f.endswith(".npz"):
            path = os.path.join(dirpath, f)
            try:
                with np.load(path, allow_pickle=False) as data:
                    _ = data["iris_code"]
                    _ = data["mask_code"]
                count += 1
                if count % 100 == 0:
                    print("checked", count)
            except Exception as e:
                print("BAD FILE:", path)
                print(e)