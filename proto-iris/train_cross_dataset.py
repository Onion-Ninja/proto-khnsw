# coding=utf-8
from __future__ import print_function
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prototypical_batch_sampler import PrototypicalBatchSampler
from src.prototypical_loss import prototypical_loss as loss_fn
from src.protonet import ProtoNet
from src.parser_util import get_parser


# ------------------ SEED ------------------

def init_seed(opt):
    torch.backends.cudnn.enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


# ------------------ HDF5 DATASET ------------------

class IrisHDF5Dataset(Dataset):

    def __init__(self, hdf5_path, allowed_classes=None, min_samples_per_class=5):

        with h5py.File(hdf5_path, "r") as f:
            iris_codes = f["iris_codes"][:]
            mask_codes = f["mask_codes"][:]
            labels = f["labels"][:]

        # 🔥 FIX: decode labels properly
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]

        class_dict = defaultdict(list)

        for i, lbl in enumerate(labels):
            class_dict[lbl].append(i)

        valid_classes = [
            cls for cls, idxs in class_dict.items()
            if len(idxs) >= min_samples_per_class
        ]

        # 🔥 FIX: ensure same type comparison
        if allowed_classes is not None:
            allowed_classes = set(allowed_classes)
            valid_classes = [cls for cls in valid_classes if cls in allowed_classes]

        self.class_to_idx = {
            cls: i for i, cls in enumerate(sorted(valid_classes))
        }

        self.data = []

        for cls in valid_classes:
            for idx in class_dict[cls]:
                self.data.append(
                    (iris_codes[idx], mask_codes[idx], self.class_to_idx[cls])
                )

        self.y = np.array([label for _, _, label in self.data])

        print(f"== HDF5 Dataset: {len(valid_classes)} classes, {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        iris_code, mask_code, label = self.data[idx]

        iris_code = iris_code.astype(np.float32)
        mask_code = mask_code.astype(np.float32)

        iris_code = iris_code * mask_code

        # (16,256,2) → (2,16,256)
        iris_code = np.transpose(iris_code, (2, 0, 1))

        return torch.from_numpy(iris_code), label
    
# ------------------ SPLITS ------------------

def get_hdf5_splits(hdf5_path, min_samples=5, seed=0):

    rng = np.random.RandomState(seed)

    with h5py.File(hdf5_path, "r") as f:
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in f["labels"][:]]

    class_counts = defaultdict(int)
    for lbl in labels:
        class_counts[lbl] += 1

    valid_classes = [
        cls for cls, cnt in class_counts.items()
        if cnt >= min_samples
    ]

    rng.shuffle(valid_classes)

    n = len(valid_classes)
    n_train = int(0.8 * n)

    train_classes = valid_classes[:n_train]
    val_classes = valid_classes[n_train:]

    print(f"Train: {len(train_classes)}, Val: {len(val_classes)}")

    return train_classes, val_classes


# ------------------ DATALOADER ------------------

def get_loader(hdf5_path, classes, opt, mode):

    if mode == "train":
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    dataset = IrisHDF5Dataset(
        hdf5_path=hdf5_path,
        allowed_classes=classes,
        min_samples_per_class=5
    )

    sampler = PrototypicalBatchSampler(
        labels=dataset.y,
        classes_per_it=classes_per_it,
        num_samples=num_samples,
        iterations=opt.iterations
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        pin_memory=True,
        num_workers=0
    )


def save_metrics(opt, train_loss, train_acc, val_acc, test_acc, dataset_name):

    n = opt.classes_per_it_tr
    k = opt.num_support_tr

    filename = f"{n}way_{k}shot_{dataset_name}.txt"
    filepath = os.path.join(opt.experiment_root, filename)

    with open(filepath, "w") as f:

        f.write("==== Experiment Details ====\n")
        f.write(f"N-way: {n}\n")
        f.write(f"K-shot: {k}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Epochs: {opt.epochs}\n")
        f.write("\n")

        f.write("==== Training Loss ====\n")
        for val in train_loss:
            f.write(f"{val}\n")

        f.write("\n==== Training Accuracy ====\n")
        for val in train_acc:
            f.write(f"{val}\n")

        f.write("\n==== Validation Accuracy ====\n")
        for val in val_acc:
            f.write(f"{val}\n")

        f.write("\n==== Final Test Accuracy ====\n")
        f.write(f"{test_acc}\n")

    print(f"\n Metrics saved to: {filepath}")
# ------------------ TRAIN ------------------

# def train(opt, tr_loader, model, optim, scheduler, val_loader):

#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     print("Training on device →", device)

#     best_acc = 0
#     best_state = None

#     for epoch in range(opt.epochs):
#         print(f"\n=== Epoch {epoch} ===")

#         model.train()
#         train_acc = []

#         for batch in tqdm(tr_loader):
#             x, y = batch
#             x, y = x.to(device), y.to(device)

#             optim.zero_grad()

#             embeddings = model(x)
#             loss, acc = loss_fn(
#                 embeddings,
#                 target=y,
#                 n_support=opt.num_support_tr
#             )

#             loss.backward()
#             optim.step()

#             train_acc.append(acc.item())

#         print(f"Train Acc: {np.mean(train_acc):.4f}")
#         scheduler.step()

#         # VALIDATION
#         model.eval()
#         val_acc = []

#         for batch in val_loader:
#             x, y = batch
#             x, y = x.to(device), y.to(device)

#             embeddings = model(x)
#             _, acc = loss_fn(
#                 embeddings,
#                 target=y,
#                 n_support=opt.num_support_val
#             )

#             val_acc.append(acc.item())

#         avg_val = np.mean(val_acc)
#         print(f"Val Acc: {avg_val:.4f}")

#         if avg_val > best_acc:
#             best_acc = avg_val
#             best_state = model.state_dict()

#     return best_state

def train(opt, tr_loader, model, optim, scheduler, val_loader):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Training on device →", device)

    best_acc = 0
    best_state = None

    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(opt.epochs):

        print(f"\n=== Epoch {epoch} ===")

        model.train()

        epoch_loss = []
        epoch_acc = []

        for batch in tqdm(tr_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            optim.zero_grad()

            embeddings = model(x)

            loss, acc = loss_fn(
                embeddings,
                target=y,
                n_support=opt.num_support_tr
            )

            loss.backward()
            optim.step()

            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())

        avg_loss = np.mean(epoch_loss)
        avg_acc = np.mean(epoch_acc)

        train_loss_list.append(avg_loss)
        train_acc_list.append(avg_acc)

        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}")

        scheduler.step()

        # VALIDATION
        model.eval()

        val_acc = []

        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            embeddings = model(x)

            _, acc = loss_fn(
                embeddings,
                target=y,
                n_support=opt.num_support_val
            )

            val_acc.append(acc.item())

        avg_val = np.mean(val_acc)
        val_acc_list.append(avg_val)

        print(f"Val Acc: {avg_val:.4f}")

        if avg_val >= best_acc:
            best_acc = avg_val
            best_state = model.state_dict()

    return best_state, train_loss_list, train_acc_list, val_acc_list

# ------------------ TEST ------------------

# def test(opt, test_loader, model):

#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     model.eval()

#     acc_list = []

#     for _ in range(20):
#         for batch in test_loader:
#             x, y = batch
#             x, y = x.to(device), y.to(device)

#             embeddings = model(x)
#             _, acc = loss_fn(
#                 embeddings,
#                 target=y,
#                 n_support=opt.num_support_val
#             )

#             acc_list.append(acc.item())

#     print(f"\n🔥 Cross-Dataset Test Acc: {np.mean(acc_list):.4f}")

def test(opt, test_loader, model):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()

    acc_list = []

    for _ in range(20):
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            embeddings = model(x)

            _, acc = loss_fn(
                embeddings,
                target=y,
                n_support=opt.num_support_val
            )

            acc_list.append(acc.item())

    final_acc = np.mean(acc_list)

    print(f"\n🔥 Cross-Dataset Test Acc: {final_acc:.4f}")

    return final_acc


# ------------------ MAIN ------------------

def main():

    opt = get_parser().parse_args()
    init_seed(opt)

    # ----------- PATHS (CHANGE THESE) -----------
    IITD_H5 = os.path.expanduser(
        "~/datasets/iris_db/IITD_v1/worldcoin_outputs_npz/templates.h5"
    )

    CASIA_H5 = os.path.expanduser(
        "~/datasets/iris_db/CASIA_iris_thousand/worldcoin_outputs_npz/templates.h5"
    )

    # ----------- SPLITS -----------
    train_cls, val_cls = get_hdf5_splits(CASIA_H5)

    # ----------- LOADERS -----------
    tr_loader = get_loader(CASIA_H5, train_cls, opt, "train")
    val_loader = get_loader(CASIA_H5, val_cls, opt, "val")

    # TEST = ALL CLASSES (CASIA)
    test_dataset = IrisHDF5Dataset(
        hdf5_path=IITD_H5,
        allowed_classes=None,
        min_samples_per_class=5
    )

    test_sampler = PrototypicalBatchSampler(
        labels=test_dataset.y,
        classes_per_it=opt.classes_per_it_val,
        num_samples=opt.num_support_val + opt.num_query_val,
        iterations=opt.iterations
    )

    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        pin_memory=True,
        num_workers=0
    )

    # ----------- MODEL -----------
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = ProtoNet(x_dim=2).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        gamma=opt.lr_scheduler_gamma,
        step_size=opt.lr_scheduler_step
    )

    # ----------- TRAIN -----------
    best_state, train_loss, train_acc, val_acc = train(
    opt, tr_loader, model, optim, scheduler, val_loader
)


    # ----------- TEST -----------
    print("\nTesting on IITD (cross-dataset)...")
    model.load_state_dict(best_state)
    test_acc = test(opt, test_loader, model)

    dataset_name = "casia_to_iitd"   # you can make this dynamic later

    save_metrics(
        opt,
        train_loss,
        train_acc,
        val_acc,
        test_acc,
        dataset_name
    )


if __name__ == "__main__":
    main()