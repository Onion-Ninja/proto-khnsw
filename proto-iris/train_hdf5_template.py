# coding=utf-8
from __future__ import print_function
import sys
import os
import numpy as np
import torch
import h5py
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prototypical_batch_sampler import PrototypicalBatchSampler
from src.prototypical_loss import prototypical_loss as loss_fn
from src.protonet import ProtoNet
from src.parser_util import get_parser

from iris_hdf5_dataset import IrisTemplateHDF5Dataset


# ------------------ SEED ------------------

def init_seed(opt):
    torch.backends.cudnn.enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


# ------------------ CLASS SPLITS ------------------

def get_valid_class_splits(hdf5_path, min_samples=5, seed=0):

    rng = np.random.RandomState(seed)

    with h5py.File(hdf5_path, "r") as f:
        labels = [l.decode() for l in f["labels"][:]]

    class_counts = defaultdict(int)
    for cls in labels:
        class_counts[cls] += 1

    valid_classes = [
        cls for cls, cnt in class_counts.items()
        if cnt >= min_samples
    ]

    rng.shuffle(valid_classes)

    n = len(valid_classes)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_classes = valid_classes[:n_train]
    val_classes = valid_classes[n_train:n_train+n_val]
    test_classes = valid_classes[n_train+n_val:]

    print(
        f"Class split → Train:{len(train_classes)} "
        f"Val:{len(val_classes)} Test:{len(test_classes)}"
    )

    return train_classes, val_classes, test_classes


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

# ------------------ DATALOADER ------------------

def init_dataloader(hdf5_path, opt, mode, class_splits):

    if mode == "train":
        classes = class_splits["train"]
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr

    elif mode == "val":
        classes = class_splits["val"]
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    else:
        classes = class_splits["test"]
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    dataset = IrisTemplateHDF5Dataset(
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

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        pin_memory=True
    )


# ------------------ TRAIN ------------------

def train(opt, tr_loader, model, optim, scheduler, val_loader=None):

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    train_loss_list = [] 
    train_acc_list = []
    val_acc_list = []

    best_acc = 0
    best_state = None

    print("Training on device →", device)

    for epoch in range(opt.epochs):

        print(f'=== Epoch: {epoch} ===')

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

        print(
            f'Avg Train Loss: {np.mean(avg_loss):.4f}, '
            f'Avg Train Acc: {np.mean(avg_acc ):.4f}'
        )

        scheduler.step()

        # ---------- VALIDATION ----------
        if val_loader is not None:

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

            avg_val_acc = np.mean(val_acc)
            val_acc_list.append(avg_val_acc)

            print(f'Avg Val Acc: {avg_val_acc:.4f}')

            if avg_val_acc >= best_acc:
                best_acc = avg_val_acc
                best_state = model.state_dict()

    return best_state, train_loss_list, train_acc_list, val_acc_list


# ------------------ TEST ------------------

def test(opt, test_loader, model):

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

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

    # Add embedding dim manually if not in parser
    if not hasattr(opt, "embedding_dim"):
        opt.embedding_dim = 64

    os.makedirs(opt.experiment_root, exist_ok=True)

    init_seed(opt)

    # ---------- DATA PATH ----------
    CASIA_H5 = "C:/Users/tbmmd/OneDrive/Desktop/10th Sem/mtp/datasets/casia_iris_thousand/templates.h5"

    print("\nMode: SAME DATASET (CASIA → CASIA)")

    # ---------- SPLITS ----------
    train_cls, val_cls, test_cls = get_valid_class_splits(
        CASIA_H5,
        min_samples=5,
        seed=opt.manual_seed
    )

    class_splits = {
        "train": train_cls,
        "val": val_cls,
        "test": test_cls
    }

    # ---------- LOADERS ----------
    tr_loader = init_dataloader(CASIA_H5, opt, "train", class_splits)
    val_loader = init_dataloader(CASIA_H5, opt, "val", class_splits)
    test_loader = init_dataloader(CASIA_H5, opt, "test", class_splits)

    # ---------- MODEL ----------
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    model = ProtoNet(
        x_dim=2,
        hid_dim=64,
        z_dim=opt.embedding_dim
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        gamma=opt.lr_scheduler_gamma,
        step_size=opt.lr_scheduler_step
    )

    # ---------- TRAIN ----------
    best_state, train_loss, train_acc, val_acc = train(
        opt, tr_loader, model, optim, scheduler, val_loader
    )

    # ---------- TEST ----------
    print("\n🔥 Testing on CASIA (Same Dataset)...")

    model.load_state_dict(best_state)

    test_acc = test(opt, test_loader, model)

    # ---------- SAVE ----------
    dataset_name = f"casia_dim{opt.embedding_dim}"

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