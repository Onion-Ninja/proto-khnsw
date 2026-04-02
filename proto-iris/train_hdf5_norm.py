from __future__ import print_function
import sys
import os
import numpy as np
import torch
import h5py
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prototypical_batch_sampler import PrototypicalBatchSampler
from src.prototypical_loss import prototypical_loss as loss_fn
from src.protonet import ProtoNet
from src.parser_util import get_parser


class IrisNormalizedHDF5Dataset(Dataset):

    def __init__(self, hdf5_path, allowed_classes=None, min_samples_per_class=5):

        with h5py.File(hdf5_path, "r") as f:
            images = f["images"][:]   # (N,128,512)
            masks = f["masks"][:]     # (N,128,512)
            labels = f["labels"][:]

        # decode labels
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]

        class_dict = defaultdict(list)

        for i, lbl in enumerate(labels):
            class_dict[lbl].append(i)

        valid_classes = [
            cls for cls, idxs in class_dict.items()
            if len(idxs) >= min_samples_per_class
        ]

        if allowed_classes is not None:
            allowed_classes = set(allowed_classes)
            valid_classes = [cls for cls in valid_classes if cls in allowed_classes]

        self.class_to_idx = {
            cls: i for i, cls in enumerate(sorted(valid_classes))
        }

        self.data = []

        for cls in valid_classes:
            for idx in class_dict[cls]:

                img = images[idx]
                mask = masks[idx]

                self.data.append(
                    (img, mask, self.class_to_idx[cls])
                )

        self.y = np.array([label for _, _, label in self.data])

        print(f"== Normalized Dataset: {len(valid_classes)} classes, {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img, mask, label = self.data[idx]

        img = img.astype(np.float32)

        #  Normalize pixel values
        img = img / 255.0

        #  OPTIONAL (recommended): apply mask
        mask = mask.astype(np.float32)
        img = img * mask

        #  Add channel dimension: (128,512) → (1,128,512)
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img), label



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


# ------------------ SAVE METRICS ------------------

def save_metrics(opt, train_loss, train_acc, val_acc, test_acc, dataset_name):

    filename = f"{opt.classes_per_it_tr}way_{opt.num_support_tr}shot_{dataset_name}.txt"
    filepath = os.path.join(opt.experiment_root, filename)

    with open(filepath, "w") as f:

        f.write("==== Experiment Details ====\n")
        f.write(f"N-way: {opt.classes_per_it_tr}\n")
        f.write(f"K-shot: {opt.num_support_tr}\n")
        f.write(f"Dataset: {dataset_name}\n\n")

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

    # 🔥 CHANGE HERE
    dataset = IrisNormalizedHDF5Dataset(
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

        train_loss_list.append(np.mean(epoch_loss))
        train_acc_list.append(np.mean(epoch_acc))

        print(f'Avg Train Acc: {np.mean(epoch_acc):.4f}')

        scheduler.step()

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
    print(f"\nTest Acc: {final_acc:.4f}")

    return final_acc


# ------------------ MAIN ------------------

def main():

    opt = get_parser().parse_args()
    init_seed(opt)

    # 🔥 CHANGE: normalized dataset
    CASIA_H5 = os.path.expanduser(
        "~/datasets/iris_db/CASIA_iris_thousand/worldcoin_outputs_npz/normalized.h5"
    )

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

    tr_loader = init_dataloader(CASIA_H5, opt, "train", class_splits)
    val_loader = init_dataloader(CASIA_H5, opt, "val", class_splits)
    test_loader = init_dataloader(CASIA_H5, opt, "test", class_splits)

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    emb_d = 64

    # 🔥 CHANGE: x_dim = 1
    model = ProtoNet(
        x_dim=1,
        hid_dim=64,
        z_dim=emb_d
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        gamma=opt.lr_scheduler_gamma,
        step_size=opt.lr_scheduler_step
    )

    best_state, train_loss, train_acc, val_acc = train(
        opt, tr_loader, model, optim, scheduler, val_loader
    )

    model.load_state_dict(best_state)

    test_acc = test(opt, test_loader, model)

    # 🔥 renamed
    dataset_name = f"casia_normalized"

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