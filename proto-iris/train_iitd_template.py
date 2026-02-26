# coding=utf-8
from __future__ import print_function
import sys
import os
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prototypical_batch_sampler import PrototypicalBatchSampler
from src.prototypical_loss import prototypical_loss as loss_fn
from src.protonet import ProtoNet
from src.parser_util import get_parser

from iitd_template_dataset import IITDTemplateDataset


# ------------------ SEED ------------------

def init_seed(opt):
    torch.backends.cudnn.enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


# ------------------ SPLITS ------------------

def get_valid_class_splits(root, min_samples=5, seed=0):
    rng = np.random.RandomState(seed)
    class_counts = defaultdict(int)

    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            if f.endswith(".npz"):
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
    val_classes = valid_classes[n_train:n_train + n_val]
    test_classes = valid_classes[n_train + n_val:]

    print(f"Class split → Train: {len(train_classes)}, "
          f"Val: {len(val_classes)}, Test: {len(test_classes)}")

    return train_classes, val_classes, test_classes


# ------------------ DATALOADER ------------------

def init_dataloader(opt, mode, class_splits):

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

    dataset = IITDTemplateDataset(
        root=opt.dataset_root,
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

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    best_acc = 0
    best_state = None

    for epoch in range(opt.epochs):
        print(f'=== Epoch: {epoch} ===')

        # ---------------- TRAIN ----------------
        model.train()

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

            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_train_loss = np.mean(train_loss[-opt.iterations:])
        avg_train_acc = np.mean(train_acc[-opt.iterations:])

        print(f'Avg Train Loss: {avg_train_loss:.4f}, '
              f'Avg Train Acc: {avg_train_acc:.4f}')

        scheduler.step()

        # ---------------- VALIDATION ----------------
        if val_loader is not None:
            model.eval()

            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                embeddings = model(x)
                loss, acc = loss_fn(
                    embeddings,
                    target=y,
                    n_support=opt.num_support_val
                )

                val_loss.append(loss.item())
                val_acc.append(acc.item())

            avg_val_acc = np.mean(val_acc[-opt.iterations:])
            print(f'Avg Val Acc: {avg_val_acc:.4f}')

            if avg_val_acc >= best_acc:
                best_acc = avg_val_acc
                best_state = model.state_dict()

    return best_state

# ------------------ TEST ------------------

def test(opt, test_loader, model):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model.eval()

    acc_list = []

    for _ in range(10):
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

    print(f"Test Acc: {np.mean(acc_list):.4f}")
    return np.mean(acc_list)


# ------------------ MAIN ------------------

def main():
    opt = get_parser().parse_args()
    os.makedirs(opt.experiment_root, exist_ok=True)

    init_seed(opt)

    train_cls, val_cls, test_cls = get_valid_class_splits(
        opt.dataset_root,
        min_samples=5,
        seed=opt.manual_seed
    )

    class_splits = {
        "train": train_cls,
        "val": val_cls,
        "test": test_cls
    }

    tr_loader = init_dataloader(opt, "train", class_splits)
    val_loader = init_dataloader(opt, "val", class_splits)
    test_loader = init_dataloader(opt, "test", class_splits)

    # IMPORTANT: x_dim=2 because templates have 2 channels
    model = ProtoNet(x_dim=2).to(
        'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    )

    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        gamma=opt.lr_scheduler_gamma,
        step_size=opt.lr_scheduler_step
    )

    best_state = train(opt, tr_loader, model, optim, scheduler, val_loader)

    print("Testing best model...")
    model.load_state_dict(best_state)
    test(opt, test_loader, model)


if __name__ == "__main__":
    main()