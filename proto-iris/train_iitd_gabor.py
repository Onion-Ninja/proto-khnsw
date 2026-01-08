# coding=utf-8
from __future__ import print_function
import sys
from tqdm import tqdm
import numpy as np
import torch
import os
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prototypical_batch_sampler import PrototypicalBatchSampler
from src.prototypical_loss import prototypical_loss as loss_fn
from src.protonet_gabor import ProtoNetGabor
from src.parser_util import get_parser

from iitd_gabor_template_dataset import IITDGaborTemplateDataset


# ------------------ SEED ------------------

def init_seed(opt):
    torch.backends.cudnn.enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


# ------------------ CLASS SPLIT ------------------

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
    n_val   = int(0.15 * n)

    train = valid_classes[:n_train]
    val   = valid_classes[n_train:n_train + n_val]
    test  = valid_classes[n_train + n_val:]

    print(
        f"Class split → Train: {len(train)}, "
        f"Val: {len(val)}, Test: {len(test)}"
    )

    return train, val, test


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

    dataset = IITDGaborTemplateDataset(
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
        num_workers=0,
        pin_memory=False
    )


# ------------------ MODEL ------------------

def init_protonet(opt, dataset):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    # Infer input dimension from one sample
    sample, _ = dataset[0]
    input_dim = sample.numel()

    model = ProtoNetGabor(input_dim=input_dim).to(device)
    print(f"Gabor input dim: {input_dim}")

    return model


def init_optim(opt, model):
    return torch.optim.Adam(model.parameters(), lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    return torch.optim.lr_scheduler.StepLR(
        optim,
        gamma=opt.lr_scheduler_gamma,
        step_size=opt.lr_scheduler_step
    )


# ------------------ TRAIN ------------------

def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    best_acc = 0
    best_state = None

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')

    for epoch in range(opt.epochs):
        print(f'=== Epoch: {epoch} ===')
        model.train()

        for x, y in tqdm(tr_dataloader):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()

            emb = model(x)
            loss, acc = loss_fn(emb, y, opt.num_support_tr)

            loss.backward()
            optim.step()

            train_loss.append(loss.item())
            train_acc.append(acc.item())

        print(
            f'Avg Train Acc: {np.mean(train_acc[-opt.iterations:]):.4f}'
        )

        lr_scheduler.step()

        model.eval()
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)
            emb = model(x)
            loss, acc = loss_fn(emb, y, opt.num_support_val)

            val_loss.append(loss.item())
            val_acc.append(acc.item())

        avg_val_acc = np.mean(val_acc[-opt.iterations:])
        print(f'Avg Val Acc: {avg_val_acc:.4f}')

        if avg_val_acc >= best_acc:
            best_acc = avg_val_acc
            best_state = model.state_dict()
            torch.save(best_state, best_model_path)

    return best_state


# ------------------ TEST ------------------

def test(opt, test_dataloader, model):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    acc_list = []

    model.eval()
    for _ in range(10):
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            emb = model(x)
            _, acc = loss_fn(emb, y, opt.num_support_val)
            acc_list.append(acc.item())

    print(f"Test Acc: {np.mean(acc_list):.4f}")


# ------------------ MAIN ------------------

def main():
    opt = get_parser().parse_args()
    os.makedirs(opt.experiment_root, exist_ok=True)

    init_seed(opt)

    train_c, val_c, test_c = get_valid_class_splits(
        opt.dataset_root,
        min_samples=5,
        seed=opt.manual_seed
    )

    class_splits = {"train": train_c, "val": val_c, "test": test_c}

    # Build one dataset early to infer input dim
    temp_dataset = IITDGaborTemplateDataset(
        root=opt.dataset_root,
        allowed_classes=train_c,
        min_samples_per_class=5
    )

    tr_loader = init_dataloader(opt, "train", class_splits)
    val_loader = init_dataloader(opt, "val", class_splits)
    test_loader = init_dataloader(opt, "test", class_splits)

    model = init_protonet(opt, temp_dataset)
    optim = init_optim(opt, model)
    scheduler = init_lr_scheduler(opt, optim)

    best_state = train(
        opt, tr_loader, model, optim, scheduler, val_loader
    )

    print("Testing best model...")
    model.load_state_dict(best_state)
    test(opt, test_loader, model)


if __name__ == "__main__":
    main()
