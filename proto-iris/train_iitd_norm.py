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
from src.protonet import ProtoNet
from src.parser_util import get_parser

from iitd_normalized_dataset import IITDNormalizedDataset




# ------------------ SEED ------------------

def init_seed(opt):
    """
    Disable cudnn to maximize reproducibility
    """
    torch.backends.cudnn.enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


# ------------------ DATASET ------------------

def init_dataset(opt):
    """
    IITD does not have predefined train/val/test splits.
    ProtoNet handles this via episodic sampling.
    """
    dataset = IITDNormalizedDataset(
        root=opt.dataset_root,
        min_samples_per_class=5
    )


    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr:
        raise ValueError(
            "Not enough classes for ProtoNet episodic training"
        )

    return dataset


def init_sampler(opt, labels, mode):
    """
    Same sampler logic as Omniglot
    """
    if mode == 'train':
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(
        labels=labels,
        classes_per_it=classes_per_it,
        num_samples=num_samples,
        iterations=opt.iterations
    )


from iitd_normalized_dataset import IITDNormalizedDataset


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

    dataset = IITDNormalizedDataset(
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



# ------------------ MODEL ------------------

def init_protonet(opt):
    """
    Initialize ProtoNet (CNN encoder)
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(opt, model):
    return torch.optim.Adam(
        params=model.parameters(),
        lr=opt.learning_rate
    )


def init_lr_scheduler(opt, optim):
    return torch.optim.lr_scheduler.StepLR(
        optimizer=optim,
        gamma=opt.lr_scheduler_gamma,
        step_size=opt.lr_scheduler_step
    )


# ------------------ UTIL ------------------

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def get_valid_class_splits(root, min_samples=5, seed=0):
    """
    Returns train / val / test class splits (identity-disjoint)
    """
    rng = np.random.RandomState(seed)

    class_counts = defaultdict(int)

    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            if f.endswith("_norm.npz"):
                class_counts[cls] += 1

    # Filter classes
    valid_classes = [
        cls for cls, cnt in class_counts.items()
        if cnt >= min_samples
    ]

    rng.shuffle(valid_classes)

    n = len(valid_classes)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)

    train_classes = valid_classes[:n_train]
    val_classes   = valid_classes[n_train:n_train + n_val]
    test_classes  = valid_classes[n_train + n_val:]

    print(
        f"Class split → Train: {len(train_classes)}, "
        f"Val: {len(val_classes)}, Test: {len(test_classes)}"
    )

    return train_classes, val_classes, test_classes


# ------------------ TRAIN ------------------

def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    best_acc = 0
    best_state = None

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print(f'=== Epoch: {epoch} ===')

        model.train()
        for batch in tqdm(tr_dataloader):
            optim.zero_grad()

            x, y = batch
            x, y = x.to(device), y.to(device)

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

        print(
            f'Avg Train Loss: {np.mean(train_loss[-opt.iterations:]):.4f}, '
            f'Avg Train Acc: {np.mean(train_acc[-opt.iterations:]):.4f}'
        )

        lr_scheduler.step()

        # ---------- Validation ----------
        if val_dataloader is not None:
            model.eval()
            for batch in val_dataloader:
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
                torch.save(best_state, best_model_path)

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(
            os.path.join(opt.experiment_root, name + '.txt'),
            locals()[name]
        )

    return best_state, best_acc


# ------------------ TEST ------------------

def test(opt, test_dataloader, model):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    acc_list = []
    model.eval()

    for _ in range(10):
        for batch in test_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            embeddings = model(x)
            _, acc = loss_fn(
                embeddings,
                target=y,
                n_support=opt.num_support_val
            )
            acc_list.append(acc.item())

    print(f'Test Acc: {np.mean(acc_list):.4f}')
    return np.mean(acc_list)


# ------------------ MAIN ------------------

def main():
    opt = get_parser().parse_args()
    os.makedirs(opt.experiment_root, exist_ok=True)

    init_seed(opt)

    train_classes, val_classes, test_classes = get_valid_class_splits(
        opt.dataset_root,
        min_samples=5,
        seed=opt.manual_seed
    )

    class_splits = {
        "train": train_classes,
        "val": val_classes,
        "test": test_classes
    }

    tr_dataloader = init_dataloader(opt, "train", class_splits)
    val_dataloader = init_dataloader(opt, "val", class_splits)
    test_dataloader = init_dataloader(opt, "test", class_splits)

    model = init_protonet(opt)
    optim = init_optim(opt, model)
    lr_scheduler = init_lr_scheduler(opt, optim)

    best_state, _ = train(
        opt,
        tr_dataloader,
        model,
        optim,
        lr_scheduler,
        val_dataloader
    )

    print("Testing with best model...")
    model.load_state_dict(best_state)
    test(opt, test_dataloader, model)


if __name__ == '__main__':
    main()
