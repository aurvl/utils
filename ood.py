import numpy as np
import os

import torch
from torch.utils.data import DataLoader, TensorDataset


def build_ood_dataloaders(
    X_past,
    X_future,
    labels,
    category_map,
    cfg,
    ood_categories=("stochastic",),
    val_ratio=0.10,
):
    labels_arr = np.array(labels)
    categories = np.array([category_map[label] for label in labels_arr])

    ood_mask = np.isin(categories, list(ood_categories))
    id_mask = ~ood_mask

    X_id = X_past[id_mask]
    y_id = X_future[id_mask]
    labels_id = labels_arr[id_mask]
    categories_id = categories[id_mask]

    X_ood = X_past[ood_mask]
    y_ood = X_future[ood_mask]
    labels_ood = labels_arr[ood_mask]
    categories_ood = categories[ood_mask]

    rng = np.random.default_rng(cfg.seed)

    id_idx = rng.permutation(len(X_id))
    ood_idx = rng.permutation(len(X_ood))

    X_id = X_id[id_idx]
    y_id = y_id[id_idx]
    labels_id = labels_id[id_idx]
    categories_id = categories_id[id_idx]

    X_ood = X_ood[ood_idx]
    y_ood = y_ood[ood_idx]
    labels_ood = labels_ood[ood_idx]
    categories_ood = categories_ood[ood_idx]

    n_val = int(val_ratio * len(X_id))

    X_val = X_id[:n_val]
    y_val = y_id[:n_val]
    labels_val = labels_id[:n_val]
    categories_val = categories_id[:n_val]

    X_train = X_id[n_val:]
    y_train = y_id[n_val:]
    labels_train = labels_id[n_val:]
    categories_train = categories_id[n_val:]

    def to_loader(X, y, shuffle):
        X_t = torch.from_numpy(X).unsqueeze(-1).float()
        y_t = torch.from_numpy(y).unsqueeze(-1).float()

        ds = TensorDataset(X_t, y_t)

        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=cfg.device == "cuda",
            persistent_workers=True,
        )

    train_dl = to_loader(X_train, y_train, True)
    val_dl = to_loader(X_val, y_val, False)
    ood_dl = to_loader(X_ood, y_ood, False)

    print("OOD categories:", ood_categories)
    print("Train ID:", X_train.shape)
    print("Val ID:", X_val.shape)
    print("Test OOD:", X_ood.shape)

    return {
        "train_dl": train_dl,
        "val_dl": val_dl,
        "ood_dl": ood_dl,
        "labels_train": labels_train,
        "labels_val": labels_val,
        "labels_ood": labels_ood,
        "categories_train": categories_train,
        "categories_val": categories_val,
        "categories_ood": categories_ood,
    }