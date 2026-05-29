from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .configs import FinetuneConfig
from .datasets import DownstreamFMriDataset


def compute_class_weights_for_dataset(dataset: DownstreamFMriDataset, cfg: FinetuneConfig):
    labels_df = dataset.labels
    if cfg.enable_disease and cfg.disease_col in labels_df.columns:
        col = labels_df[cfg.disease_col].dropna()
        if len(col) > 0:
            counts = col.value_counts()
            num_classes = cfg.num_disease_classes
            total = float(len(col))
            weights = []
            for c in range(num_classes):
                if c in counts:
                    weights.append(total / (num_classes * float(counts[c])))
                else:
                    weights.append(0.0)
            cfg.disease_class_weights = tuple(weights)
            print('[ClassWeight] disease:', cfg.disease_class_weights)
    if cfg.enable_sex and cfg.sex_col in labels_df.columns:
        col = labels_df[cfg.sex_col].dropna()
        if len(col) > 0:
            counts = col.value_counts()
            num_classes = cfg.num_sex_classes
            total = float(len(col))
            weights = []
            for c in range(num_classes):
                if c in counts:
                    weights.append(total / (num_classes * float(counts[c])))
                else:
                    weights.append(0.0)
            cfg.sex_class_weights = tuple(weights)
            print('[ClassWeight] sex:', cfg.sex_class_weights)

def build_stratified_train_val_split(dataset: DownstreamFMriDataset, cfg: FinetuneConfig) -> Tuple[np.ndarray, np.ndarray]:
    if not (cfg.enable_disease and cfg.disease_col in dataset.labels.columns):
        raise ValueError('Cannot do stratified split: disease label not available.')
    labels_series = dataset.labels[cfg.disease_col]
    labels = labels_series.to_numpy()
    rng = np.random.RandomState(cfg.seed)
    valid_mask = ~pd.isna(labels)
    all_indices = np.where(valid_mask)[0]
    labels_valid = labels[valid_mask].astype(int)
    unique_classes = np.unique(labels_valid)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for c in unique_classes:
        idx_c_all = all_indices[labels_valid == c]
        rng.shuffle(idx_c_all)
        n_c = len(idx_c_all)
        n_train_c = int(round(n_c * cfg.train_ratio))
        if n_c > 1:
            n_train_c = min(max(n_train_c, 1), n_c - 1)
        else:
            n_train_c = 1
        train_indices.append(idx_c_all[:n_train_c])
        val_indices.append(idx_c_all[n_train_c:])
    train_indices = np.concatenate(train_indices)
    val_indices = np.concatenate(val_indices)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return (train_indices, val_indices)

def build_stratified_kfold_indices(dataset: DownstreamFMriDataset, cfg: FinetuneConfig) -> List[np.ndarray]:
    if not (cfg.enable_disease and cfg.disease_col in dataset.labels.columns):
        raise ValueError('Cannot do stratified K-fold: disease label not available.')
    labels_series = dataset.labels[cfg.disease_col]
    labels = labels_series.to_numpy()
    rng = np.random.RandomState(cfg.seed)
    valid_mask = ~pd.isna(labels)
    all_indices = np.where(valid_mask)[0]
    labels_valid = labels[valid_mask].astype(int)
    unique_classes = np.unique(labels_valid)
    K = cfg.num_folds
    folds: List[List[int]] = [[] for _ in range(K)]
    for c in unique_classes:
        idx_c_all = all_indices[labels_valid == c]
        rng.shuffle(idx_c_all)
        fold_sizes = np.full(K, len(idx_c_all) // K, dtype=int)
        fold_sizes[:len(idx_c_all) % K] += 1
        start = 0
        for k in range(K):
            stop = start + fold_sizes[k]
            folds[k].extend(idx_c_all[start:stop])
            start = stop
    folds_np: List[np.ndarray] = []
    for k in range(K):
        fk = np.array(folds[k], dtype=int)
        rng.shuffle(fk)
        folds_np.append(fk)
    return folds_np

def build_random_kfold_indices(dataset: DownstreamFMriDataset, cfg: FinetuneConfig) -> List[np.ndarray]:
    num_graphs = len(dataset)
    K = cfg.num_folds
    indices = np.arange(num_graphs)
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(indices)
    fold_sizes = np.full(K, num_graphs // K, dtype=int)
    fold_sizes[:num_graphs % K] += 1
    folds: List[np.ndarray] = []
    current = 0
    for k in range(K):
        start, stop = (current, current + fold_sizes[k])
        folds.append(indices[start:stop])
        current = stop
    return folds
