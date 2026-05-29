from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from .configs import FinetuneConfig
from .datasets import DownstreamFMriDataset
from .finetune_engine import run_one_fold
from .splits import compute_class_weights_for_dataset, build_stratified_kfold_indices, build_stratified_train_val_split


def main(cfg: Optional[FinetuneConfig] = None):
    if cfg is None:
        cfg = FinetuneConfig()
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        distributed = True
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        distributed = False
        world_size = 1
        rank = 0
        local_rank = 0
    if distributed:
        dist.init_process_group(backend=cfg.ddp_backend, init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        print('Device:', device, '| Distributed:', distributed, '| World size:', world_size)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    pretrain_ckpt: Optional[Dict[str, Any]] = None
    if os.path.exists(cfg.pretrained_encoder_path):
        pretrain_ckpt = torch.load(cfg.pretrained_encoder_path, map_location='cpu')
        if rank == 0:
            print(f'[Main] Loaded pretrained GraphSSLModel ckpt from {cfg.pretrained_encoder_path}')
    elif rank == 0:
        print(f'[Main] [Warning] pretrained_encoder_path not found: {cfg.pretrained_encoder_path}')
    pre_cfg = pretrain_ckpt.get('cfg', {}) if pretrain_ckpt is not None else {}
    dataset = DownstreamFMriDataset(cfg, pretrain_cfg=pre_cfg)
    num_graphs = len(dataset)
    if rank == 0:
        print('Num subjects after filtering:', num_graphs)
        first_data = dataset[0]
        print('Num ROIs:', first_data.num_nodes)
        print('Node feature dim:', first_data.x.size(1))
        print('Edge attr dim:', first_data.edge_attr.size(1))
        if hasattr(first_data, 'graph_attr'):
            print('Graph attr dim:', first_data.graph_attr.size(1))
        if hasattr(first_data, 'pe'):
            print('PE dim:', first_data.pe.size(1))
    if num_graphs < 2:
        if rank == 0:
            print('Not enough samples for training.')
        if distributed:
            dist.destroy_process_group()
        return
    if rank == 0 and cfg.use_class_weights:
        compute_class_weights_for_dataset(dataset, cfg)
    if distributed:
        for attr_name in ['disease_class_weights', 'sex_class_weights']:
            val = getattr(cfg, attr_name)
            if rank == 0:
                if val is None:
                    tensor = torch.tensor([0.0, 0.0], dtype=torch.float32)
                else:
                    tensor = torch.tensor(list(val), dtype=torch.float32)
            else:
                tensor = torch.empty(2, dtype=torch.float32)
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            tensor = tensor.contiguous()
            dist.broadcast(tensor, src=0)
            tuple_val = (float(tensor[0].item()), float(tensor[1].item()))
            setattr(cfg, attr_name, tuple_val)
    if cfg.num_folds is None or cfg.num_folds <= 1:
        if rank == 0:
            print('Running single train/val split (no cross-validation).')
        if cfg.enable_disease and cfg.disease_col in dataset.labels.columns:
            train_indices, val_indices = build_stratified_train_val_split(dataset, cfg)
            if rank == 0:
                print(f"Stratified split on '{cfg.disease_col}': Train={len(train_indices)}, Val={len(val_indices)}")
        else:
            train_len = int(cfg.train_ratio * num_graphs)
            indices = np.arange(num_graphs)
            rng = np.random.RandomState(cfg.seed)
            rng.shuffle(indices)
            train_indices = indices[:train_len]
            val_indices = indices[train_len:]
            if rank == 0:
                print('Disease label not available; using non-stratified random split.')
        best_metrics = run_one_fold(fold_idx=1, cfg=cfg, dataset=dataset, train_indices=train_indices, val_indices=val_indices, device=device, distributed=distributed, rank=rank, local_rank=local_rank, world_size=world_size, pretrain_ckpt=pretrain_ckpt)
        if rank == 0 and best_metrics:
            print('[Single split] Final metrics:', {k: round(v, 4) for k, v in best_metrics.items()})
    else:
        K = cfg.num_folds
        if rank == 0:
            print(f'Running {K}-fold cross-validation.')
        if cfg.enable_disease and cfg.disease_col in dataset.labels.columns:
            folds = build_stratified_kfold_indices(dataset, cfg)
            if rank == 0:
                for i, f in enumerate(folds):
                    print(f"Fold {i + 1}: {len(f)} samples (stratified on '{cfg.disease_col}')")
        else:
            indices = np.arange(num_graphs)
            rng = np.random.RandomState(cfg.seed)
            rng.shuffle(indices)
            fold_sizes = np.full(K, num_graphs // K, dtype=int)
            fold_sizes[:num_graphs % K] += 1
            current = 0
            folds = []
            for fold_size in fold_sizes:
                start, stop = (current, current + fold_size)
                folds.append(indices[start:stop])
                current = stop
            if rank == 0:
                print('Disease label not available; using non-stratified K-fold.')
        cv_metric_values: Dict[str, List[float]] = {}
        for fold_idx in range(K):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(K) if i != fold_idx])
            if rank == 0:
                print('=' * 40)
                print(f'Start Fold {fold_idx + 1}/{K} | Train={len(train_idx)}, Val={len(val_idx)}')
            best_metrics = run_one_fold(fold_idx=fold_idx + 1, cfg=cfg, dataset=dataset, train_indices=train_idx, val_indices=val_idx, device=device, distributed=distributed, rank=rank, local_rank=local_rank, world_size=world_size, pretrain_ckpt=pretrain_ckpt)
            if rank == 0 and best_metrics:
                for k, v in best_metrics.items():
                    cv_metric_values.setdefault(k, []).append(v)
        if rank == 0 and cv_metric_values:
            print('=' * 40)
            print(f'{K}-fold cross-validation results (mean ± std):')
            for k, values in cv_metric_values.items():
                arr = np.array(values, dtype=float)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                print(f'  {k}: {mean:.4f} ± {std:.4f}  (values: {[round(x, 4) for x in values]})')
    if distributed:
        dist.destroy_process_group()
