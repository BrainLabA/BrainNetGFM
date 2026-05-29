from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .configs import FinetuneConfig
from .layers import GraphBackbone, MultiTaskGNN
from .losses import binary_auc_score, compute_total_loss


def train_epoch(model: nn.Module, loader, optimizer, device, cfg: FinetuneConfig) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    task_loss_sum: Dict[str, float] = {}
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss, loss_info = compute_total_loss(outputs, batch, cfg)
        loss.backward()
        optimizer.step()
        num_graphs = batch.num_graphs
        total_graphs += num_graphs
        total_loss += loss.item() * num_graphs
        for k, v in loss_info.items():
            task_loss_sum[k] = task_loss_sum.get(k, 0.0) + v * num_graphs
    avg_loss = total_loss / max(total_graphs, 1)
    avg_task_loss = {k: v / max(total_graphs, 1) for k, v in task_loss_sum.items()}
    return (avg_loss, avg_task_loss)

@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device, cfg: FinetuneConfig) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    task_loss_sum: Dict[str, float] = {}
    metrics_sum = {'disease_correct': 0, 'disease_total': 0, 'disease_tp': 0, 'disease_fp': 0, 'disease_tn': 0, 'disease_fn': 0, 'sex_correct': 0, 'sex_total': 0, 'sex_tp': 0, 'sex_fp': 0, 'sex_tn': 0, 'sex_fn': 0, 'symptom_abs_error_sum': 0.0, 'symptom_count': 0, 'age_abs_error_sum': 0.0, 'age_count': 0}
    disease_probs = []
    disease_labels = []
    sex_probs = []
    sex_labels = []
    symptom_preds = []
    symptom_targets = []
    age_preds = []
    age_targets = []
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        loss, loss_info = compute_total_loss(outputs, batch, cfg)
        num_graphs = batch.num_graphs
        total_graphs += num_graphs
        total_loss += loss.item() * num_graphs
        for k, v in loss_info.items():
            task_loss_sum[k] = task_loss_sum.get(k, 0.0) + v * num_graphs
        if cfg.enable_disease and hasattr(batch, 'y_disease') and (outputs['disease'] is not None):
            y = batch.y_disease
            mask = y >= 0
            if mask.any():
                logits = outputs['disease'][mask]
                labels = y[mask].long()
                preds = logits.argmax(dim=-1)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                correct = (preds == labels).sum().item()
                total = labels.size(0)
                metrics_sum['disease_correct'] += correct
                metrics_sum['disease_total'] += total
                tp = ((preds == 1) & (labels == 1)).sum().item()
                tn = ((preds == 0) & (labels == 0)).sum().item()
                fp = ((preds == 1) & (labels == 0)).sum().item()
                fn = ((preds == 0) & (labels == 1)).sum().item()
                metrics_sum['disease_tp'] += tp
                metrics_sum['disease_tn'] += tn
                metrics_sum['disease_fp'] += fp
                metrics_sum['disease_fn'] += fn
                disease_probs.append(probs.cpu())
                disease_labels.append(labels.cpu())
        if cfg.enable_sex and hasattr(batch, 'y_sex') and (outputs['sex'] is not None):
            y = batch.y_sex
            mask = y >= 0
            if mask.any():
                logits = outputs['sex'][mask]
                labels = y[mask].long()
                preds = logits.argmax(dim=-1)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                correct = (preds == labels).sum().item()
                total = labels.size(0)
                metrics_sum['sex_correct'] += correct
                metrics_sum['sex_total'] += total
                tp = ((preds == 1) & (labels == 1)).sum().item()
                tn = ((preds == 0) & (labels == 0)).sum().item()
                fp = ((preds == 1) & (labels == 0)).sum().item()
                fn = ((preds == 0) & (labels == 1)).sum().item()
                metrics_sum['sex_tp'] += tp
                metrics_sum['sex_tn'] += tn
                metrics_sum['sex_fp'] += fp
                metrics_sum['sex_fn'] += fn
                sex_probs.append(probs.cpu())
                sex_labels.append(labels.cpu())
        if cfg.enable_symptom and hasattr(batch, 'y_symptom') and (outputs['symptom'] is not None):
            y = batch.y_symptom
            mask = ~torch.isnan(y)
            if mask.any():
                pred = outputs['symptom'][mask]
                target = y[mask]
                abs_err = torch.abs(pred - target).sum().item()
                count = target.numel()
                metrics_sum['symptom_abs_error_sum'] += abs_err
                metrics_sum['symptom_count'] += count
                symptom_preds.append(pred.cpu())
                symptom_targets.append(target.cpu())
        if cfg.enable_age and hasattr(batch, 'y_age') and (outputs['age'] is not None):
            y = batch.y_age
            mask = ~torch.isnan(y)
            if mask.any():
                pred = outputs['age'][mask]
                target = y[mask]
                abs_err = torch.abs(pred - target).sum().item()
                count = target.numel()
                metrics_sum['age_abs_error_sum'] += abs_err
                metrics_sum['age_count'] += count
                age_preds.append(pred.cpu())
                age_targets.append(target.cpu())
    avg_loss = total_loss / max(total_graphs, 1)
    avg_task_loss = {k: v / max(total_graphs, 1) for k, v in task_loss_sum.items()}
    metrics: Dict[str, float] = {}
    if metrics_sum['disease_total'] > 0:
        acc = metrics_sum['disease_correct'] / metrics_sum['disease_total']
        metrics['disease_acc'] = acc
        tp = metrics_sum['disease_tp']
        tn = metrics_sum['disease_tn']
        fp = metrics_sum['disease_fp']
        fn = metrics_sum['disease_fn']
        sens = tp / (tp + fn) if tp + fn > 0 else float('nan')
        spec = tn / (tn + fp) if tn + fp > 0 else float('nan')
        metrics['disease_sens'] = sens
        metrics['disease_spec'] = spec
        if len(disease_labels) > 0:
            y_true = torch.cat(disease_labels).numpy()
            y_score = torch.cat(disease_probs).numpy()
            if np.unique(y_true).size == 2:
                metrics['disease_auc'] = binary_auc_score(y_true, y_score)
            else:
                metrics['disease_auc'] = float('nan')
        else:
            metrics['disease_auc'] = float('nan')
        if not math.isnan(sens) and (not math.isnan(spec)):
            balanced_acc = 0.5 * (sens + spec)
        else:
            balanced_acc = float('nan')
        metrics['disease_bal_acc'] = balanced_acc
        auc_val = metrics.get('disease_auc', float('nan'))
        if isinstance(auc_val, float) and (not math.isnan(auc_val)):
            auc_norm = max(0.0, (auc_val - 0.5) * 2.0)
        else:
            auc_norm = 0.0
        if not math.isnan(acc) and (not math.isnan(sens)) and (not math.isnan(spec)):
            metrics['disease_combo2'] = (acc + auc_norm) / 2.0
        else:
            metrics['disease_combo2'] = float('nan')
        if not math.isnan(acc) and (not math.isnan(sens)) and (not math.isnan(spec)):
            metrics['disease_combo4'] = (acc + sens + spec + auc_norm) / 4.0
        else:
            metrics['disease_combo4'] = float('nan')
    if metrics_sum['sex_total'] > 0:
        acc = metrics_sum['sex_correct'] / metrics_sum['sex_total']
        metrics['sex_acc'] = acc
        tp = metrics_sum['sex_tp']
        tn = metrics_sum['sex_tn']
        fp = metrics_sum['sex_fp']
        fn = metrics_sum['sex_fn']
        sens = tp / (tp + fn) if tp + fn > 0 else float('nan')
        spec = tn / (tn + fp) if tn + fp > 0 else float('nan')
        metrics['sex_sens'] = sens
        metrics['sex_spec'] = spec
        if len(sex_labels) > 0:
            y_true = torch.cat(sex_labels).numpy()
            y_score = torch.cat(sex_probs).numpy()
            if np.unique(y_true).size == 2:
                metrics['sex_auc'] = binary_auc_score(y_true, y_score)
            else:
                metrics['sex_auc'] = float('nan')
    if metrics_sum['symptom_count'] > 0 and len(symptom_targets) > 0:
        preds = torch.cat(symptom_preds).numpy()
        targets = torch.cat(symptom_targets).numpy()
        diff = preds - targets
        mae = metrics_sum['symptom_abs_error_sum'] / metrics_sum['symptom_count']
        mse = float(np.mean(diff ** 2))
        metrics['symptom_mae'] = mae
        metrics['symptom_mse'] = mse
        if np.std(preds) > 1e-08 and np.std(targets) > 1e-08:
            corr = float(np.corrcoef(preds, targets)[0, 1])
        else:
            corr = float('nan')
        metrics['symptom_corr'] = abs(corr)
        denom = np.sum((targets - targets.mean()) ** 2)
        if denom > 0:
            r2 = 1.0 - float(np.sum(diff ** 2) / denom)
        else:
            r2 = float('nan')
        metrics['symptom_r2'] = r2
    if metrics_sum['age_count'] > 0 and len(age_targets) > 0:
        preds = torch.cat(age_preds).numpy()
        targets = torch.cat(age_targets).numpy()
        diff = preds - targets
        mae = metrics_sum['age_abs_error_sum'] / metrics_sum['age_count']
        mse = float(np.mean(diff ** 2))
        metrics['age_mae'] = mae
        metrics['age_mse'] = mse
        if np.std(preds) > 1e-08 and np.std(targets) > 1e-08:
            corr = float(np.corrcoef(preds, targets)[0, 1])
        else:
            corr = float('nan')
        metrics['age_corr'] = abs(corr)
        denom = np.sum((targets - targets.mean()) ** 2)
        if denom > 0:
            r2 = 1.0 - float(np.sum(diff ** 2) / denom)
        else:
            r2 = float('nan')
        metrics['age_r2'] = r2
    return (avg_loss, avg_task_loss, metrics)

def get_param_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    encoder_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            encoder_params.append(param)
        else:
            head_params.append(param)
    return (encoder_params, head_params)

def run_one_fold(fold_idx: int, cfg: FinetuneConfig, dataset: Dataset, train_indices: np.ndarray, val_indices: np.ndarray, device, distributed: bool, rank: int, local_rank: int, world_size: int, pretrain_ckpt: Optional[Dict[str, Any]]):
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size_per_gpu, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size_per_gpu, sampler=val_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size_per_gpu, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size_per_gpu, shuffle=False)
    first_data: Data = dataset[0]
    in_dim = first_data.x.size(1)
    edge_attr_dim = first_data.edge_attr.size(1) if first_data.edge_attr is not None else 0
    graph_attr_dim = first_data.graph_attr.size(1) if hasattr(first_data, 'graph_attr') else 0
    pre_cfg = pretrain_ckpt.get('cfg', {}) if pretrain_ckpt is not None else {}
    gnn_hidden_dim = int(pre_cfg.get('gnn_hidden_dim', cfg.gnn_hidden_dim))
    num_gnn_layers = int(pre_cfg.get('num_gnn_layers', cfg.num_gnn_layers))
    dropout = float(pre_cfg.get('dropout', cfg.dropout))
    heads = int(pre_cfg.get('transformer_heads', cfg.transformer_heads))
    pe_dim = int(pre_cfg.get('pe_dim', cfg.pe_dim))
    jk_mode = pre_cfg.get('jk_mode', cfg.jk_mode)
    use_graph_attr_flag = bool(pre_cfg.get('use_graph_attr', cfg.use_graph_attr))
    node_norm_type = pre_cfg.get('node_norm_type', cfg.node_norm_type)
    backbone = GraphBackbone(base_in_dim=in_dim, pe_dim=pe_dim, use_graph_attr=use_graph_attr_flag, graph_attr_dim=graph_attr_dim if use_graph_attr_flag else 0, gnn_hidden_dim=gnn_hidden_dim, num_gnn_layers=num_gnn_layers, dropout=dropout, heads=heads, edge_dim=edge_attr_dim, jk_mode=jk_mode, node_norm_type=node_norm_type).to(device)
    if pretrain_ckpt is not None:
        save_mode = pretrain_ckpt.get('save_mode', None)
        loaded = False
        if save_mode in ('node+encoder', 'backbone') and 'backbone_state' in pretrain_ckpt:
            backbone_state = pretrain_ckpt['backbone_state']
            missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
            loaded = True
            if rank == 0:
                print(f'[Fold {fold_idx}] Loaded backbone_state (node_feat_proj + encoder) from {cfg.pretrained_encoder_path}')
                if missing:
                    print('  Missing keys in backbone:', missing)
                if unexpected:
                    print('  Unexpected keys in backbone:', unexpected)
        elif save_mode == 'encoder' and 'encoder_state' in pretrain_ckpt:
            encoder_state = pretrain_ckpt['encoder_state']
            missing, unexpected = backbone.encoder.load_state_dict(encoder_state, strict=False)
            loaded = True
            if rank == 0:
                print(f'[Fold {fold_idx}] Loaded encoder_state into backbone.encoder from {cfg.pretrained_encoder_path}')
                if missing:
                    print('  Missing keys in encoder:', missing)
                if unexpected:
                    print('  Unexpected keys in encoder:', unexpected)
        elif 'model_state' in pretrain_ckpt:
            state = pretrain_ckpt['model_state']
            backbone_state = {k: v for k, v in state.items() if k.startswith('node_feat_proj.') or k.startswith('encoder.')}
            missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
            loaded = True
            if rank == 0:
                print(f'[Fold {fold_idx}] Loaded backbone from full model_state ({cfg.pretrained_encoder_path})')
                if missing:
                    print('  Missing keys in backbone:', missing)
                if unexpected:
                    print('  Unexpected keys in backbone:', unexpected)
        if not loaded and rank == 0:
            print(f'[Fold {fold_idx}] [Warning] Could not find suitable pretrained weights in {cfg.pretrained_encoder_path}, backbone will train from scratch.')
    elif rank == 0:
        print(f'[Fold {fold_idx}] [Warning] pretrain_ckpt is None, backbone will train from scratch.')
    graph_emb_dim = 2 * backbone.encoder_out_dim + (graph_attr_dim if use_graph_attr_flag else 0)
    model = MultiTaskGNN(backbone=backbone, cfg=cfg, graph_emb_dim=graph_emb_dim).to(device)
    if cfg.freeze_encoder:
        for p in model.backbone.parameters():
            p.requires_grad = False
        if rank == 0:
            print(f'[Fold {fold_idx}] Backbone is frozen. Only heads will be trained.')
    encoder_params, head_params = get_param_groups(model)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optim_param_groups = []
    if len(encoder_params) > 0:
        optim_param_groups.append({'params': encoder_params, 'lr': cfg.lr_encoder})
    if len(head_params) > 0:
        optim_param_groups.append({'params': head_params, 'lr': cfg.lr_head})
    optimizer = torch.optim.Adam(optim_param_groups, weight_decay=cfg.weight_decay)
    if cfg.use_scheduler:
        if cfg.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
    else:
        scheduler = None
    best_monitor: Optional[float] = None
    best_metrics: Dict[str, float] = {}
    best_monitor_name = cfg.monitor_metric if cfg.monitor_metric else 'val_loss'
    for epoch in range(1, cfg.num_epochs + 1):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        train_loss, train_task_loss = train_epoch(model, train_loader, optimizer, device, cfg)
        val_loss, val_task_loss, val_metrics = eval_epoch(model, val_loader, device, cfg)
        if scheduler is not None:
            scheduler.step()
        if rank == 0:
            print(f'[Fold {fold_idx}] [Epoch {epoch}/{cfg.num_epochs}] TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}')
            if val_metrics:
                print('  Val metrics:', {k: round(v, 4) for k, v in val_metrics.items()})
        if cfg.monitor_metric and cfg.monitor_metric in val_metrics:
            monitor_val = val_metrics[cfg.monitor_metric]
            if monitor_val is None or (isinstance(monitor_val, float) and math.isnan(monitor_val)):
                monitor_val = -val_loss if cfg.monitor_mode == 'max' else val_loss
                monitor_name_used = 'val_loss(fallback)'
            else:
                monitor_name_used = cfg.monitor_metric
        else:
            monitor_val = -val_loss if cfg.monitor_mode == 'max' else val_loss
            monitor_name_used = 'val_loss(fallback)'
        update_best = False
        if best_monitor is None:
            update_best = True
        elif cfg.monitor_mode == 'max':
            if monitor_val >= best_monitor:
                update_best = True
        elif monitor_val <= best_monitor:
            update_best = True
        if update_best:
            best_monitor = monitor_val
            best_metrics = dict(val_metrics)
            if rank == 0:
                dir_enc, name_enc = os.path.split(cfg.save_encoder_path)
                dir_full, name_full = os.path.split(cfg.save_full_model_path)
                save_enc_path = os.path.join(dir_enc, f'{os.path.splitext(name_enc)[0]}_fold{fold_idx}.pt')
                save_full_path = os.path.join(dir_full, f'{os.path.splitext(name_full)[0]}_fold{fold_idx}.pt')
                if distributed:
                    backbone_state = model.module.backbone.state_dict()
                    full_state = model.module.state_dict()
                else:
                    backbone_state = model.backbone.state_dict()
                    full_state = model.state_dict()
                if dir_enc:
                    os.makedirs(dir_enc, exist_ok=True)
                if dir_full:
                    os.makedirs(dir_full, exist_ok=True)
                torch.save(backbone_state, save_enc_path)
                torch.save(full_state, save_full_path)
                print(f'  >> [Fold {fold_idx}] New best model: monitor={monitor_name_used}, value={monitor_val:.4f}  (saved)')
    if rank == 0:
        if best_monitor is not None:
            print(f'[Fold {fold_idx}] Best monitor({best_monitor_name}) = {best_monitor:.4f}')
        else:
            print(f'[Fold {fold_idx}] No best monitor recorded.')
    return best_metrics
