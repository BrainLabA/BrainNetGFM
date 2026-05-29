from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.data import Batch

from .configs import FinetuneConfig


def sce_loss(x_recon: torch.Tensor, x_target: torch.Tensor, gamma: float=2.0):
    cos_sim = F.cosine_similarity(x_recon, x_target, dim=-1)
    loss = torch.pow(1.0 - cos_sim, gamma)
    return loss.mean()

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    if not (dist.is_available() and dist.is_initialized()):
        batch_size = z1.size(0)
        logits = torch.matmul(z1, z2.t()) / temperature
        labels = torch.arange(batch_size, device=z1.device)
        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_12 + loss_21)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batch_size = z1.size(0)
    z1_list = [torch.zeros_like(z1) for _ in range(world_size)]
    z2_list = [torch.zeros_like(z2) for _ in range(world_size)]
    dist.all_gather(z1_list, z1.contiguous())
    dist.all_gather(z2_list, z2.contiguous())
    z1_all = torch.cat(z1_list, dim=0)
    z2_all = torch.cat(z2_list, dim=0)
    logits_12 = torch.matmul(z1, z2_all.t()) / temperature
    logits_21 = torch.matmul(z2, z1_all.t()) / temperature
    labels = rank * batch_size + torch.arange(batch_size, device=z1.device)
    loss_12 = F.cross_entropy(logits_12, labels)
    loss_21 = F.cross_entropy(logits_21, labels)
    return 0.5 * (loss_12 + loss_21)

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    if not (dist.is_available() and dist.is_initialized()):
        batch_size = z1.size(0)
        logits = torch.matmul(z1, z2.t()) / temperature
        labels = torch.arange(batch_size, device=z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batch_size = z1.size(0)
    z2_list = [torch.zeros_like(z2) for _ in range(world_size)]
    dist.all_gather(z2_list, z2.contiguous())
    z2_all = torch.cat(z2_list, dim=0)
    logits = torch.matmul(z1, z2_all.t()) / temperature
    labels = rank * batch_size + torch.arange(batch_size, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def node_supervised_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, roi_ids: torch.Tensor, graph_ids: torch.Tensor, temperature: float=0.2, cross_subject_pos_weight: float=0.0):
    device = z1.device
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        graph_ids = graph_ids + rank * 1000000
        z1_list = [torch.zeros_like(z1) for _ in range(world_size)]
        z2_list = [torch.zeros_like(z2) for _ in range(world_size)]
        roi_list = [torch.zeros_like(roi_ids) for _ in range(world_size)]
        gid_list = [torch.zeros_like(graph_ids) for _ in range(world_size)]
        dist.all_gather(z1_list, z1.contiguous())
        dist.all_gather(z2_list, z2.contiguous())
        dist.all_gather(roi_list, roi_ids.contiguous())
        dist.all_gather(gid_list, graph_ids.contiguous())
        z1_all = torch.cat(z1_list, dim=0)
        z2_all = torch.cat(z2_list, dim=0)
        roi_all = torch.cat(roi_list, dim=0)
        gid_all = torch.cat(gid_list, dim=0)
    else:
        z1_all = z1
        z2_all = z2
        roi_all = roi_ids
        gid_all = graph_ids
    N_local = z1.size(0)
    N_all = z1_all.size(0)
    roi_local = roi_ids.view(-1, 1)
    roi_all_b = roi_all.view(1, -1)
    gid_local = graph_ids.view(-1, 1)
    gid_all_b = gid_all.view(1, -1)
    sim_12 = torch.matmul(z1, z2_all.t()) / temperature
    sim_21 = torch.matmul(z2, z1_all.t()) / temperature
    pos_strict = (roi_local == roi_all_b) & (gid_local == gid_all_b)
    pos_cross = (roi_local == roi_all_b) & (gid_local != gid_all_b)

    def _supcon_from_logits(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        denom = exp_logits.sum(dim=1, keepdim=True) + 1e-12
        log_prob = logits - torch.log(denom)
        pos_count = pos_mask.sum(dim=1)
        pos_count_clamped = pos_count.clamp(min=1)
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_count_clamped
        valid = pos_count > 0
        if valid.any():
            loss = -mean_log_prob_pos[valid].mean()
        else:
            loss = torch.tensor(0.0, device=logits.device)
        return loss
    loss_12_strict = _supcon_from_logits(sim_12, pos_strict)
    loss_12_cross = _supcon_from_logits(sim_12, pos_cross) if cross_subject_pos_weight > 0 else 0.0
    loss_21_strict = _supcon_from_logits(sim_21, pos_strict)
    loss_21_cross = _supcon_from_logits(sim_21, pos_cross) if cross_subject_pos_weight > 0 else 0.0
    loss_strict = 0.5 * (loss_12_strict + loss_21_strict)
    loss_cross = 0.5 * (loss_12_cross + loss_21_cross) if cross_subject_pos_weight > 0 else 0.0
    return loss_strict + cross_subject_pos_weight * loss_cross


def compute_total_loss(outputs: Dict[str, Any], batch: Batch, cfg: FinetuneConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    total_loss = None
    loss_info: Dict[str, float] = {}
    if cfg.enable_disease and hasattr(batch, 'y_disease') and (outputs['disease'] is not None):
        y = batch.y_disease
        mask = y >= 0
        if mask.any():
            logits = outputs['disease'][mask]
            labels = y[mask].long()
            weight = None
            if cfg.use_class_weights and cfg.disease_class_weights is not None:
                weight = torch.tensor(cfg.disease_class_weights, device=logits.device, dtype=torch.float32)
            loss_d = F.cross_entropy(logits, labels, weight=weight)
            total_loss = cfg.lambda_disease * loss_d if total_loss is None else total_loss + cfg.lambda_disease * loss_d
            loss_info['disease'] = loss_d.detach().item()
    if cfg.enable_sex and hasattr(batch, 'y_sex') and (outputs['sex'] is not None):
        y = batch.y_sex
        mask = y >= 0
        if mask.any():
            logits = outputs['sex'][mask]
            labels = y[mask].long()
            weight = None
            if cfg.use_class_weights and cfg.sex_class_weights is not None:
                weight = torch.tensor(cfg.sex_class_weights, device=logits.device, dtype=torch.float32)
            loss_s = F.cross_entropy(logits, labels, weight=weight)
            total_loss = cfg.lambda_sex * loss_s if total_loss is None else total_loss + cfg.lambda_sex * loss_s
            loss_info['sex'] = loss_s.detach().item()
    if cfg.enable_symptom and hasattr(batch, 'y_symptom') and (outputs['symptom'] is not None):
        y = batch.y_symptom
        mask = ~torch.isnan(y)
        if mask.any():
            pred = outputs['symptom'][mask]
            target = y[mask]
            loss_sym = F.mse_loss(pred, target)
            total_loss = cfg.lambda_symptom * loss_sym if total_loss is None else total_loss + cfg.lambda_symptom * loss_sym
            loss_info['symptom'] = loss_sym.detach().item()
    if cfg.enable_age and hasattr(batch, 'y_age') and (outputs['age'] is not None):
        y = batch.y_age
        mask = ~torch.isnan(y)
        if mask.any():
            pred = outputs['age'][mask]
            target = y[mask]
            loss_age = F.mse_loss(pred, target)
            total_loss = cfg.lambda_age * loss_age if total_loss is None else total_loss + cfg.lambda_age * loss_age
            loss_info['age'] = loss_age.detach().item()
    if total_loss is None:
        total_loss = torch.tensor(0.0, device=batch.x.device)
    return (total_loss, loss_info)

def binary_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    assert y_true.shape == y_score.shape
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)
    if P == 0 or N == 0:
        return float('nan')
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    tpr = tps / P
    fpr = fps / N
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    auc = float(np.trapz(tpr, fpr))
    return auc

