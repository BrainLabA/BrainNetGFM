from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .configs import PretrainConfig


def train_epoch(model: nn.Module, loader, optimizer, device: torch.device, cfg: PretrainConfig, scaler: Optional[GradScaler] = None) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_cl = 0.0
    total_graphs = 0
    use_amp = scaler is not None and device.type == 'cuda'
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                recon_loss, cl_loss, _, _ = model(batch, mask_ratio_node=cfg.mask_ratio_node, mask_ratio_edge=cfg.mask_ratio_edge, edge_loss_weight=cfg.edge_loss_weight, drop_node_ratio=cfg.drop_node_ratio, drop_edge_ratio=cfg.drop_edge_ratio, feat_mask_ratio=cfg.feat_mask_ratio, temperature=cfg.temperature, sce_gamma=cfg.sce_gamma)
                loss = cfg.alpha_recon * recon_loss + cfg.beta_cl * cl_loss
            scaler.scale(loss).backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            recon_loss, cl_loss, _, _ = model(batch, mask_ratio_node=cfg.mask_ratio_node, mask_ratio_edge=cfg.mask_ratio_edge, edge_loss_weight=cfg.edge_loss_weight, drop_node_ratio=cfg.drop_node_ratio, drop_edge_ratio=cfg.drop_edge_ratio, feat_mask_ratio=cfg.feat_mask_ratio, temperature=cfg.temperature, sce_gamma=cfg.sce_gamma)
            loss = cfg.alpha_recon * recon_loss + cfg.beta_cl * cl_loss
            loss.backward()
            if cfg.max_grad_norm is not None and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
        num_graphs = batch.num_graphs
        total_graphs += num_graphs
        total_loss += loss.item() * num_graphs
        total_recon += recon_loss.item() * num_graphs
        total_cl += cl_loss.item() * num_graphs
    return total_loss / total_graphs, total_recon / total_graphs, total_cl / total_graphs


@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device: torch.device, cfg: PretrainConfig) -> Tuple[float, float, float]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_recon = torch.tensor(0.0, device=device)
    total_cl = torch.tensor(0.0, device=device)
    total_graphs = torch.tensor(0.0, device=device)
    for batch in loader:
        batch = batch.to(device)
        recon_loss, cl_loss, _, _ = model(batch, mask_ratio_node=cfg.mask_ratio_node, mask_ratio_edge=cfg.mask_ratio_edge, edge_loss_weight=cfg.edge_loss_weight, drop_node_ratio=cfg.drop_node_ratio, drop_edge_ratio=cfg.drop_edge_ratio, feat_mask_ratio=cfg.feat_mask_ratio, temperature=cfg.temperature, sce_gamma=cfg.sce_gamma)
        loss = cfg.alpha_recon * recon_loss + cfg.beta_cl * cl_loss
        num_graphs = batch.num_graphs
        total_loss += loss * num_graphs
        total_recon += recon_loss * num_graphs
        total_cl += cl_loss * num_graphs
        total_graphs += num_graphs
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_recon, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_cl, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_graphs, op=dist.ReduceOp.SUM)
    return (total_loss / total_graphs).item(), (total_recon / total_graphs).item(), (total_cl / total_graphs).item()
