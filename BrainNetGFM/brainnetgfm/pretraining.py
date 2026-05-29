from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from .configs import PretrainConfig
from .datasets import FMriFcDataset
from .pretrain_engine import eval_epoch, train_epoch
from .ssl_model import GraphSSLModel


def _distributed_context(cfg: PretrainConfig):
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
    return distributed, world_size, rank, local_rank, device


def _build_loaders(dataset: FMriFcDataset, cfg: PretrainConfig, distributed: bool, world_size: int, rank: int):
    train_len = int(cfg.train_ratio * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)
    loader_kwargs = dict(batch_size=cfg.batch_size_per_gpu, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, persistent_workers=True if cfg.num_workers > 0 else False)
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _save_checkpoint(model: nn.Module, cfg: PretrainConfig, distributed: bool):
    ssl_model = model.module if distributed else model
    ckpt = {'cfg': asdict(cfg), 'save_mode': cfg.save_model_mode}
    if cfg.save_model_mode == 'encoder':
        ckpt['encoder_state'] = ssl_model.encoder.state_dict()
        detail = 'encoder'
    elif cfg.save_model_mode in ('node+encoder', 'backbone'):
        ckpt['backbone_state'] = {k: v for k, v in ssl_model.state_dict().items() if k.startswith('node_feat_proj.') or k.startswith('encoder.')}
        detail = 'node_feat_proj + encoder'
    elif cfg.save_model_mode == 'full':
        ckpt['model_state'] = ssl_model.state_dict()
        detail = 'full GraphSSLModel'
    else:
        raise ValueError(f'Unknown save_model_mode: {cfg.save_model_mode}')
    save_dir = os.path.dirname(cfg.save_encoder_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(ckpt, cfg.save_encoder_path)
    return detail


def main(cfg: Optional[PretrainConfig] = None):
    if cfg is None:
        cfg = PretrainConfig()
    distributed, world_size, rank, local_rank, device = _distributed_context(cfg)
    if rank == 0:
        print('Using device:', device, '| Distributed:', distributed, '| World size:', world_size)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dataset = FMriFcDataset(npy_path=cfg.fc_npy_path, topk_ratio=cfg.topk_ratio, fisher_z=cfg.fisher_z, use_abs_for_topk=cfg.use_abs_for_topk, use_graph_attr=cfg.use_graph_attr, roi_mni_path=cfg.roi_mni_path, pe_type=cfg.pe_type, pe_dim=cfg.pe_dim, recon_target_mode=cfg.recon_target_mode, recon_topk_ratio=cfg.recon_topk_ratio)
    if rank == 0:
        print('Number of subjects (graphs):', len(dataset))
    first_data = dataset[0]
    node_in_dim = first_data.x.size(1)
    fc_dim = first_data.x_target.size(1) if hasattr(first_data, 'x_target') else node_in_dim
    pe_dim = first_data.pe.size(1) if hasattr(first_data, 'pe') else 0
    edge_attr_dim = first_data.edge_attr.size(1) if first_data.edge_attr is not None else 0
    graph_attr_dim = first_data.graph_attr.size(1) if hasattr(first_data, 'graph_attr') else 0
    if rank == 0:
        print('Number of ROIs:', first_data.num_nodes)
        print('Node input dim (x):', node_in_dim)
        print('FC dim (x_target row dim):', fc_dim)
        print('PE dim:', pe_dim)
        print('Edge attr dim:', edge_attr_dim)
        print('Graph attr dim:', graph_attr_dim)
    train_loader, val_loader = _build_loaders(dataset, cfg, distributed, world_size, rank)
    model = GraphSSLModel(node_in_dim=node_in_dim, fc_dim=fc_dim, pe_dim=pe_dim, use_graph_attr=cfg.use_graph_attr, graph_attr_dim=graph_attr_dim, gnn_hidden_dim=cfg.gnn_hidden_dim, proj_dim=cfg.proj_dim, recon_hidden_dim=cfg.recon_hidden_dim, num_gnn_layers=cfg.num_gnn_layers, num_decoder_layers=cfg.num_decoder_layers, dropout=cfg.dropout, heads=cfg.transformer_heads, edge_dim=edge_attr_dim, jk_mode=cfg.jk_mode, node_norm_type=cfg.node_norm_type, decoder_hidden_dim=cfg.recon_hidden_dim, recon_target_mode=cfg.recon_target_mode, recon_topk_ratio=cfg.recon_topk_ratio, recon_proj_dim=cfg.recon_proj_dim, node_mask_edge_mode=cfg.node_mask_edge_mode, node_cl_cross_subject_weight=cfg.node_cl_cross_subject_weight, recon_loss_type=cfg.recon_loss_type, contrastive_loss_type=cfg.contrastive_loss_type).to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    best_val_loss = float('inf')
    for epoch in range(1, cfg.num_epochs + 1):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        train_loss, train_recon, train_cl = train_epoch(model, train_loader, optimizer, device, cfg, scaler=scaler if use_amp else None)
        val_loss, val_recon, val_cl = eval_epoch(model, val_loader, device, cfg)
        if rank == 0:
            print(f'Epoch [{epoch}/{cfg.num_epochs}] TrainLoss={train_loss:.4f} (Recon={train_recon:.4f}, CL={train_cl:.4f})  ValLoss={val_loss:.4f} (Recon={val_recon:.4f}, CL={val_cl:.4f})')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                detail = _save_checkpoint(model, cfg, distributed)
                print(f'  >> Saved best {detail} checkpoint to {cfg.save_encoder_path}')
        scheduler.step()
    if rank == 0:
        print('Pretraining done. Best val loss:', best_val_loss)
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
