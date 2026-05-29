from __future__ import annotations

from typing import Optional

import numpy as np
import torch

def build_graph_from_fc(fc: np.ndarray, topk_ratio: float=0.2, fisher_z: float=True, use_abs_for_topk: bool=True, roi_coords: Optional[np.ndarray]=None, roi_dist: Optional[np.ndarray]=None, return_graph_stats: bool=False):
    if not (fc.ndim == 2 and fc.shape[0] == fc.shape[1]):
        raise ValueError(f'build_graph_from_fc expects [R, R], got {fc.shape}')
    fc = fc.astype(np.float32)
    fc = 0.5 * (fc + fc.T)
    np.fill_diagonal(fc, 0.0)
    if fisher_z:
        fc = np.clip(fc, -0.999999, 0.999999)
        fc = np.arctanh(fc)
    graph_attr = None
    if return_graph_stats:
        mean_val = float(fc.mean())
        std_val = float(fc.std())
        graph_attr = torch.tensor([[mean_val, std_val]], dtype=torch.float32)
    weight_mat = np.abs(fc) if use_abs_for_topk else fc
    R = fc.shape[0]
    triu_idx = np.triu_indices(R, k=1)
    vals = weight_mat[triu_idx]
    num_possible = vals.shape[0]
    k = max(int(num_possible * topk_ratio), 1)
    if k < num_possible:
        thresh = np.partition(vals, -k)[-k]
        mask = vals >= thresh
    else:
        mask = np.ones_like(vals, dtype=bool)
    src = triu_idx[0][mask]
    dst = triu_idx[1][mask]
    fc_vals = fc[triu_idx][mask]
    sign_vals = np.sign(fc_vals)
    if roi_dist is not None:
        dist_norm = roi_dist[src, dst].astype(np.float32)
    elif roi_coords is not None:
        coord_i = roi_coords[src]
        coord_j = roi_coords[dst]
        diff = coord_i - coord_j
        dist = np.sqrt((diff ** 2).sum(axis=1))
        mu = dist.mean()
        sigma = dist.std()
        if sigma < 1e-06:
            sigma = 1.0
        dist_norm = ((dist - mu) / sigma).astype(np.float32)
    else:
        dist_norm = np.zeros_like(fc_vals, dtype=np.float32)
    edge_feat = np.stack([fc_vals, sign_vals, dist_norm], axis=1).astype(np.float32)
    edge_feat = np.concatenate([edge_feat, edge_feat], axis=0)
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    edge_index = torch.from_numpy(edge_index).long()
    edge_attr = torch.from_numpy(edge_feat)
    x_fc = torch.from_numpy(fc.copy())
    x_mean = x_fc.mean(dim=1, keepdim=True)
    x_std = x_fc.std(dim=1, keepdim=True).clamp(min=1e-06)
    x_z = (x_fc - x_mean) / x_std
    x_input = x_z
    x_target = x_fc.clone()
    if return_graph_stats:
        return (x_input, x_target, edge_index, edge_attr, graph_attr)
    else:
        return (x_input, x_target, edge_index, edge_attr, None)

def compute_positional_encoding(num_nodes: int, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor]=None, pe_dim: int=16, pe_type: str='lap') -> torch.Tensor:
    if pe_type == 'none' or pe_dim <= 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float32)
    pe_type = pe_type.lower()
    device = edge_index.device
    edge_index_cpu = edge_index.cpu()
    if edge_weight is not None:
        edge_weight_cpu = edge_weight.view(-1).cpu()
    else:
        edge_weight_cpu = torch.ones(edge_index_cpu.size(1), dtype=torch.float64)
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    A[edge_index_cpu[0].long(), edge_index_cpu[1].long()] = edge_weight_cpu.double()
    deg = A.sum(dim=1)
    if pe_type == 'lap':
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        I = torch.eye(num_nodes, dtype=torch.float64)
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
        L = 0.5 * (L + L.t())
        L = L + 1e-06 * torch.eye(num_nodes, dtype=torch.float64)
        evals, evecs = torch.linalg.eigh(L)
        k = min(pe_dim, num_nodes)
        pe = evecs[:, :k]
    elif pe_type == 'rw':
        deg_inv = torch.pow(deg.clamp(min=1e-12), -1.0)
        deg_inv[torch.isinf(deg_inv)] = 0.0
        P = A * deg_inv.view(-1, 1)
        M = 0.5 * (P + P.t())
        M = 0.5 * (M + M.t())
        M = M + 1e-06 * torch.eye(num_nodes, dtype=torch.float64)
        evals, evecs = torch.linalg.eigh(M)
        k = min(pe_dim, num_nodes)
        pe = evecs[:, :k]
    else:
        return torch.zeros((num_nodes, 0), dtype=torch.float32)
    if pe.size(1) < pe_dim:
        pad = torch.zeros((num_nodes, pe_dim - pe.size(1)), dtype=torch.float64)
        pe = torch.cat([pe, pad], dim=1)
    return pe.to(device=device, dtype=torch.float32)
