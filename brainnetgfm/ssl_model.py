from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GraphNorm, TransformerConv, global_max_pool, global_mean_pool

from .layers import MLP, GraphTransformerEncoder, NodeFeatureProjector
from .losses import info_nce_loss, node_supervised_contrastive_loss, nt_xent_loss, sce_loss


class GraphSSLModel(nn.Module):

    def __init__(self, node_in_dim: int, fc_dim: int, pe_dim: int, use_graph_attr: bool, graph_attr_dim: int, gnn_hidden_dim: int=128, proj_dim: int=128, recon_hidden_dim: int=128, num_gnn_layers: int=3, num_decoder_layers: int=2, dropout: float=0.1, heads: int=4, edge_dim: int=3, jk_mode: str='none', node_norm_type: str='batchnorm', decoder_hidden_dim: int=128, recon_target_mode: str='full', recon_topk_ratio: float=0.3, recon_proj_dim: int=64, node_mask_edge_mode: str='token', node_cl_cross_subject_weight: float=0.0, recon_loss_type: str='sce', contrastive_loss_type: str='nt_xent'):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.fc_dim = fc_dim
        self.pe_dim = pe_dim
        self.input_feat_dim = node_in_dim + pe_dim
        self.use_graph_attr = use_graph_attr
        self.graph_attr_dim = graph_attr_dim
        self.recon_target_mode = recon_target_mode.lower()
        self.recon_topk_ratio = recon_topk_ratio
        self.recon_proj_dim = recon_proj_dim
        self.recon_loss_type = recon_loss_type.lower()
        self.node_mask_edge_mode = node_mask_edge_mode.lower()
        self.node_cl_cross_subject_weight = node_cl_cross_subject_weight
        self.contrastive_loss_type = contrastive_loss_type.lower()
        self.node_feat_proj = NodeFeatureProjector(self.input_feat_dim, norm_type=node_norm_type)
        self.encoder = GraphTransformerEncoder(in_dim=self.node_feat_proj.out_dim, hidden_dim=gnn_hidden_dim, num_layers=num_gnn_layers, dropout=dropout, heads=heads, edge_dim=edge_dim, jk_mode=jk_mode)
        self.encoder_out_dim = self.encoder.out_dim
        self.decoder_convs = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        assert num_decoder_layers >= 1
        for layer in range(num_decoder_layers):
            in_channels = self.encoder_out_dim if layer == 0 else decoder_hidden_dim
            conv = TransformerConv(in_channels=in_channels, out_channels=decoder_hidden_dim, heads=heads, concat=False, edge_dim=edge_dim, dropout=0.0)
            self.decoder_convs.append(conv)
            self.decoder_norms.append(GraphNorm(decoder_hidden_dim))
        self.node_decoder_linear = nn.Linear(decoder_hidden_dim, fc_dim)
        if self.recon_target_mode == 'proj':
            self.target_proj = nn.Sequential(nn.Linear(fc_dim, recon_proj_dim), nn.ReLU(inplace=True), nn.Linear(recon_proj_dim, recon_proj_dim))
        else:
            self.target_proj = None
        self.edge_decoder = MLP(in_dim=2 * self.encoder_out_dim, hidden_dim=recon_hidden_dim, out_dim=edge_dim, num_layers=2, dropout=dropout)
        graph_proj_in_dim = 2 * self.encoder_out_dim + (self.graph_attr_dim if self.use_graph_attr else 0)
        self.graph_proj_head = MLP(in_dim=graph_proj_in_dim, hidden_dim=proj_dim, out_dim=proj_dim, num_layers=2, dropout=dropout)
        self.node_proj_head = MLP(in_dim=self.encoder_out_dim, hidden_dim=proj_dim, out_dim=proj_dim, num_layers=2, dropout=dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, node_in_dim))
        self.mask_edge_token = nn.Parameter(torch.zeros(1, edge_dim))

    def encode_node_and_graph(self, data: Batch):
        x = data.x
        batch = getattr(data, 'batch', None)
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        if hasattr(data, 'pe') and data.pe is not None and (self.pe_dim > 0):
            pe = data.pe
            if pe.size(0) != x.size(0):
                raise ValueError('PE and X have mismatched node counts.')
            x_in = torch.cat([x, pe], dim=-1)
        else:
            x_in = x
        x_proj = self.node_feat_proj(x_in)
        node_emb = self.encoder(x_proj, edge_index, edge_attr=edge_attr, batch=batch)
        if batch is not None:
            g_mean = global_mean_pool(node_emb, batch)
            g_max = global_max_pool(node_emb, batch)
            graph_emb = torch.cat([g_mean, g_max], dim=-1)
            if self.use_graph_attr and hasattr(data, 'graph_attr'):
                g_attr = data.graph_attr
                if g_attr.dim() == 1:
                    g_attr = g_attr.unsqueeze(0)
                graph_emb = torch.cat([graph_emb, g_attr], dim=-1)
        else:
            g_mean = node_emb.mean(dim=0, keepdim=True)
            g_max = node_emb.max(dim=0, keepdim=True).values
            graph_emb = torch.cat([g_mean, g_max], dim=-1)
            if self.use_graph_attr and hasattr(data, 'graph_attr'):
                g_attr = data.graph_attr
                if g_attr.dim() == 1:
                    g_attr = g_attr.unsqueeze(0)
                graph_emb = torch.cat([graph_emb, g_attr], dim=-1)
        return (node_emb, graph_emb)

    def decode_nodes(self, node_emb, edge_index, edge_attr, batch):
        h = node_emb
        for i, conv in enumerate(self.decoder_convs):
            if edge_attr is not None:
                h_out = conv(h, edge_index, edge_attr)
            else:
                h_out = conv(h, edge_index)
            h_out = F.dropout(h_out, p=conv.dropout if hasattr(conv, 'dropout') else 0.0, training=self.training)
            if batch is not None:
                h_out = self.decoder_norms[i](h_out, batch)
            else:
                h_out = self.decoder_norms[i](h_out)
            h_out = F.relu(h_out)
            h = h_out
        return h

    def _compute_node_recon_loss(self, x_recon_full: torch.Tensor, x_target_full: torch.Tensor, node_mask: torch.Tensor, data: Batch, sce_gamma: float) -> torch.Tensor:
        device = x_recon_full.device
        if not node_mask.any():
            return torch.tensor(0.0, device=device)
        x_r_full = x_recon_full[node_mask]
        x_t_full = x_target_full[node_mask]
        mode = self.recon_target_mode
        if mode == 'full':
            z_r = x_r_full
            z_t = x_t_full
        elif mode == 'topk':
            if not hasattr(data, 'topk_idx'):
                raise ValueError("recon_target_mode='topk' requires topk_idx in the batch.")
            topk_idx_all = data.topk_idx.to(device)
            topk_idx = topk_idx_all[node_mask]
            z_r = torch.gather(x_r_full, dim=1, index=topk_idx)
            z_t = torch.gather(x_t_full, dim=1, index=topk_idx)
        elif mode == 'proj':
            if self.target_proj is None:
                raise ValueError("recon_target_mode='proj' requires initialized target_proj.")
            z_r = self.target_proj(x_r_full)
            z_t = self.target_proj(x_t_full)
        else:
            raise ValueError(f'Unknown recon_target_mode: {mode}')
        loss_type = getattr(self, 'recon_loss_type', 'sce')
        if loss_type == 'sce':
            return sce_loss(z_r, z_t, gamma=sce_gamma)
        elif loss_type == 'mse':
            return F.mse_loss(z_r, z_t)
        elif loss_type in ('sce+mse', 'mse+sce'):
            return sce_loss(z_r, z_t, gamma=sce_gamma) + F.mse_loss(z_r, z_t)
        else:
            raise ValueError(f'Unknown recon_loss_type: {loss_type}')

    def masked_recon_loss(self, data: Batch, mask_ratio_node: float=0.3, mask_ratio_edge: float=0.3, edge_loss_weight: float=1.0, sce_gamma: float=2.0):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        device = x.device
        num_nodes = x.size(0)
        if mask_ratio_node > 0.0:
            node_mask = torch.rand(num_nodes, device=device) < mask_ratio_node
            if not node_mask.any():
                node_mask[torch.randint(0, num_nodes, (1,), device=device)] = True
        else:
            node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        x_masked = x.clone()
        if hasattr(data, 'x_target'):
            x_original_full = data.x_target.to(device)
        else:
            x_original_full = x.clone()
        if node_mask.any():
            x_masked[node_mask] = self.mask_token.to(device)
        edge_mask = None
        edge_attr_masked = None
        edge_attr_original = None
        edge_index_effective = edge_index
        if edge_attr is not None:
            src, dst = edge_index
            node_mask = node_mask.to(device)
            mode = self.node_mask_edge_mode
            if mode == 'drop':
                keep_edges = ~(node_mask[src] | node_mask[dst])
                edge_index_kept = edge_index[:, keep_edges]
                edge_attr_kept = edge_attr[keep_edges]
                num_edges_kept = edge_attr_kept.size(0)
                if mask_ratio_edge > 0.0 and num_edges_kept > 0:
                    edge_mask = torch.rand(num_edges_kept, device=device) < mask_ratio_edge
                    if not edge_mask.any():
                        edge_mask[torch.randint(0, num_edges_kept, (1,), device=device)] = True
                else:
                    edge_mask = torch.zeros(num_edges_kept, dtype=torch.bool, device=device)
                edge_attr_masked = edge_attr_kept.clone()
                edge_attr_original = edge_attr_kept.clone()
                if edge_mask.any():
                    edge_attr_masked[edge_mask] = self.mask_edge_token.to(device)
                edge_index_effective = edge_index_kept
            else:
                num_edges = edge_attr.size(0)
                if mask_ratio_edge > 0.0 and num_edges > 0:
                    edge_mask = torch.rand(num_edges, device=device) < mask_ratio_edge
                    if not edge_mask.any():
                        edge_mask[torch.randint(0, num_edges, (1,), device=device)] = True
                else:
                    edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
                if mode == 'token':
                    incident_mask = node_mask[src] | node_mask[dst]
                    edge_mask = edge_mask | incident_mask
                edge_attr_masked = edge_attr.clone()
                edge_attr_original = edge_attr.clone()
                if edge_mask.any():
                    edge_attr_masked[edge_mask] = self.mask_edge_token.to(device)
                edge_index_effective = edge_index
        else:
            edge_mask = None
            edge_attr_masked = None
            edge_attr_original = None
            edge_index_effective = edge_index
        masked_data = Data(x=x_masked, edge_index=edge_index_effective, edge_attr=edge_attr_masked, batch=batch)
        if hasattr(data, 'pe'):
            masked_data.pe = data.pe
        if hasattr(data, 'graph_attr'):
            masked_data.graph_attr = data.graph_attr
        node_emb, _ = self.encode_node_and_graph(masked_data)
        node_dec_emb = self.decode_nodes(node_emb, edge_index_effective, edge_attr_masked, batch)
        x_recon_full = self.node_decoder_linear(node_dec_emb)
        node_loss = self._compute_node_recon_loss(x_recon_full=x_recon_full, x_target_full=x_original_full, node_mask=node_mask, data=data, sce_gamma=sce_gamma)
        if edge_attr_original is not None and edge_mask is not None and edge_mask.any():
            src_eff, dst_eff = edge_index_effective
            edge_rep = torch.cat([node_emb[src_eff], node_emb[dst_eff]], dim=-1)
            edge_recon = self.edge_decoder(edge_rep)
            edge_loss = F.mse_loss(edge_recon[edge_mask], edge_attr_original[edge_mask])
        else:
            edge_loss = torch.tensor(0.0, device=device)
        total_recon_loss = node_loss + edge_loss_weight * edge_loss
        return (total_recon_loss, node_loss, edge_loss)

    def _random_node_mask(self, data: Data, mask_ratio: float=0.1) -> Data:
        x, edge_index, edge_attr = (data.x, data.edge_index, data.edge_attr)
        num_nodes = x.size(0)
        device = x.device
        if mask_ratio <= 0.0 or num_nodes == 0:
            return data
        node_mask = torch.rand(num_nodes, device=device) < mask_ratio
        if not node_mask.any():
            node_mask[torch.randint(0, num_nodes, (1,), device=device)] = True
        x_new = x.clone()
        x_new[node_mask] = self.mask_token.to(device)
        if edge_attr is not None:
            src, dst = edge_index
            incident_mask = node_mask[src] | node_mask[dst]
            if incident_mask.any():
                edge_attr_new = edge_attr.clone()
                edge_attr_new[incident_mask] = self.mask_edge_token.to(device)
            else:
                edge_attr_new = edge_attr
        else:
            edge_attr_new = edge_attr
        new_data = Data(x=x_new, edge_index=edge_index, edge_attr=edge_attr_new)
        if hasattr(data, 'pe'):
            new_data.pe = data.pe
        if hasattr(data, 'graph_attr'):
            new_data.graph_attr = data.graph_attr
        if hasattr(data, 'roi'):
            new_data.roi = data.roi
        if hasattr(data, 'x_target'):
            new_data.x_target = data.x_target
        if hasattr(data, 'topk_idx'):
            new_data.topk_idx = data.topk_idx
        return new_data

    def _random_edge_mask(self, data: Data, mask_ratio: float=0.1) -> Data:
        x, edge_index, edge_attr = (data.x, data.edge_index, data.edge_attr)
        device = x.device
        if edge_attr is None:
            return data
        num_edges = edge_index.size(1)
        if mask_ratio <= 0.0 or num_edges == 0:
            return data
        edge_mask = torch.rand(num_edges, device=device) < mask_ratio
        if not edge_mask.any():
            edge_mask[torch.randint(0, num_edges, (1,), device=device)] = True
        edge_attr_new = edge_attr.clone()
        edge_attr_new[edge_mask] = self.mask_edge_token.to(device)
        new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_new)
        if hasattr(data, 'pe'):
            new_data.pe = data.pe
        if hasattr(data, 'graph_attr'):
            new_data.graph_attr = data.graph_attr
        if hasattr(data, 'roi'):
            new_data.roi = data.roi
        if hasattr(data, 'x_target'):
            new_data.x_target = data.x_target
        if hasattr(data, 'topk_idx'):
            new_data.topk_idx = data.topk_idx
        return new_data

    def _random_feature_mask(self, data: Data, mask_ratio: float=0.1) -> Data:
        x, edge_index, edge_attr = (data.x, data.edge_index, data.edge_attr)
        num_nodes = x.size(0)
        device = x.device
        if mask_ratio <= 0.0 or num_nodes == 0:
            return data
        node_mask = torch.rand(num_nodes, device=device) < mask_ratio
        if not node_mask.any():
            node_mask[torch.randint(0, num_nodes, (1,), device=device)] = True
        x_new = x.clone()
        x_new[node_mask] = self.mask_token.to(device)
        new_data = Data(x=x_new, edge_index=edge_index, edge_attr=edge_attr)
        if hasattr(data, 'pe'):
            new_data.pe = data.pe
        if hasattr(data, 'graph_attr'):
            new_data.graph_attr = data.graph_attr
        if hasattr(data, 'roi'):
            new_data.roi = data.roi
        if hasattr(data, 'x_target'):
            new_data.x_target = data.x_target
        if hasattr(data, 'topk_idx'):
            new_data.topk_idx = data.topk_idx
        return new_data

    def _make_augmented_pair(self, batch_data: Batch, node_mask_ratio: float=0.1, edge_mask_ratio: float=0.2, feat_mask_ratio: float=0.1):
        data_list = batch_data.to_data_list()
        aug_list1 = []
        aug_list2 = []
        for data in data_list:
            data = data.to(batch_data.x.device)
            v1 = data.clone()
            v1 = self._random_node_mask(v1, node_mask_ratio)
            v1 = self._random_edge_mask(v1, edge_mask_ratio)
            v1 = self._random_feature_mask(v1, feat_mask_ratio)
            v2 = data.clone()
            v2 = self._random_edge_mask(v2, edge_mask_ratio)
            v2 = self._random_feature_mask(v2, feat_mask_ratio)
            v2 = self._random_node_mask(v2, node_mask_ratio)
            aug_list1.append(v1)
            aug_list2.append(v2)
        view1 = Batch.from_data_list(aug_list1)
        view2 = Batch.from_data_list(aug_list2)
        return (view1, view2)

    def _make_node_level_augmented_pair(self, batch_data: Batch, edge_mask_ratio: float=0.1, feat_mask_ratio: float=0.1):
        data_list = batch_data.to_data_list()
        aug_list1 = []
        aug_list2 = []
        for data in data_list:
            data = data.to(batch_data.x.device)
            v1 = data.clone()
            v1 = self._random_edge_mask(v1, edge_mask_ratio)
            v1 = self._random_feature_mask(v1, feat_mask_ratio)
            v2 = data.clone()
            v2 = self._random_edge_mask(v2, edge_mask_ratio)
            v2 = self._random_feature_mask(v2, feat_mask_ratio)
            aug_list1.append(v1)
            aug_list2.append(v2)
        view1 = Batch.from_data_list(aug_list1)
        view2 = Batch.from_data_list(aug_list2)
        return (view1, view2)

    def contrastive_loss(self, data_batch: Batch, drop_node_ratio: float=0.1, drop_edge_ratio: float=0.2, feat_mask_ratio: float=0.1, temperature: float=0.2):
        view1_g, view2_g = self._make_augmented_pair(data_batch, node_mask_ratio=drop_node_ratio, edge_mask_ratio=drop_edge_ratio, feat_mask_ratio=feat_mask_ratio)
        _, g1 = self.encode_node_and_graph(view1_g)
        _, g2 = self.encode_node_and_graph(view2_g)
        zg1 = self.graph_proj_head(g1)
        zg2 = self.graph_proj_head(g2)
        if self.contrastive_loss_type == 'nt_xent':
            graph_cl = nt_xent_loss(zg1, zg2, temperature=temperature)
        elif self.contrastive_loss_type == 'info_nce':
            graph_cl = info_nce_loss(zg1, zg2, temperature=temperature)
        else:
            raise ValueError(f'Unknown contrastive_loss_type: {self.contrastive_loss_type}')
        view1_n, view2_n = self._make_node_level_augmented_pair(data_batch, edge_mask_ratio=drop_edge_ratio, feat_mask_ratio=feat_mask_ratio)
        n1, _ = self.encode_node_and_graph(view1_n)
        n2, _ = self.encode_node_and_graph(view2_n)
        zn1 = self.node_proj_head(n1)
        zn2 = self.node_proj_head(n2)
        if hasattr(view1_n, 'roi'):
            roi_ids = view1_n.roi.to(zn1.device)
        elif hasattr(data_batch, 'roi'):
            roi_ids = data_batch.roi.to(zn1.device)
        else:
            raise ValueError('ROI ids are required for node-level supervised contrastive loss.')
        graph_ids = view1_n.batch.to(zn1.device)
        node_cl = node_supervised_contrastive_loss(zn1, zn2, roi_ids, graph_ids, temperature=temperature, cross_subject_pos_weight=self.node_cl_cross_subject_weight)
        total_cl = 0.5 * (graph_cl + node_cl)
        return (total_cl, graph_cl, node_cl)

    def forward(self, data: Batch, mask_ratio_node: float=0.3, mask_ratio_edge: float=0.3, edge_loss_weight: float=1.0, drop_node_ratio: float=0.1, drop_edge_ratio: float=0.2, feat_mask_ratio: float=0.1, temperature: float=0.2, sce_gamma: float=2.0):
        recon_loss, node_recon_loss, edge_recon_loss = self.masked_recon_loss(data, mask_ratio_node=mask_ratio_node, mask_ratio_edge=mask_ratio_edge, edge_loss_weight=edge_loss_weight, sce_gamma=sce_gamma)
        cl_loss_total, graph_cl, node_cl = self.contrastive_loss(data, drop_node_ratio=drop_node_ratio, drop_edge_ratio=drop_edge_ratio, feat_mask_ratio=feat_mask_ratio, temperature=temperature)
        return (recon_loss, cl_loss_total, node_recon_loss, edge_recon_loss)

