from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GraphNorm, TransformerConv, global_max_pool, global_mean_pool

class MLP(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int=2, dropout: float=0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class NodeFeatureProjector(nn.Module):

    def __init__(self, in_dim: int, norm_type: str='batchnorm'):
        super().__init__()
        hidden1 = max(int(round(in_dim * 1.5)), 1)
        hidden2 = max(int(round(hidden1 * 0.5)), 1)
        self.out_dim = hidden2
        norm_type = norm_type.lower()
        if norm_type == 'batchnorm':
            Norm1 = lambda dim: nn.BatchNorm1d(dim)
            Norm2 = lambda dim: nn.BatchNorm1d(dim)
        elif norm_type == 'layernorm':
            Norm1 = lambda dim: nn.LayerNorm(dim)
            Norm2 = lambda dim: nn.LayerNorm(dim)
        elif norm_type == 'none':
            Norm1 = lambda dim: nn.Identity()
            Norm2 = lambda dim: nn.Identity()
        else:
            raise ValueError(f'Unknown norm_type for NodeFeatureProjector: {norm_type}')
        self.net = nn.Sequential(nn.Linear(in_dim, hidden1), Norm1(hidden1), nn.ReLU(), nn.Linear(hidden1, hidden2), Norm2(hidden2), nn.ReLU())

    def forward(self, x):
        return self.net(x)

class GraphTransformerEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int=128, num_layers: int=3, dropout: float=0.1, heads: int=4, edge_dim: int=3, jk_mode: str='none'):
        super().__init__()
        assert num_layers >= 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.jk_mode = jk_mode.lower()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_channels = in_dim if layer == 0 else hidden_dim
            conv = TransformerConv(in_channels=in_channels, out_channels=hidden_dim, heads=heads, concat=False, edge_dim=edge_dim, dropout=dropout)
            self.layers.append(conv)
            self.norms.append(GraphNorm(in_channels))

    @property
    def out_dim(self) -> int:
        if self.jk_mode == 'concat':
            return self.hidden_dim * self.num_layers
        else:
            return self.hidden_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = x
        layer_outputs = []
        for i, conv in enumerate(self.layers):
            if batch is not None:
                h_norm = self.norms[i](h, batch)
            else:
                h_norm = self.norms[i](h)
            if edge_attr is not None:
                h_out = conv(h_norm, edge_index, edge_attr)
            else:
                h_out = conv(h_norm, edge_index)
            h_out = F.dropout(h_out, p=self.dropout, training=self.training)
            if h_out.size(-1) == h.size(-1):
                h = h + h_out
            else:
                h = h_out
            h = F.relu(h)
            layer_outputs.append(h)
        if self.jk_mode == 'concat':
            node_emb = torch.cat(layer_outputs, dim=-1)
        elif self.jk_mode == 'max':
            node_emb = torch.stack(layer_outputs, dim=0).max(dim=0).values
        else:
            node_emb = layer_outputs[-1]
        return node_emb

class GraphBackbone(nn.Module):

    def __init__(self, base_in_dim: int, pe_dim: int, use_graph_attr: bool, graph_attr_dim: int, gnn_hidden_dim: int=128, num_gnn_layers: int=3, dropout: float=0.1, heads: int=4, edge_dim: int=3, jk_mode: str='none', node_norm_type: str='batchnorm'):
        super().__init__()
        self.base_in_dim = base_in_dim
        self.pe_dim = pe_dim
        self.input_feat_dim = base_in_dim + pe_dim
        self.use_graph_attr = use_graph_attr
        self.graph_attr_dim = graph_attr_dim
        self.node_feat_proj = NodeFeatureProjector(self.input_feat_dim, norm_type=node_norm_type)
        self.encoder = GraphTransformerEncoder(in_dim=self.node_feat_proj.out_dim, hidden_dim=gnn_hidden_dim, num_layers=num_gnn_layers, dropout=dropout, heads=heads, edge_dim=edge_dim, jk_mode=jk_mode)
        self.encoder_out_dim = self.encoder.out_dim

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

def make_mlp_head(in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)

class MultiTaskGNN(nn.Module):

    def __init__(self, backbone: GraphBackbone, cfg: Any, graph_emb_dim: int):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.head_disease = make_mlp_head(graph_emb_dim, cfg.num_disease_classes, cfg.head_hidden_dim, cfg.head_dropout) if cfg.enable_disease else None
        self.head_sex = make_mlp_head(graph_emb_dim, cfg.num_sex_classes, cfg.head_hidden_dim, cfg.head_dropout) if cfg.enable_sex else None
        self.head_symptom = make_mlp_head(graph_emb_dim, 1, cfg.head_hidden_dim, cfg.head_dropout) if cfg.enable_symptom else None
        self.head_age = make_mlp_head(graph_emb_dim, 1, cfg.head_hidden_dim, cfg.head_dropout) if cfg.enable_age else None

    def forward(self, data: Batch) -> Dict[str, Any]:
        node_emb, graph_emb = self.backbone.encode_node_and_graph(data)
        out: Dict[str, Any] = {'graph_emb': graph_emb}
        out['disease'] = self.head_disease(graph_emb) if self.head_disease is not None else None
        out['sex'] = self.head_sex(graph_emb) if self.head_sex is not None else None
        out['symptom'] = self.head_symptom(graph_emb).squeeze(-1) if self.head_symptom is not None else None
        out['age'] = self.head_age(graph_emb).squeeze(-1) if self.head_age is not None else None
        return out
