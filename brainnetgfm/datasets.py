from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .configs import FinetuneConfig
from .graph import build_graph_from_fc, compute_positional_encoding


def _load_fc_array(path: str, mmap_mode: Optional[str] = None):
    try:
        raw = np.load(path, allow_pickle=True, mmap_mode=mmap_mode)
    except ValueError:
        raw = np.load(path, allow_pickle=True, mmap_mode=None)
    is_object_array = False
    if raw.ndim == 3 and raw.shape[1] == raw.shape[2]:
        fc_all = raw
    elif raw.ndim == 4 and raw.shape[1] == 1 and raw.shape[2] == raw.shape[3]:
        fc_all = raw
    elif raw.ndim == 2:
        fc_all = raw
    elif raw.ndim == 1 and isinstance(raw[0], (np.ndarray, list)):
        fc_all = raw
        is_object_array = True
    else:
        raise ValueError(f'Unsupported FC npy shape {raw.shape}. Expected [N,R,R], [N,1,R,R], [N,R*R], or an object array.')
    return raw, fc_all, is_object_array


def _fc_count(fc_all, is_object_array: bool) -> int:
    if is_object_array:
        return len(fc_all)
    return fc_all.shape[0]


def _get_fc_matrix(fc_all, is_object_array: bool, idx: int) -> np.ndarray:
    if is_object_array:
        fc = np.asarray(fc_all[idx], dtype=np.float32)
    else:
        fc = np.asarray(fc_all[idx])
        if fc.ndim == 3 and fc.shape[0] == 1 and fc.shape[1] == fc.shape[2]:
            fc = fc[0]
        elif fc.ndim == 1:
            F = fc.shape[0]
            R = int(math.sqrt(F))
            if R * R != F:
                raise ValueError(f'Sample {idx}: cannot reshape ({F},) into [R,R].')
            fc = fc.reshape(R, R)
    if not (fc.ndim == 2 and fc.shape[0] == fc.shape[1]):
        raise ValueError(f'Sample {idx}: expected a square FC matrix, got shape {fc.shape}.')
    return fc.astype(np.float32)


def _load_roi_geometry(roi_mni_path: Optional[str], num_rois: int):
    if roi_mni_path is None or roi_mni_path == '':
        return None, None
    try:
        df = pd.read_excel(roi_mni_path)
        coords = df.iloc[:, :3].values.astype(np.float32)
        if coords.shape[0] != num_rois:
            print(f'Warning: ROI coords row {coords.shape[0]} != FC size {num_rois}; edge distances will be zeros.')
            return None, None
        diff = coords[:, None, :] - coords[None, :, :]
        dist_mat = np.sqrt((diff ** 2).sum(axis=-1))
        triu = np.triu_indices(num_rois, k=1)
        dist_vals = dist_mat[triu]
        mu = dist_vals.mean()
        sigma = dist_vals.std()
        if sigma < 1e-06:
            sigma = 1.0
        return coords, ((dist_mat - mu) / sigma).astype(np.float32)
    except Exception as exc:
        print(f'Failed to load ROI MNI coordinates from {roi_mni_path}: {exc}')
        return None, None


def _attach_graph_encoding(data: Data, edge_attr: torch.Tensor, pe_type: str, pe_dim: int) -> Data:
    if pe_type != 'none' and pe_dim > 0:
        data.pe = compute_positional_encoding(num_nodes=data.x.size(0), edge_index=data.edge_index, edge_weight=edge_attr[:, 0], pe_dim=pe_dim, pe_type=pe_type)
    return data


class FMriFcDataset(Dataset):

    def __init__(self, npy_path: str, topk_ratio: float = 0.2, fisher_z: Optional[bool] = True, use_abs_for_topk: bool = True, use_graph_attr: bool = False, roi_mni_path: Optional[str] = None, pe_type: str = 'none', pe_dim: int = 16, recon_target_mode: str = 'full', recon_topk_ratio: float = 0.3):
        super().__init__()
        raw, fc_all, is_object_array = _load_fc_array(npy_path, mmap_mode='r')
        print(f'Loaded npy (mmap) from {npy_path}, raw.shape = {raw.shape}, ndim = {raw.ndim}, dtype = {raw.dtype}')
        if is_object_array:
            print('Detected object array: each element will be treated separately.')
        self.fc_all = fc_all
        self.is_object_array = is_object_array
        self.topk_ratio = topk_ratio
        self.fisher_z = fisher_z
        self.use_abs_for_topk = use_abs_for_topk
        self.use_graph_attr = use_graph_attr
        self.pe_type = pe_type.lower()
        self.pe_dim = pe_dim if self.pe_type != 'none' else 0
        self.recon_target_mode = recon_target_mode.lower()
        self.recon_topk_ratio = recon_topk_ratio
        num_rois = self._get_fc_matrix_internal(0).shape[0]
        self.roi_coords, self.roi_dist = _load_roi_geometry(roi_mni_path, num_rois)
        print(f'FMriFcDataset initialized. Total graphs: {len(self)}')

    def __len__(self):
        return _fc_count(self.fc_all, self.is_object_array)

    def _get_fc_matrix_internal(self, idx: int) -> np.ndarray:
        return _get_fc_matrix(self.fc_all, self.is_object_array, idx)

    def __getitem__(self, idx):
        fc = self._get_fc_matrix_internal(idx)
        x_input, x_target, edge_index, edge_attr, graph_attr = build_graph_from_fc(fc, topk_ratio=self.topk_ratio, fisher_z=self.fisher_z, use_abs_for_topk=self.use_abs_for_topk, roi_coords=self.roi_coords, roi_dist=self.roi_dist, return_graph_stats=self.use_graph_attr)
        data = Data(x=x_input, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x_input.shape[0])
        data.x_target = x_target
        if graph_attr is not None:
            data.graph_attr = graph_attr
        data.roi = torch.arange(x_input.shape[0], dtype=torch.long)
        if self.recon_target_mode == 'topk' and self.recon_topk_ratio > 0:
            R = x_target.size(1)
            k = max(int(round(R * self.recon_topk_ratio)), 1)
            fc_abs = x_target.abs()
            idx_arange = torch.arange(R)
            fc_abs[idx_arange, idx_arange] = 0.0
            _, topk_idx = torch.topk(fc_abs, k=k, dim=1)
            data.topk_idx = topk_idx.long()
        return _attach_graph_encoding(data, edge_attr, self.pe_type, self.pe_dim)


class DownstreamFMriDataset(Dataset):

    def __init__(self, cfg: FinetuneConfig, pretrain_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        raw, fc_all, is_object_array = _load_fc_array(cfg.fc_npy_path, mmap_mode=None)
        print(f'[Downstream] Loaded FC npy from {cfg.fc_npy_path}, shape={raw.shape}, ndim={raw.ndim}')
        if raw.ndim == 4 and raw.shape[1] == 1 and raw.shape[2] == raw.shape[3]:
            fc_all = raw[:, 0].astype(np.float32)
        elif raw.ndim == 2 and not is_object_array:
            N, F = raw.shape
            R = int(math.sqrt(F))
            if R * R != F:
                raise ValueError(f'Cannot reshape ({N}, {F}) into [N, R, R].')
            fc_all = raw.reshape(N, R, R).astype(np.float32)
        elif raw.ndim == 3 and raw.shape[1] == raw.shape[2]:
            fc_all = raw.astype(np.float32)
        elif is_object_array:
            print('[Downstream] Detected object array for FC.')
        self.fc_all = fc_all
        self.is_object_array = is_object_array
        self.labels = pd.read_excel(cfg.label_excel_path)
        if len(self.labels) != len(self):
            raise ValueError(f'FC count ({len(self)}) != label rows ({len(self.labels)}).')
        self.cfg = cfg
        self.pre_cfg = pretrain_cfg or {}
        self.topk_ratio = float(self.pre_cfg.get('topk_ratio', cfg.topk_ratio))
        self.fisher_z = self.pre_cfg.get('fisher_z', cfg.fisher_z)
        self.use_abs_for_topk = bool(self.pre_cfg.get('use_abs_for_topk', cfg.use_abs_for_topk))
        self.use_graph_attr = bool(self.pre_cfg.get('use_graph_attr', cfg.use_graph_attr))
        self.pe_type = str(self.pre_cfg.get('pe_type', cfg.pe_type)).lower()
        self.pe_dim = int(self.pre_cfg.get('pe_dim', cfg.pe_dim))
        self.roi_mni_path = self.pre_cfg.get('roi_mni_path', getattr(cfg, 'roi_mni_path', None))
        keep_mask = np.ones(len(self.labels), dtype=bool)
        if cfg.enable_symptom:
            if cfg.symptom_col not in self.labels.columns:
                raise ValueError(f"symptom_col '{cfg.symptom_col}' not in label excel columns.")
            keep_mask &= ~self.labels[cfg.symptom_col].isna().to_numpy()
        if cfg.enable_age:
            if cfg.age_col not in self.labels.columns:
                raise ValueError(f"age_col '{cfg.age_col}' not in label excel columns.")
            keep_mask &= ~self.labels[cfg.age_col].isna().to_numpy()
        if not keep_mask.all():
            before_n = len(self.labels)
            self.labels = self.labels.loc[keep_mask].reset_index(drop=True)
            self.fc_all = self.fc_all[keep_mask]
            print(f'[Downstream] Filtered NaN in continuous labels: {before_n} -> {len(self.labels)} samples')
        num_rois = self._get_fc_matrix(0).shape[0]
        self.roi_coords, self.roi_dist = _load_roi_geometry(self.roi_mni_path, num_rois)

    def __len__(self):
        return _fc_count(self.fc_all, self.is_object_array)

    def _get_fc_matrix(self, idx: int) -> np.ndarray:
        return _get_fc_matrix(self.fc_all, self.is_object_array, idx)

    def __getitem__(self, idx):
        fc = self._get_fc_matrix(idx)
        x, _, edge_index, edge_attr, graph_attr = build_graph_from_fc(fc, topk_ratio=self.topk_ratio, fisher_z=self.fisher_z, use_abs_for_topk=self.use_abs_for_topk, roi_coords=self.roi_coords, roi_dist=self.roi_dist, return_graph_stats=self.use_graph_attr)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])
        if graph_attr is not None:
            data.graph_attr = graph_attr
        data = _attach_graph_encoding(data, edge_attr, self.pe_type, self.pe_dim)
        row = self.labels.iloc[idx]
        y_disease = -1
        y_sex = -1
        y_symptom = np.nan
        y_age = np.nan
        if self.cfg.enable_disease and self.cfg.disease_col in row:
            value = row[self.cfg.disease_col]
            y_disease = -1 if pd.isna(value) else int(value)
        if self.cfg.enable_sex and self.cfg.sex_col in row:
            value = row[self.cfg.sex_col]
            y_sex = -1 if pd.isna(value) else int(value)
        if self.cfg.enable_symptom and self.cfg.symptom_col in row:
            value = row[self.cfg.symptom_col]
            y_symptom = np.nan if pd.isna(value) else float(value)
        if self.cfg.enable_age and self.cfg.age_col in row:
            value = row[self.cfg.age_col]
            y_age = np.nan if pd.isna(value) else float(value)
        data.y_disease = torch.tensor(y_disease, dtype=torch.long)
        data.y_sex = torch.tensor(y_sex, dtype=torch.long)
        data.y_symptom = torch.tensor(y_symptom, dtype=torch.float32)
        data.y_age = torch.tensor(y_age, dtype=torch.float32)
        return data
