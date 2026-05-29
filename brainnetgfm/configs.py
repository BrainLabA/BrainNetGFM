from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PretrainConfig:
    fc_npy_path: str = 'data/pretrain/fc.npy'
    save_encoder_path: str = 'checkpoints/pretrain/brainnetgfm_pretrain.pt'
    save_model_mode: str = 'full'
    roi_mni_path: str = 'data/roi/roi_mni.xlsx'
    train_ratio: float = 0.8
    batch_size_per_gpu: int = 64
    num_epochs: int = 100
    lr: float = 0.0005
    weight_decay: float = 1e-05
    seed: int = 42
    num_workers: int = 8
    pin_memory: bool = True
    topk_ratio: float = 0.1
    fisher_z: Optional[bool] = None
    use_abs_for_topk: bool = True
    alpha_recon: float = 1.0
    beta_cl: float = 1.0
    mask_ratio_node: float = 0.5
    mask_ratio_edge: float = 0.5
    edge_loss_weight: float = 1.0
    sce_gamma: float = 2.0
    drop_node_ratio: float = 0.3
    drop_edge_ratio: float = 0.3
    feat_mask_ratio: float = 0.3
    temperature: float = 0.25
    gnn_hidden_dim: int = 256
    proj_dim: int = 256
    recon_hidden_dim: int = 256
    num_gnn_layers: int = 3
    num_decoder_layers: int = 1
    dropout: float = 0.2
    transformer_heads: int = 4
    max_grad_norm: float = 3.0
    ddp_backend: str = 'nccl'
    use_graph_attr: bool = True
    pe_type: str = 'rw'
    pe_dim: int = 256
    jk_mode: str = 'concat'
    node_norm_type: str = 'layernorm'
    node_mask_edge_mode: str = 'token'
    recon_target_mode: str = 'full'
    recon_topk_ratio: float = 0.3
    recon_proj_dim: int = 128
    recon_loss_type: str = 'sce+mse'
    node_cl_cross_subject_weight: float = 0.2
    contrastive_loss_type: str = 'nt_xent'


@dataclass
class FinetuneConfig:
    fc_npy_path: str = 'data/downstream/fc.npy'
    label_excel_path: str = 'data/downstream/labels.xlsx'
    pretrained_encoder_path: str = 'checkpoints/pretrain/brainnetgfm_pretrain.pt'
    save_encoder_path: str = 'checkpoints/finetune/finetuned_backbone.pt'
    save_full_model_path: str = 'checkpoints/finetune/finetuned_full_model.pt'
    train_ratio: float = 0.8
    num_folds: int = 10
    batch_size_per_gpu: int = 2
    disease_col: str = 'disease'
    sex_col: str = 'sex'
    symptom_col: str = 'symptom_score'
    age_col: str = 'age'
    enable_disease: bool = True
    enable_sex: bool = False
    enable_symptom: bool = False
    enable_age: bool = False
    num_disease_classes: int = 2
    num_sex_classes: int = 2
    lambda_disease: float = 1.0
    lambda_sex: float = 1.0
    lambda_symptom: float = 1.0
    lambda_age: float = 1.0
    num_epochs: int = 100
    lr: float = 0.0005
    weight_decay: float = 1e-05
    seed: int = 42
    freeze_encoder: bool = False
    lr_encoder: float = 3e-05
    lr_head: float = 0.0003
    use_class_weights: bool = False
    head_hidden_dim: int = 256
    head_dropout: float = 0.2
    topk_ratio: float = 0.3
    fisher_z: Optional[bool] = None
    use_abs_for_topk: bool = True
    roi_mni_path: str = 'data/roi/roi_mni.xlsx'
    use_graph_attr: bool = True
    pe_type: str = 'rw'
    pe_dim: int = 256
    jk_mode: str = 'max'
    node_norm_type: str = 'layernorm'
    transformer_heads: int = 4
    gnn_hidden_dim: int = 256
    num_gnn_layers: int = 3
    dropout: float = 0.2
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    disease_class_weights: Optional[Tuple[float, ...]] = None
    sex_class_weights: Optional[Tuple[float, ...]] = None
    monitor_metric: str = 'disease_auc'
    monitor_mode: str = 'max'
    ddp_backend: str = 'nccl'
