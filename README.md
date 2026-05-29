# BrainNetGFM

**BrainNetGFM: A Graph-Based Foundation Model for Brain Network Construction Integrating Individualized Geometry and Joint Self-Supervised Learning**

Chunzhi Zhao, Tulay Adali, Jing Sui, Dailin Wen, Shile Qi*, Vince D. Calhoun

Published in *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)*. DOI: https://doi.org/10.1145/3770855.3818988

`*` Corresponding author.

## Overview

BrainNetGFM is a graph-based foundation model for brain network construction and downstream brain phenotyping. The implementation builds sparse graph representations from fMRI functional connectivity matrices, integrates individualized geometric information through ROI-level spatial distances, and learns reusable graph representations with joint self-supervised objectives.

The released code includes:

- self-supervised pretraining for fMRI functional connectivity graphs;
- graph construction with functional connectivity values, connection signs, ROI-level geometric distances, optional graph-level connectivity statistics, and optional positional encodings;
- a Graph Transformer backbone with GraphNorm, residual connections, Jumping Knowledge aggregation, mean-plus-max graph pooling, and optional graph attributes;
- masked node and edge reconstruction with learnable mask tokens;
- graph-level and node-level contrastive learning;
- downstream fine-tuning for classification and regression tasks, including disease classification, sex classification, symptom regression, and age regression;
- single-node multi-GPU training through PyTorch DistributedDataParallel.

An individualized brain atlas generation module is part of the broader research framework. The manuscript associated with that module is currently under review, and the corresponding source code will be released after the related paper is accepted or published.

## Pipeline and interpretability figures

Place the overview pipeline figure and interpretability figure in the `assets/` directory using the following file names:

<p align="center">
  <img src="assets/brainnetgfm_pipeline.png" width="850" alt="BrainNetGFM overall pipeline">
</p>

<p align="center"><b>Figure 1.</b> Overall BrainNetGFM workflow.</p>

<p align="center">
  <img src="assets/brainnetgfm_interpretability.png" width="850" alt="BrainNetGFM interpretability analysis">
</p>

<p align="center"><b>Figure 2.</b> Interpretability analysis of learned brain network representations.</p>

## Repository structure

```text
BrainNetGFM/
├── assets/
│   ├── brainnetgfm_pipeline.png
│   └── brainnetgfm_interpretability.png
├── brainnetgfm/
│   ├── __init__.py
│   ├── config_io.py
│   ├── configs.py
│   ├── datasets.py
│   ├── finetune_engine.py
│   ├── finetuning.py
│   ├── graph.py
│   ├── layers.py
│   ├── losses.py
│   ├── pretrain_engine.py
│   ├── pretraining.py
│   ├── splits.py
│   └── ssl_model.py
├── configs/
│   ├── finetune.yaml
│   └── pretrain.yaml
├── scripts/
│   ├── finetune.py
│   └── pretrain.py
├── .gitignore
├── CITATION.cff
├── README.md
├── pyproject.toml
└── requirements.txt
```

The implementation is organized so that `pretraining.py` and `finetuning.py` remain lightweight execution entry points. Configuration dataclasses are in `configs.py`, graph datasets are in `datasets.py`, graph construction utilities are in `graph.py`, model layers are in `layers.py`, self-supervised model logic is in `ssl_model.py`, losses and metrics are in `losses.py`, training loops are in `pretrain_engine.py` and `finetune_engine.py`, and data splitting utilities are in `splits.py`.

## Installation

Create a clean Python environment and install the required dependencies.

```bash
conda create -n brainnetgfm python=3.10 -y
conda activate brainnetgfm
pip install -r requirements.txt
```

Install PyTorch and PyTorch Geometric according to your CUDA version if your system requires version-specific wheels.

## Data format

### Pretraining data

The pretraining functional connectivity file should be a NumPy array stored as `.npy`. Supported shapes are:

- `[N, R, R]`;
- `[N, 1, R, R]`;
- `[N, R * R]`;
- object arrays where each element is an `[R, R]` matrix.

Here, `N` is the number of subjects and `R` is the number of ROIs.

### Downstream data

Fine-tuning requires:

- a functional connectivity `.npy` file with the same supported shapes as above;
- an Excel label file containing the downstream target columns.

The default label column names in `configs/finetune.yaml` are:

| Task | Default column | Target type |
|---|---:|---|
| Disease classification | `disease` | integer class label |
| Sex classification | `sex` | integer class label |
| Symptom regression | `symptom_score` | continuous value |
| Age regression | `age` | continuous value |

### ROI geometry file

If individualized or atlas-level ROI geometry is used, provide an Excel file in which the first three columns contain ROI coordinates. The number of coordinate rows should match the number of ROIs in the functional connectivity matrix.

## Pretraining

Edit `configs/pretrain.yaml` to point to your data and output checkpoint path.

Single GPU:

```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

Multi-GPU:

```bash
torchrun --standalone --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain.yaml
```

The default checkpoint path is:

```text
checkpoints/pretrain/brainnetgfm_pretrain.pt
```

The pretraining checkpoint stores the configuration dictionary together with the selected model state according to `save_model_mode`.

## Fine-tuning

Edit `configs/finetune.yaml` to set the downstream data path, label path, pretrained checkpoint path, enabled tasks, and model-selection metric.

Single GPU:

```bash
python scripts/finetune.py --config configs/finetune.yaml
```

Multi-GPU:

```bash
torchrun --standalone --nproc_per_node=8 scripts/finetune.py --config configs/finetune.yaml
```

Fine-tuned checkpoints are saved under:

```text
checkpoints/finetune/
```

For K-fold validation, fold-specific suffixes are appended to the checkpoint names.

## Core configuration options

### Graph construction

| Option | Description |
|---|---|
| `topk_ratio` | Ratio of strongest upper-triangular FC edges retained when constructing the graph. |
| `fisher_z` | Whether to apply Fisher z-transformation. Use `null` or `false` if the matrix is already transformed. |
| `use_abs_for_topk` | Whether to select edges by absolute FC magnitude. |
| `roi_mni_path` | Excel file containing ROI coordinates. |
| `use_graph_attr` | Whether to append graph-level FC mean and standard deviation to graph embeddings. |
| `pe_type` | Positional encoding type: `none`, `lap`, or `rw`. |
| `pe_dim` | Positional encoding dimension. |

### Backbone

| Option | Description |
|---|---|
| `gnn_hidden_dim` | Hidden dimension of the Graph Transformer encoder. |
| `num_gnn_layers` | Number of Graph Transformer layers. |
| `transformer_heads` | Number of attention heads. |
| `jk_mode` | Jumping Knowledge mode: `none`, `concat`, or `max`. |
| `node_norm_type` | Node feature projector normalization: `batchnorm`, `layernorm`, or `none`. |

### Self-supervised objectives

| Option | Description |
|---|---|
| `mask_ratio_node` | Node mask ratio for reconstruction. |
| `mask_ratio_edge` | Edge mask ratio for reconstruction. |
| `recon_target_mode` | Reconstruction target mode: `full`, `topk`, or `proj`. |
| `recon_loss_type` | Reconstruction loss: `sce`, `mse`, or `sce+mse`. |
| `contrastive_loss_type` | Graph-level contrastive loss: `nt_xent` or `info_nce`. |
| `node_cl_cross_subject_weight` | Weight for weak cross-subject same-ROI positives in node-level contrastive learning. |

## Notes on reproducibility

Set `seed` in the YAML configuration files before running experiments. For multi-GPU training, use `torchrun` so that the scripts correctly initialize DistributedDataParallel. Data files, generated checkpoints, and local logs should not be committed to the repository.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zhao2026brainnetgfm,
  title = {BrainNetGFM: A Graph-Based Foundation Model for Brain Network Construction Integrating Individualized Geometry and Joint Self-Supervised Learning},
  author = {Zhao, Chunzhi and Adali, Tulay and Sui, Jing and Wen, Dailin and Qi, Shile and Calhoun, Vince D.},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  series = {KDD '26},
  year = {2026},
  doi = {10.1145/3770855.3818988}
}
```

## Contact

For questions about the released implementation, please open an issue on GitHub.
