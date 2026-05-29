# BrainNetGFM

**BrainNetGFM: A Graph-Based Foundation Model for Brain Network Construction Integrating Individualized Geometry and Joint Self-Supervised Learning**

Published in *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)*. DOI: https://doi.org/10.1145/3770855.3818988


## Overview

BrainNetGFM is a graph-based foundation model for brain network construction and downstream brain phenotyping. The implementation builds sparse graph representations from fMRI functional connectivity matrices, integrates individualized geometric information through ROI-level spatial distances, and learns reusable graph representations with joint self-supervised objectives.

The released code includes:

- self-supervised pretraining for fMRI functional connectivity graphs;
- graph construction with functional connectivity values, connection signs, ROI-level geometric distances, optional graph-level connectivity statistics, and optional positional encodings;
- a Graph Transformer backbone with GraphNorm, residual connections, Jumping Knowledge aggregation, mean-plus-max graph pooling, and optional graph attributes;
- masked node and edge reconstruction with learnable mask tokens;
- graph-level and node-level contrastive learning;
- downstream fine-tuning for classification and regression tasks.

An individualized brain atlas generation module is part of the broader research framework. The manuscript associated with that module is currently under review, and the corresponding source code will be released after the related paper is accepted or published.

## Pipeline and interpretability figures

Place the overview pipeline figure and interpretability figure in the `assets/` directory using the following file names:

<p align="center">
  <img width="935" height="508" alt="image" src="https://github.com/user-attachments/assets/0ee0ab78-a221-49c1-87d1-bb4d54549f5b" />
</p>

<p align="center"><b>Figure 1.</b> Overall BrainNetGFM workflow.</p>

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

## Notes on reproducibility

Set `seed` in the YAML configuration files before running experiments. For multi-GPU training, use `torchrun` so that the scripts correctly initialize DistributedDataParallel. Data files, generated checkpoints, and local logs should not be committed to the repository.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zhao2026brainnetgfm,
  title = {BrainNetGFM: A Graph-Based Foundation Model for Brain Network Construction Integrating Individualized Geometry and Joint Self-Supervised Learning},
  author = {Zhao et al., Tulay Adali, Sui et al., Wen et al., Qi et al.and Vince D. Calhoun},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  series = {KDD '26},
  year = {2026},
  doi = {10.1145/3770855.3818988}
}
```

## Contact

<img width="119" height="41" alt="image" src="https://github.com/user-attachments/assets/3841e46d-0643-47ce-bdb0-a1ce6246ea99" />
This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivs International 4.0 License.
