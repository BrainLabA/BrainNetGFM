from __future__ import annotations

from brainnetgfm.config_io import load_config, parse_config_path
from brainnetgfm.configs import FinetuneConfig
from brainnetgfm.finetuning import main


if __name__ == '__main__':
    args = parse_config_path('BrainNetGFM fine-tuning')
    main(load_config(FinetuneConfig, args.config))
