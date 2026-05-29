from __future__ import annotations

from brainnetgfm.config_io import load_config, parse_config_path
from brainnetgfm.configs import PretrainConfig
from brainnetgfm.pretraining import main


if __name__ == '__main__':
    args = parse_config_path('BrainNetGFM pretraining')
    main(load_config(PretrainConfig, args.config))
