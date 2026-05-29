from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any, Type, TypeVar

import yaml

T = TypeVar('T')


def load_config(config_cls: Type[T], path: str | None) -> T:
    cfg = config_cls()
    if path is None:
        return cfg
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    valid = {field.name for field in fields(cfg)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(f'Unknown config fields: {unknown}')
    for key, value in data.items():
        setattr(cfg, key, value)
    return cfg


def parse_config_path(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, default=None)
    return parser.parse_args()
