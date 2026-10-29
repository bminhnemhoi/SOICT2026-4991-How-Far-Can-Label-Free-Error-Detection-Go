"""Tests for src/utils/seeds.py + io.py"""

from __future__ import annotations

import random
from pathlib import Path

from src.utils.io import md5_of_str, read_json, read_yaml, write_json, write_yaml
from src.utils.seeds import set_global_seed


def test_seed_determinism():
    set_global_seed(42)
    a = [random.random() for _ in range(5)]
    set_global_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_yaml_roundtrip(tmp_path: Path):
    obj = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}
    p = tmp_path / "x.yaml"
    write_yaml(obj, p)
    assert read_yaml(p) == obj


def test_json_roundtrip(tmp_path: Path):
    obj = {"a": 1, "b": [1, 2]}
    p = tmp_path / "x.json"
    write_json(obj, p)
    assert read_json(p) == obj


def test_md5_stable():
    assert md5_of_str("hello") == md5_of_str("hello")
    assert md5_of_str("hello") != md5_of_str("world")
