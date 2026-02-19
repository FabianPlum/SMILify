"""
Minimal test to verify the new config system loads JSON and produces legacy dicts.

Run from project root:
  python -m smal_fitter.neuralSMIL.configs.test_config_load

Or from smal_fitter/neuralSMIL:
  python configs/test_config_load.py
"""
from __future__ import print_function

import os
import sys

# Ensure neuralSMIL is on path when run as script
if __name__ == "__main__":
    _neural_smil = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _neural_smil not in sys.path:
        sys.path.insert(0, _neural_smil)

from configs import (
    load_config,
    load_from_json,
    validate_json_mode,
    SingleViewConfig,
    MultiViewConfig,
    ConfigurationError,
)

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


def test_singleview():
    path = os.path.join(EXAMPLES_DIR, "singleview_baseline.json")
    if not os.path.isfile(path):
        print("SKIP singleview: file not found", path)
        return False
    validate_json_mode(path, "singleview")
    config = load_config(config_file=path)
    assert isinstance(config, SingleViewConfig)
    config.validate()
    legacy = config.to_legacy_dict()
    assert "smal_file" in legacy
    assert "shape_family" in legacy
    assert "training_params" in legacy
    assert "model_config" in legacy
    assert legacy["model_config"]["backbone_name"] == "vit_large_patch16_224"
    print("OK singleview: load + validate + to_legacy_dict")
    return True


def test_multiview():
    path = os.path.join(EXAMPLES_DIR, "multiview_6cam.json")
    if not os.path.isfile(path):
        print("SKIP multiview: file not found", path)
        return False
    validate_json_mode(path, "multiview")
    config = load_config(config_file=path)
    assert isinstance(config, MultiViewConfig)
    config.validate()
    legacy = config.to_multiview_legacy_dict()
    assert "smal_file" in legacy
    assert "shape_family" in legacy
    assert "batch_size" in legacy
    assert "cross_attention_layers" in legacy
    assert legacy["cross_attention_layers"] == 2
    print("OK multiview: load + validate + to_multiview_legacy_dict")
    return True


def test_mode_required():
    import tempfile
    import json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"dataset": {"data_path": "x.h5"}}, f)
        path = f.name
    try:
        try:
            load_from_json(path)
            print("FAIL mode_required: expected ConfigurationError")
            return False
        except ConfigurationError as e:
            if "mode" in str(e).lower():
                print("OK mode_required: ConfigurationError as expected")
                return True
            raise
    finally:
        os.unlink(path)


if __name__ == "__main__":
    ok = 0
    ok += test_singleview()
    ok += test_multiview()
    ok += test_mode_required()
    print("Total:", ok, "/ 3 passed")
    sys.exit(0 if ok == 3 else 1)
