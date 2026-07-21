#!/usr/bin/env python3
"""
Generate reference tensors for test_flows.py.

Run this once with a known-good environment/zuko version, then commit the
resulting .pt files under tests/flow_references/.
"""

import os
import sys
import torch

from tests.test_flows import (
    REFERENCE_DIR,
    _available_models,
    _load_flow,
    _make_inputs,
    _reference_path,
)
from higgs_dna.tools.flow_corrections import apply_flow


def main():
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    inputs, conditions = _make_inputs()
    generated = []

    for name, model_dir, cfg in _available_models():
        print(f"Generating reference for {name} ...")
        flow = _load_flow(model_dir, cfg)
        result = apply_flow(inputs, conditions, flow)

        ref_path = _reference_path(name)
        torch.save(result, ref_path)
        print(f"  saved {ref_path}  (shape={tuple(result.shape)})")
        generated.append(name)

    if not generated:
        print("No models found — nothing to generate.", file=sys.stderr)
        return 1

    print(f"\nDone. Generated references for: {', '.join(generated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
