"""
To (re-)generate the reference files after a verified-good run:
    python tests/generate_flow_references.py

"""

import os
import pytest
import torch
import zuko

from higgs_dna.tools.flow_corrections import apply_flow

print(f"zuko v{zuko.__version__}")
FLOW_CONFIGS = {
    "preEE": dict(features=16, context=5, bins=10, transforms=5, hidden_features=[256, 256]),
    "postEE": dict(features=16, context=5, bins=10, transforms=5, hidden_features=[256, 256]),
    "2023_model": dict(features=16, context=5, bins=10, transforms=5, hidden_features=[256, 256], passes=2),
    "2024_model": dict(features=16, context=5, bins=10, transforms=5, hidden_features=[256, 256], passes=2),
}
NUM_VARS = 16
NUM_CONDITIONS = 5
BATCH_SIZE = 4
SEED = 42
DTYPE = torch.float64
ATOL = 1e-6

FLOWS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "higgs_dna", "tools", "flows")
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "flow_references")


def _make_inputs(seed: int = SEED):
    """Return deterministic (input, conditions) tensors."""
    gen = torch.Generator().manual_seed(seed)
    inputs = torch.randn(BATCH_SIZE, NUM_VARS, generator=gen, dtype=DTYPE)
    # Last column of conditions is the isData boolean (0 for MC)
    conditions = torch.randn(BATCH_SIZE, NUM_CONDITIONS, generator=gen, dtype=DTYPE)
    conditions[:, -1] = 0.0
    return inputs, conditions


def _load_flow(model_dir: str, cfg: dict):
    """Construct an NSF flow and load the saved state dict."""
    flow = zuko.flows.NSF(**cfg)
    state_path = os.path.join(model_dir, "best_model_.pth")
    flow.load_state_dict(
        torch.load(state_path, map_location=torch.device("cpu"), weights_only=False)
    )
    return flow


def _available_models():
    """Yield (model_name, model_dir, config) for every model that exists on disk."""
    for name, cfg in FLOW_CONFIGS.items():
        model_dir = os.path.join(FLOWS_DIR, name)
        if os.path.isfile(os.path.join(model_dir, "best_model_.pth")):
            yield name, model_dir, cfg


def _reference_path(model_name: str) -> str:
    return os.path.join(REFERENCE_DIR, f"{model_name}_ref.pt")


@pytest.fixture(params=list(_available_models()), ids=lambda p: p[0])
def model_fixture(request):
    name, model_dir, cfg = request.param
    flow = _load_flow(model_dir, cfg)
    return name, flow, _reference_path(name)


def test_flow_output_matches_reference(model_fixture):
    """Run apply_flow and compare output to stored reference tensor."""
    name, flow, ref_path = model_fixture

    if not os.path.isfile(ref_path):
        pytest.skip(
            f"Reference file missing for {name}. "
            "Run 'python tests/generate_flow_references.py' first."
        )

    inputs, conditions = _make_inputs()
    result = apply_flow(inputs, conditions, flow)
    reference = torch.load(ref_path, map_location="cpu", weights_only=False)

    torch.testing.assert_close(
        result,
        reference,
        atol=ATOL,
        rtol=0,
        msg=f"Flow output for '{name}' differs from reference.",
    )
