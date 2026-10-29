"""Test verify_results execution and sensitivity to headline number drifts."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_results import verify_results

ARTIFACTS = ROOT / "artifacts"
FLOWS = ROOT / "data" / "processed" / "amfi_flows.csv"


def test_verify_results_passes_on_baseline() -> None:
    """Verify that verify_results passes on baseline artifacts with expected numbers."""
    result = verify_results(root=ARTIFACTS, flows_path=FLOWS, write_output=False)
    assert result["status"] == "PASS"
    assert np.isclose(result["independent_mae_crore"], 27310.0156, atol=1e-3)
    assert np.isclose(result["independent_coverage95_pct"], 85.4167, atol=1e-3)
    assert np.isclose(result["independent_policy_cost_crore"], 111863.359, atol=1e-2)


def test_verify_results_fails_if_mae_drifts(tmp_path: Path) -> None:
    """Verify that a drift in recorded or computed MAE raises an AssertionError."""
    for filename in ["evidence.json", "predictions.csv", "forecasts.csv", "policy_months.csv"]:
        shutil.copy(ARTIFACTS / filename, tmp_path / filename)

    evidence_file = tmp_path / "evidence.json"
    evidence = json.loads(evidence_file.read_text())
    evidence["selected_test"]["mae_crore"] += 500.0
    evidence_file.write_text(json.dumps(evidence, indent=2))

    with pytest.raises(AssertionError):
        verify_results(root=tmp_path, flows_path=FLOWS, write_output=False)


def test_verify_results_fails_if_coverage_drifts(tmp_path: Path) -> None:
    """Verify that a drift in recorded or computed coverage raises an AssertionError."""
    for filename in ["evidence.json", "predictions.csv", "forecasts.csv", "policy_months.csv"]:
        shutil.copy(ARTIFACTS / filename, tmp_path / filename)

    evidence_file = tmp_path / "evidence.json"
    evidence = json.loads(evidence_file.read_text())
    evidence["selected_test"]["coverage95_pct"] -= 5.0
    evidence_file.write_text(json.dumps(evidence, indent=2))

    with pytest.raises(AssertionError):
        verify_results(root=tmp_path, flows_path=FLOWS, write_output=False)


def test_verify_results_fails_if_policy_cost_drifts(tmp_path: Path) -> None:
    """Verify that a drift in policy cost raises an AssertionError."""
    for filename in ["evidence.json", "predictions.csv", "forecasts.csv", "policy_months.csv"]:
        shutil.copy(ARTIFACTS / filename, tmp_path / filename)

    evidence_file = tmp_path / "evidence.json"
    evidence = json.loads(evidence_file.read_text())
    for item in evidence["policy"]:
        if item["policy"] == "fixed_q95":
            item["cost_crore"] += 1000.0
    evidence_file.write_text(json.dumps(evidence, indent=2))

    with pytest.raises(AssertionError):
        verify_results(root=tmp_path, flows_path=FLOWS, write_output=False)
