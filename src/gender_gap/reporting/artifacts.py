"""Machine-readable artifacts export.

Produces a single JSON file that consolidates all model results
for programmatic consumption by downstream tools or dashboards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _csv_to_records(path: Path) -> list[dict]:
    """Load a CSV and return as list of dicts."""
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def export_json_artifacts(input_dir: Path, output_dir: Path) -> Path:
    """Consolidate all model CSVs into a single JSON artifact.

    Parameters
    ----------
    input_dir : Path
        Directory containing model output CSVs.
    output_dir : Path
        Directory for the consolidated JSON.

    Returns
    -------
    Path
        Path to the generated JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "version": "0.1.0",
        "models": {},
    }

    # Raw gap
    raw = _csv_to_records(input_dir / "raw_gap.csv")
    if raw:
        artifact["models"]["raw_gap"] = raw[0]

    # OLS sequential
    ols = _csv_to_records(input_dir / "ols_sequential.csv")
    if ols:
        artifact["models"]["ols_sequential"] = ols

    # Oaxaca
    oaxaca = _csv_to_records(input_dir / "oaxaca.csv")
    if oaxaca:
        artifact["models"]["oaxaca_blinder"] = oaxaca

    # Elastic net interactions
    en = _csv_to_records(input_dir / "elastic_net_interactions.csv")
    if en:
        artifact["models"]["elastic_net"] = en

    # DML
    dml = _csv_to_records(input_dir / "dml.csv")
    if dml:
        artifact["models"]["double_ml"] = dml[0]

    # Quantile regression
    qr = _csv_to_records(input_dir / "quantile_regression.csv")
    if qr:
        artifact["models"]["quantile_regression"] = qr

    # Reproductive-extension outputs
    fertility = _csv_to_records(input_dir / "acs_fertility_risk_penalty.csv")
    if fertility:
        artifact["models"]["fertility_risk_penalty"] = fertility

    fertility_quartiles = _csv_to_records(input_dir / "acs_fertility_risk_by_quartile.csv")
    if fertility_quartiles:
        artifact["models"]["fertility_risk_quartiles"] = fertility_quartiles

    variance = _csv_to_records(input_dir / "acs_variance_suite.csv")
    if variance:
        artifact["models"]["variance_suite"] = variance

    interactions = _csv_to_records(input_dir / "acs_onet_interactions.csv")
    if interactions:
        artifact["models"]["onet_interactions"] = interactions

    # Heterogeneity (one entry per dimension)
    het = {}
    for het_path in sorted(input_dir.glob("heterogeneity_*.csv")):
        dim = het_path.stem.replace("heterogeneity_", "")
        records = _csv_to_records(het_path)
        if records:
            het[dim] = records
    if het:
        artifact["models"]["heterogeneity"] = het

    # Manifest
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        artifact["manifest"] = json.loads(manifest_path.read_text())

    # Write
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    logger.info("Exported JSON artifact to %s", out_path)
    return out_path
