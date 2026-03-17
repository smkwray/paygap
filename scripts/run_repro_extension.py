#!/usr/bin/env python3
"""Run the reproductive-burden extension against available project data."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gender_gap.repro import run_repro_extension


if __name__ == "__main__":
    outputs = run_repro_extension()
    for name, path in outputs.items():
        print(f"{name}: {path}")
