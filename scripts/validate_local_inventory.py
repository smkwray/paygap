#!/usr/bin/env python3
"""Validate the local repro inventory layout and write pack-style diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from gender_gap.reporting.repro import build_local_inventory_report, write_local_inventory_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("inventory/inventory_paths.yaml"),
        help="Inventory YAML with sibling repo and data paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/local_inventory_report.json"),
        help="JSON output path; a markdown companion is written beside it.",
    )
    args = parser.parse_args()

    report = build_local_inventory_report(args.inventory)
    path = write_local_inventory_report(report, args.output)
    print(path)


if __name__ == "__main__":
    main()
