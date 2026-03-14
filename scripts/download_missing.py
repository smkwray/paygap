#!/usr/bin/env python3
"""Download only the missing datasets (ACS 2019, 2021, 2023 + context data)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from download_all import (
    download_acs_year,
    download_bea_rpp,
    download_cpi_u,
    download_laus,
    download_oews,
    download_qcew,
    logger,
)


def main():
    start = time.time()

    logger.info("=" * 60)
    logger.info("DOWNLOADING MISSING DATA")
    logger.info("=" * 60)

    # ACS 2019, 2021, 2023 (missing from previous run)
    for year in [2019, 2021, 2023]:
        try:
            df = download_acs_year(year)
            logger.info("ACS %d: %d rows", year, len(df))
        except Exception as e:
            logger.error("ACS %d failed: %s", year, e)

    # Context data
    logger.info("--- Context data ---")
    try:
        laus = download_laus()
        logger.info("LAUS: %d rows", len(laus))
    except Exception as e:
        logger.error("LAUS failed: %s", e)

    try:
        cpi = download_cpi_u()
        logger.info("CPI-U: %d rows", len(cpi))
    except Exception as e:
        logger.error("CPI-U failed: %s", e)

    try:
        bea = download_bea_rpp()
        logger.info("BEA RPP: %d rows", len(bea))
    except Exception as e:
        logger.error("BEA RPP failed: %s", e)

    try:
        oews = download_oews(list(range(2019, 2024)))
        logger.info("OEWS: %d years", len(oews))
    except Exception as e:
        logger.error("OEWS failed: %s", e)

    try:
        qcew = download_qcew(list(range(2020, 2024)))
        logger.info("QCEW: %d years", len(qcew))
    except Exception as e:
        logger.error("QCEW failed: %s", e)

    elapsed = time.time() - start
    logger.info("Done in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
