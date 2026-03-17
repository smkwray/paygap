"""CLI entry point for gender-gap analysis toolkit.

Subcommands
-----------
registry     Show dataset registry
download     Download raw datasets
standardize  Standardize raw data into canonical schemas
features     Build feature tables
model        Run estimation models
report       Generate reports and artifacts
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_registry_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("registry", help="Show dataset registry")


def _add_download_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("download", help="Download raw datasets")
    p.add_argument("--dataset", required=True,
                   choices=["acs", "cps", "sipp", "atus", "context", "sce_labor_market"],
                   help="Dataset to download")
    p.add_argument("--years", nargs="+", type=int, help="Years to download")
    p.add_argument("--output-dir", type=Path, help="Override output directory")
    p.add_argument("--variant", default="default",
                   help="Downloader variant (for ACS: default, api, official, ipums)")
    p.add_argument("--include-replicate-weights", action="store_true",
                   help="For ACS API downloads, include PWGTP1-PWGTP80")
    p.add_argument("--force", action="store_true",
                   help="Redownload even if destination files already exist")


def _add_standardize_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("standardize", help="Standardize raw data")
    p.add_argument("--dataset", required=True,
                   choices=["acs", "cps", "sipp", "atus", "nlsy", "psid", "context"],
                   help="Dataset to standardize")
    p.add_argument("--input", type=Path, help="Input file/directory")
    p.add_argument("--output", type=Path, help="Output parquet path")
    p.add_argument("--variant", default="ipums",
                   help="Data variant (e.g., ipums, official)")


def _add_features_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("features", help="Build feature tables")
    p.add_argument("--input", required=True, type=Path,
                   help="Standardized parquet file")
    p.add_argument("--output", required=True, type=Path,
                   help="Output parquet with features")
    p.add_argument("--sample", default="prime_age",
                   choices=["prime_age", "all_employed"],
                   help="Sample filter to apply")


def _add_model_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("model", help="Run estimation models")
    p.add_argument("--input", required=True, type=Path,
                   help="Analysis-ready parquet file")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory for model outputs")
    p.add_argument("--models", nargs="+",
                   default=["descriptive", "ols", "oaxaca"],
                   choices=["descriptive", "ols", "oaxaca", "elastic_net",
                            "boosting", "dml", "quantile", "heterogeneity",
                            "fertility_risk", "variance_suite"],
                   help="Models to run")
    p.add_argument("--weight-col", default="person_weight",
                   help="Survey weight column")
    p.add_argument("--nlsy", action="store_true",
                   help="Use NLSY block definitions (includes g_proxy)")


def _add_report_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("report", help="Generate reports and artifacts")
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Directory with model outputs")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory for report outputs")
    p.add_argument("--formats", nargs="+", default=["markdown", "csv", "json", "plots"],
                   choices=["markdown", "csv", "json", "plots"],
                   help="Output formats")


def _add_repro_parser(sub: argparse._SubParsersAction) -> None:
    sub.add_parser(
        "repro",
        help="Run the reproductive-burden extension using available shared/local inputs",
    )


def _add_variance_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "variance",
        help="Run the dedicated variance addon using shared/local public-core inputs",
    )
    p.add_argument(
        "--config",
        type=Path,
        help="Optional path to a variance addon config file",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gender-gap",
        description="U.S. gender earnings gap analysis toolkit",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    _add_registry_parser(sub)
    _add_download_parser(sub)
    _add_standardize_parser(sub)
    _add_features_parser(sub)
    _add_model_parser(sub)
    _add_report_parser(sub)
    _add_repro_parser(sub)
    _add_variance_parser(sub)

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.command is None:
        parser.print_help()
        return

    handler = {
        "registry": _cmd_registry,
        "download": _cmd_download,
        "standardize": _cmd_standardize,
        "features": _cmd_features,
        "model": _cmd_model,
        "report": _cmd_report,
        "repro": _cmd_repro,
        "variance": _cmd_variance,
    }.get(args.command)

    if handler:
        handler(args)
    else:
        parser.print_help()


# ── Subcommand implementations ──────────────────────────────────────────


def _cmd_registry(args: argparse.Namespace) -> None:
    from gender_gap.registry import print_registry
    print_registry()


def _cmd_download(args: argparse.Namespace) -> None:
    from gender_gap.settings import DATA_RAW

    output = args.output_dir or DATA_RAW
    dataset = args.dataset

    if dataset == "acs":
        from gender_gap.downloaders.acs import ACSDownloader
        dl = ACSDownloader(raw_dir=output)
    elif dataset == "cps":
        from gender_gap.downloaders.cps import CPSDownloader
        dl = CPSDownloader(raw_dir=output)
    elif dataset == "sipp":
        from gender_gap.downloaders.sipp import SIPPDownloader
        dl = SIPPDownloader(raw_dir=output)
    elif dataset == "atus":
        from gender_gap.downloaders.atus import ATUSDownloader
        dl = ATUSDownloader(raw_dir=output)
    elif dataset == "context":
        dl = _build_context_downloader(output, args.variant)
    elif dataset == "sce_labor_market":
        from gender_gap.downloaders.sce import SCELaborMarketDownloader
        dl = SCELaborMarketDownloader(raw_dir=output)
    else:
        print(f"Unknown dataset: {dataset}")
        return

    years = args.years
    if years:
        if dataset == "acs":
            _download_acs_with_options(dl, years, args)
        else:
            dl.download(years=years)
    else:
        if dataset == "acs":
            _download_acs_with_options(dl, None, args)
        else:
            dl.download()
    print(f"Download complete: {dataset} → {output}")


def _download_acs_with_options(dl, years, args: argparse.Namespace) -> None:
    mode = "official"
    use_ipums = False
    if args.variant == "api":
        mode = "api"
    elif args.variant == "official":
        mode = "official"
    elif args.variant == "ipums":
        use_ipums = True
    elif args.variant not in ("default", "", None):
        raise ValueError(f"Unknown ACS download variant: {args.variant}")

    dl.download(
        years=years,
        use_ipums=use_ipums,
        mode=mode,
        include_replicate_weights=args.include_replicate_weights,
        force=args.force,
    )


def _build_context_downloader(output: Path, variant: str):
    from gender_gap.downloaders.context import (
        BEARPPDownloader,
        CPIDownloader,
        LAUSDownloader,
        OEWSDownloader,
        ONETDownloader,
        QCEWDownloader,
        QWIDownloader,
    )

    variant = variant or "default"
    if variant in ("default", "laus"):
        return LAUSDownloader(raw_dir=output)
    if variant == "cpi":
        return CPIDownloader(raw_dir=output)
    if variant == "qcew":
        return QCEWDownloader(raw_dir=output)
    if variant == "oews":
        return OEWSDownloader(raw_dir=output)
    if variant == "qwi":
        return QWIDownloader(raw_dir=output)
    if variant == "onet":
        return ONETDownloader(raw_dir=output)
    if variant in ("bea", "bea_rpp"):
        return BEARPPDownloader(raw_dir=output)
    raise ValueError(f"Unknown context download variant: {variant}")


def _read_input(path: Path):
    """Read CSV or Parquet input file."""
    import pandas as pd
    path_str = str(path).lower()
    if path_str.endswith(".csv"):
        return pd.read_csv(path)
    if path_str.endswith(".dta"):
        return pd.read_stata(path)
    if path_str.endswith(".sas7bdat"):
        return pd.read_sas(path)
    return pd.read_parquet(path)


def _cmd_standardize(args: argparse.Namespace) -> None:
    dataset = args.dataset
    variant = args.variant

    if dataset == "acs":
        from gender_gap.standardize.acs_standardize import standardize_acs
        if not args.input:
            print("--input required for ACS standardization")
            return
        df = _read_input(args.input)
        result = standardize_acs(df)

    elif dataset == "cps":
        from gender_gap.standardize.cps_standardize import (
            standardize_cps_ipums,
            standardize_cps_official,
        )
        if not args.input:
            print("--input required for CPS standardization")
            return
        df = _read_input(args.input)
        if variant == "official":
            result = standardize_cps_official(df)
        else:
            result = standardize_cps_ipums(df)

    elif dataset == "sipp":
        from gender_gap.standardize.sipp_standardize import standardize_sipp
        if not args.input:
            print("--input required for SIPP standardization")
            return
        df = _read_input(args.input)
        inferred_year = _infer_year_from_path(args.input)
        result = standardize_sipp(df, survey_year=inferred_year)

    elif dataset == "atus":
        from gender_gap.standardize.atus_standardize import (
            standardize_atus_ipums,
            standardize_atus_summary,
        )
        if not args.input:
            print("--input required for ATUS standardization")
            return
        df = _read_input(args.input)
        if variant == "ipums":
            result = standardize_atus_ipums(df)
        else:
            result = standardize_atus_summary(df)

    elif dataset == "nlsy":
        from gender_gap.settings import load_config
        from gender_gap.standardize.nlsy_standardize import (
            standardize_nlsy79_for_gap,
            standardize_nlsy97_for_gap,
        )
        cfg = load_config()
        nlsy_cfg = cfg.get("nlsy", {})
        source_dir = args.input or Path(nlsy_cfg.get("source_dir", "."))
        if variant == "79":
            result = standardize_nlsy79_for_gap(source_dir)
        else:
            result = standardize_nlsy97_for_gap(source_dir)

    elif dataset == "psid":
        from gender_gap.standardize.psid_standardize import standardize_psid_2023_for_gap

        result = standardize_psid_2023_for_gap(raw_dir=args.input)

    elif dataset == "context":
        from gender_gap.standardize.context_standardize import (
            standardize_bea_rpp,
            standardize_laus,
            standardize_oews,
            standardize_qcew,
        )
        if not args.input:
            print("--input required for context standardization")
            return
        df = _read_input(args.input)
        if variant == "qcew":
            result = standardize_qcew(df)
        elif variant == "oews":
            result = standardize_oews(df)
        elif variant in ("bea", "bea_rpp"):
            result = standardize_bea_rpp(df)
        else:
            result = standardize_laus(df)
    else:
        print(f"Unknown dataset: {dataset}")
        return

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
        print(f"Standardized {dataset} → {args.output} ({len(result):,} rows)")
    else:
        print(f"Standardized {dataset}: {len(result):,} rows (use --output to save)")


def _infer_year_from_path(path: Path) -> int | None:
    sipp_public_use = re.fullmatch(r"pu((19|20)\d{2})(?:_.*)?", path.stem.lower())
    if sipp_public_use:
        return int(sipp_public_use.group(1)) - 1
    match = re.search(r"(19|20)\d{2}", path.stem)
    if match:
        return int(match.group(0))
    return None


def _cmd_features(args: argparse.Namespace) -> None:
    import pandas as pd

    from gender_gap.features import (
        add_fertility_risk_features,
        add_repro_interactions,
        add_reproductive_features,
    )
    from gender_gap.crosswalks.industry_crosswalks import (
        census_ind_to_naics2,
        naics2_to_broad,
    )
    from gender_gap.crosswalks.occupation_crosswalks import (
        census_occ_to_soc_major,
        soc_major_to_broad,
    )
    from gender_gap.features.commute import commute_bin
    from gender_gap.features.earnings import compute_hourly_wage, log_wage, winsorize_wages
    from gender_gap.features.family import parenthood_category
    from gender_gap.features.sample_filters import (
        filter_all_employed,
        filter_prime_age_wage_salary,
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    # Apply sample filter
    if args.sample == "prime_age":
        df = filter_prime_age_wage_salary(df)
    else:
        df = filter_all_employed(df)
    logger.info("After sample filter: %d rows", len(df))

    # Earnings features
    if "hourly_wage" not in df.columns and "earnings_annual" in df.columns:
        df["hourly_wage"] = compute_hourly_wage(
            df["earnings_annual"], df.get("hours_usual"), df.get("weeks_worked"),
            method="annual",
        )
    if "hourly_wage_real" in df.columns:
        df["hourly_wage_real"] = winsorize_wages(df["hourly_wage_real"])
        df["log_hourly_wage_real"] = log_wage(df["hourly_wage_real"])

    # Family features
    if "n_own_children" in df.columns:
        df["parenthood_category"] = parenthood_category(
            df["n_own_children"], df.get("age_youngest_child"),
        )

    # Commute features
    if "commute_minutes" in df.columns:
        df["commute_bin"] = commute_bin(df["commute_minutes"])

    # Occupation crosswalks
    if "occupation_code" in df.columns:
        soc2 = census_occ_to_soc_major(df["occupation_code"])
        df["occupation_broad"] = soc_major_to_broad(soc2)

    # Industry crosswalks
    if "industry_code" in df.columns:
        naics2 = census_ind_to_naics2(df["industry_code"])
        df["industry_broad"] = naics2_to_broad(naics2)

    # Age squared
    if "age" in df.columns and "age_sq" not in df.columns:
        df["age_sq"] = df["age"] ** 2

    df = add_reproductive_features(df)
    df = add_fertility_risk_features(df)
    df = add_repro_interactions(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Features built: {len(df):,} rows → {args.output}")


def _cmd_model(args: argparse.Namespace) -> None:
    import json

    import pandas as pd

    df = pd.read_parquet(args.input)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_col = args.weight_col

    all_results = {}

    if "descriptive" in args.models:
        from gender_gap.models.descriptive import raw_gap
        raw = raw_gap(df, weight=weight_col)
        all_results["raw_gap"] = raw
        pd.DataFrame([raw]).to_csv(output_dir / "raw_gap.csv", index=False)
        logger.info("Raw gap: %.1f%%", raw["gap_pct"])

    if "ols" in args.models:
        from gender_gap.models.ols import results_to_dataframe, run_sequential_ols
        if args.nlsy:
            from gender_gap.models.ols import NLSY_BLOCK_DEFINITIONS
            ols = run_sequential_ols(df, blocks=NLSY_BLOCK_DEFINITIONS,
                                     weight_col=weight_col)
        else:
            ols = run_sequential_ols(df, weight_col=weight_col)
        ols_df = results_to_dataframe(ols)
        ols_df.to_csv(output_dir / "ols_sequential.csv", index=False)
        all_results["ols"] = ols
        logger.info("OLS: %d models estimated", len(ols))

    if "oaxaca" in args.models:
        from gender_gap.models.oaxaca import oaxaca_blinder, oaxaca_summary_table
        ob = oaxaca_blinder(df, weight_col=weight_col)
        oaxaca_summary_table(ob).to_csv(output_dir / "oaxaca.csv", index=False)
        all_results["oaxaca"] = ob
        logger.info("Oaxaca: explained=%.1f%%", ob.explained_pct)

    if "elastic_net" in args.models:
        from gender_gap.models.elastic_net import run_elastic_net
        en = run_elastic_net(df, weight_col=weight_col)
        en["top_interactions"].to_csv(output_dir / "elastic_net_interactions.csv", index=False)
        all_results["elastic_net"] = en

    if "boosting" in args.models:
        from gender_gap.models.boosting import run_catboost
        cb = run_catboost(df, weight_col=weight_col)
        all_results["boosting"] = cb

    if "dml" in args.models:
        from gender_gap.models.dml import run_dml
        dml = run_dml(df, weight_col=weight_col)
        pd.DataFrame([{
            "treatment_effect": dml["treatment_effect"],
            "std_error": dml["std_error"],
            "ci_lower": dml["ci_lower"],
            "ci_upper": dml["ci_upper"],
            "pvalue": dml["pvalue"],
        }]).to_csv(output_dir / "dml.csv", index=False)
        all_results["dml"] = dml

    if "quantile" in args.models:
        from gender_gap.models.quantile import (
            diagnose_distributional_pattern,
            quantile_results_to_dataframe,
            run_quantile_regression,
        )
        qr = run_quantile_regression(df, weight_col=weight_col)
        quantile_results_to_dataframe(qr).to_csv(
            output_dir / "quantile_regression.csv", index=False
        )
        pattern = diagnose_distributional_pattern(qr)
        all_results["quantile"] = {"results": qr, "pattern": pattern}
        logger.info("Quantile pattern: %s", pattern)

    if "heterogeneity" in args.models:
        from gender_gap.models.heterogeneity import run_full_heterogeneity
        het = run_full_heterogeneity(df, weight_col=weight_col)
        for dim, hr in het.items():
            hr.subgroup_gaps.to_csv(
                output_dir / f"heterogeneity_{dim}.csv", index=False
            )
        all_results["heterogeneity"] = het

    if "fertility_risk" in args.models:
        from gender_gap.models.fertility_risk import run_fertility_risk_penalty

        penalty, quartiles = run_fertility_risk_penalty(df, weight_col=weight_col)
        penalty.to_csv(output_dir / "acs_fertility_risk_penalty.csv", index=False)
        quartiles.to_csv(output_dir / "acs_fertility_risk_by_quartile.csv", index=False)
        all_results["fertility_risk"] = {"penalty": penalty, "quartiles": quartiles}

    if "variance_suite" in args.models:
        from gender_gap.models.variance_suite import run_variance_suite

        variance = run_variance_suite(df, weight_col=weight_col)
        variance.to_csv(output_dir / "acs_variance_suite.csv", index=False)
        all_results["variance_suite"] = variance

    # Save a manifest of what was run
    manifest = {
        "models_run": args.models,
        "n_obs": len(df),
        "input": str(args.input),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Models complete: {', '.join(args.models)} → {output_dir}")


def _cmd_report(args: argparse.Namespace) -> None:
    import pandas as pd

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load available model outputs
    raw_gap_path = input_dir / "raw_gap.csv"
    ols_path = input_dir / "ols_sequential.csv"
    input_dir / "oaxaca.csv"
    input_dir / "quantile_regression.csv"

    if "csv" in args.formats:
        # Copy CSVs to report dir
        for src in input_dir.glob("*.csv"):
            dst = output_dir / src.name
            dst.write_bytes(src.read_bytes())
        logger.info("Copied CSV outputs to %s", output_dir)

    if "json" in args.formats:
        from gender_gap.reporting.artifacts import export_json_artifacts
        export_json_artifacts(input_dir, output_dir)

    if "markdown" in args.formats:
        from gender_gap.models.ols import OLSResult
        from gender_gap.reporting.tables import export_markdown_summary

        raw = None
        if raw_gap_path.exists():
            raw_df = pd.read_csv(raw_gap_path)
            raw = raw_df.iloc[0].to_dict()

        ols_results = []
        if ols_path.exists():
            ols_df = pd.read_csv(ols_path)
            for _, row in ols_df.iterrows():
                ols_results.append(OLSResult(
                    model_name=row.get("model_name", row.get("model")),
                    female_coef=row["female_coef"],
                    female_se=row["female_se"],
                    female_pvalue=row["female_pvalue"],
                    r_squared=row["r_squared"],
                    n_obs=int(row["n_obs"]),
                    controls=row.get("controls", ""),
                ))

        if raw is not None:
            export_markdown_summary(raw, ols_results, output_dir)
            logger.info("Generated markdown report")

    if "plots" in args.formats:
        from gender_gap.reporting.charts import generate_all_plots
        generate_all_plots(input_dir, output_dir)
        logger.info("Generated plots")

    print(f"Reports generated → {output_dir}")


def _cmd_repro(args: argparse.Namespace) -> None:
    from gender_gap.repro import run_repro_extension

    outputs = run_repro_extension()
    print("Reproductive-burden extension complete:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


def _cmd_variance(args: argparse.Namespace) -> None:
    from gender_gap.variance import run_variance_addon

    outputs = run_variance_addon(config_path=args.config)
    print("Variance addon complete:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
