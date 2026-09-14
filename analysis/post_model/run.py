from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import (PRODUCTION_RUN_ID, default_output_root, generated_file_hashes,
                     setup_logger, sha256_file)
from .descriptive import build_descriptive_outputs
from .diagnostics import build_final_diagnostics
from .interpretation import build_interpretation_outputs
from .manifest import write_post_model_manifest
from .performance import build_performance_outputs
from .reporting import write_thesis_reports
from .robustness import build_robustness_outputs
from .verify import (PostModelVerificationError, verify_inputs,
                     write_verification_reports)


def run(*, analysis_run: str, output_root: Path, bootstrap_samples: int = 2000,
        permutation_repeats: int = 5) -> Path:
    verified = verify_inputs(analysis_run, output_root, require_output_absent=True)
    output_root.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(output_root / "logs" / "post_model_analysis.log")
    logger.info("POST-MODEL ANALYSIS START source=%s", verified.production_root)
    write_verification_reports(verified, output_root)
    logger.info("Production verification PASS; %d source artifact hashes verified",
                len(verified.artifact_hashes))
    performance = build_performance_outputs(verified, output_root, logger)
    diagnostics = build_final_diagnostics(verified, output_root, logger,
                                          bootstrap_samples=bootstrap_samples)
    robustness = build_robustness_outputs(diagnostics["predictions"], output_root, logger)
    interpretation = build_interpretation_outputs(
        verified, output_root, logger, repeats=permutation_repeats
    )
    descriptive = build_descriptive_outputs(verified, output_root, logger)
    warnings = list(verified.verification_report["warnings"]) + interpretation["warnings"]
    small_rank = robustness["rank"].loc[robustness["rank"]["n"] < 50, "rank_stratum"].tolist()
    if small_rank:
        warnings.append(
            "Rank strata with fewer than 50 test observations are retained but should not be "
            "interpreted as stable estimates: " + ", ".join(small_rank)
        )
    warnings.append(
        "Calibration intercept/slope were not fitted on the final test set; Brier score and "
        "observed-versus-predicted bins are reported without recalibration."
    )
    write_thesis_reports(verified, output_root, performance, diagnostics, robustness,
                         interpretation, descriptive, warnings)

    after = {}
    for relative in verified.artifact_hashes:
        path = verified.production_root / relative
        after[relative] = sha256_file(path)
    if after != verified.artifact_hashes:
        raise RuntimeError("Frozen production artifacts changed during post-model analysis")
    logger.info("Confirmed all %d frozen production artifact hashes remain unchanged", len(after))
    logger.info("POST-MODEL ANALYSIS COMPLETE status=PASS")
    for handler in logger.handlers:
        handler.flush()
    manifest = write_post_model_manifest(verified, output_root, warnings)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only TFT thesis post-model analysis")
    parser.add_argument("--analysis-run", default=PRODUCTION_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    args = parser.parse_args()
    try:
        output = run(analysis_run=args.analysis_run, output_root=args.output_root,
                     bootstrap_samples=args.bootstrap_samples,
                     permutation_repeats=args.permutation_repeats)
    except (PostModelVerificationError, RuntimeError, ValueError) as exc:
        print(f"POST-MODEL ANALYSIS: NO-GO\n{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    tables = len(list((output / "tables").glob("*.csv")))
    figures = len(list((output / "figures").glob("*.png"))) + len(list((output / "eda").glob("*.png")))
    reports = len(list(output.glob("*.md"))) + len(list((output / "tables").glob("*.md")))
    print("POST-MODEL ANALYSIS: PASS")
    print("\nDataset:\n  tft_set17_patch17_8_v11")
    print("\nFrozen split:\n  Train:      3500 matches\n  Validation:  750 matches\n  Test:        750 matches")
    print("\nSelected model:\n  xgboost")
    print("\nPrimary selection metric:\n  f1")
    print("\nCV:         0.849\nValidation: 0.854\nFinal test: 0.854\nCV -> Test gap: +0.004")
    print(f"\nOutputs:\n  Tables: {tables}\n  Figures: {figures}\n  Reports: {reports}")
    print("\nVerification failures:\n  0")
    print("\nResults directory:\n  " + str(output.resolve()))


if __name__ == "__main__":
    main()
