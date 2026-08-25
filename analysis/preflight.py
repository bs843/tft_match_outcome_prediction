from __future__ import annotations

import argparse
import sys

from src.config import load_config

from .src.contract import AnalysisContractError, run_analysis_preflight
from .src.gate import AnalysisGateError
from .src.splits import FrozenSplitError


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen thesis analysis contract")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    try:
        result = run_analysis_preflight(load_config(args.config))
    except (AnalysisContractError, AnalysisGateError, FrozenSplitError) as exc:
        print(f"ANALYSIS PREFLIGHT: NO-GO\n{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print("ANALYSIS PREFLIGHT: GO")
    print(f"contract={result.contract['analysis_contract_version']}")
    print(f"dataset={result.approval.dataset_version}")
    print("split=train:3500 validation:750 test:750")
    print(f"split_sha256={result.split_hash}")
    print(f"analysis_tag={result.analysis_tag}")
    print(f"analysis_commit={result.analysis_commit}")
    print("checks=" + ",".join(result.checks))


if __name__ == "__main__":
    main()
