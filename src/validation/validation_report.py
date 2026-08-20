from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Issue:
    severity: str
    scope: str
    reason_code: str
    reason: str
    match_hash: str = ""
    participant_hash: str = ""
    field: str = ""


@dataclass
class ValidationSummary:
    collection_status: str = "FAIL"
    total_matches: int = 0
    valid_matches: int = 0
    invalid_matches: int = 0
    total_participants: int = 0
    valid_participants: int = 0
    invalid_participants: int = 0
    patch_17_8_percentage: float = 0.0
    duplicate_match_count: int = 0
    invalid_placement_count: int = 0
    missing_required_field_count: int = 0
    end_result_blank_count: int = 0
    end_result_mismatch_count: int = 0
    rank_snapshot_match_rate: float = 0.0
    schema_errors: int = 0
    warnings: int = 0
    critical_failures: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def final_status(issues: Iterable[Issue]) -> str:
    issues = list(issues)
    if any(issue.severity == "critical" for issue in issues):
        return "FAIL"
    if any(issue.severity == "warning" for issue in issues):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_markdown(path: Path, summary: ValidationSummary, issues: list[Issue]) -> None:
    critical = [issue for issue in issues if issue.severity == "critical"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    lines = [
        "# Post-collection validation report", "",
        f"**STATUS: {summary.collection_status}**", "",
        "## Core results", "",
        f"- Matches checked: {summary.total_matches}",
        f"- Valid matches: {summary.valid_matches}",
        f"- Participants checked: {summary.total_participants}",
        f"- Valid participants: {summary.valid_participants}",
        f"- Patch 17.8: {summary.patch_17_8_percentage:.2f}%",
        f"- Rank snapshot match rate: {summary.rank_snapshot_match_rate:.2f}%",
        f"- Critical failures: {len(critical)}",
        f"- Warnings: {len(warnings)}", "",
        "## Reproducibility", "",
    ]
    for key, value in summary.metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Issue counts", ""])
    grouped: dict[tuple[str, str], int] = {}
    for issue in issues:
        key = (issue.severity, issue.reason_code)
        grouped[key] = grouped.get(key, 0) + 1
    if grouped:
        lines.extend(["| Severity | Reason code | Count |", "|---|---|---:|"])
        for (severity, code), count in sorted(grouped.items()):
            lines.append(f"| {severity} | `{code}` | {count} |")
    else:
        lines.append("No issues detected.")
    lines.extend([
        "", "## Interpretation", "",
        "Exception files use project-specific HMAC identifiers. Raw Riot identifiers are not included.",
        "A validated dataset is emitted only when no critical failures are present.", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value
