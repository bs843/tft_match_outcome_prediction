from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml


class LeakageError(RuntimeError):
    pass


def load_roles(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LeakageError("Column-role configuration must be a mapping")
    return payload


def validate_declared_columns(columns: Iterable[str], rules: dict) -> None:
    roles = rules.get("roles", {})
    undeclared = set(columns).difference(roles)
    if undeclared:
        raise LeakageError(f"Columns lack declared roles: {sorted(undeclared)}")
    invalid = {name: role for name, role in roles.items()
               if role not in {"TARGET", "FEATURE", "AUDIT_METADATA", "IDENTIFIER", "EXCLUDED"}}
    if invalid:
        raise LeakageError(f"Invalid column roles: {invalid}")


def assert_no_leakage(feature_columns: Iterable[str], rules: dict) -> None:
    exact = {str(value).lower() for value in rules.get("prohibited_exact", [])}
    prefixes = tuple(str(value).lower() for value in rules.get("prohibited_prefixes", []))
    violations = []
    for column in feature_columns:
        normalised = str(column).lower()
        if normalised in exact or normalised.startswith(prefixes):
            violations.append(str(column))
    if violations:
        raise LeakageError(f"Prohibited leakage columns in X: {sorted(violations)}")
