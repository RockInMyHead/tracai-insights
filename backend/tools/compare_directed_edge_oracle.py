#!/usr/bin/env python3
"""Compare production and oracle directed-edge diagnostics from replay JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _walk(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("directed_edge_search"), dict):
            yield value["directed_edge_search"]
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _first_directed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for diagnostics in _walk(payload):
        return diagnostics
    raise SystemExit(f"No directed_edge_search diagnostics found in {path}")


def _errors(diagnostics: dict[str, Any]) -> list[float]:
    progress = (
        diagnostics.get("event_progress")
        or diagnostics.get("best_terminal_event_progress")
        or {}
    )
    per_event = progress.get("per_event") or []
    return [
        round(float(item.get("abs_error", 1.0)), 6)
        for item in per_event
        if isinstance(item, dict)
    ]


def _first_divergence(left: list[float], right: list[float]) -> int | None:
    count = min(len(left), len(right))
    for index in range(count):
        if abs(float(left[index]) - float(right[index])) > 1e-6:
            return index
    if len(left) != len(right):
        return count
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("production", type=Path)
    parser.add_argument("oracle", type=Path)
    args = parser.parse_args()

    production = _first_directed(args.production)
    oracle = _first_directed(args.oracle)
    production_errors = _errors(production)
    oracle_errors = _errors(oracle)
    summary = {
        "oracle_first_divergence_event": _first_divergence(
            production_errors,
            oracle_errors,
        ),
        "oracle_path_event_errors": oracle_errors,
        "production_beam_path_event_errors": production_errors,
        "production_reason": production.get("reason"),
        "oracle_reason": oracle.get("reason"),
        "production_matched_turn_events": (
            production.get("matched_turn_events")
            or production.get("maximum_matched_turn_events")
        ),
        "oracle_matched_turn_events": (
            oracle.get("matched_turn_events")
            or oracle.get("maximum_matched_turn_events")
        ),
        "production_states_evaluated": production.get("states_evaluated"),
        "oracle_states_evaluated": oracle.get("states_evaluated"),
        "oracle_dominance_reject_count": oracle.get(
            "oracle_dominance_reject_count"
        ),
        "oracle_frontier_trim_count": oracle.get("oracle_frontier_trim_count"),
        "production_event_window_kill_count": production.get(
            "event_window_kill_count"
        ),
        "oracle_event_window_kill_count": oracle.get(
            "event_window_kill_count"
        ),
        "production_event_sign_mismatch_kill_count": production.get(
            "event_sign_mismatch_kill_count"
        ),
        "oracle_event_sign_mismatch_kill_count": oracle.get(
            "event_sign_mismatch_kill_count"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
