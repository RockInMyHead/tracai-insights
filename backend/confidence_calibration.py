"""Data-driven calibration for floorplan map-matching quality scores."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_MODEL = Path(__file__).resolve().parent / "assets" / "floorplans" / "map_confidence_calibration.json"


def fit_isotonic(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Fit monotone P(correct|score) with the pool-adjacent-violators algorithm."""
    pairs = sorted(
        (float(score), int(label))
        for score, label in zip(scores, labels)
        if math.isfinite(float(score)) and int(label) in (0, 1)
    )
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if len(pairs) < 20 or positives < 5 or negatives < 5:
        raise ValueError("at least 20 routes with >=5 positive and >=5 negative labels are required")
    blocks: list[dict[str, float]] = []
    for score, label in pairs:
        blocks.append({"left": score, "right": score, "weight": 1.0, "mean": float(label)})
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            blocks.append({
                "left": left["left"],
                "right": right["right"],
                "weight": weight,
                "mean": (
                    left["mean"] * left["weight"] + right["mean"] * right["weight"]
                ) / weight,
            })
    return {
        "schema_version": 1,
        "method": "isotonic_pav",
        "sample_count": len(pairs),
        "positive_count": positives,
        "negative_count": negatives,
        "breakpoints": [block["right"] for block in blocks],
        "probabilities": [block["mean"] for block in blocks],
    }


def calibrated_probability(
    quality_score: float,
    model_path: Path = DEFAULT_MODEL,
) -> tuple[float | None, dict[str, Any]]:
    if not model_path.exists():
        return None, {"status": "unavailable", "reason": "annotated_routes_required"}
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        x = np.asarray(model["breakpoints"], dtype=np.float64)
        y = np.asarray(model["probabilities"], dtype=np.float64)
        if len(x) == 0 or len(x) != len(y) or not np.all(np.diff(x) >= 0):
            raise ValueError("invalid isotonic model")
        probability = float(np.interp(float(quality_score), x, y, left=y[0], right=y[-1]))
        return float(np.clip(probability, 0.0, 1.0)), {
            "status": "calibrated",
            "method": model.get("method", "isotonic_pav"),
            "sample_count": int(model.get("sample_count", 0)),
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, {"status": "invalid", "reason": str(exc)}
