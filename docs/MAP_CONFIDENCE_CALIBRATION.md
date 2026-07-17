# Map confidence calibration

`confidence` remains a backward-compatible quality score. It is not presented
as a probability. `probability_correct` stays `null` until a calibration model
has been fitted from real annotated production routes.

Create one JSON object per line:

```json
{"quality_score": 0.41, "correct": 0}
{"quality_score": 0.83, "correct": 1}
```

`correct=1` should mean that the route passed the agreed production tolerance
(for example: correct corridor sequence, no wrong chirality and 95% map error
below the chosen metre threshold). Use at least 20 independent routes, with at
least five positive and five negative examples.

Fit the monotone isotonic model:

```bash
python scripts/calibrate_map_confidence.py annotated_routes.jsonl
```

The model is written to
`backend/assets/floorplans/map_confidence_calibration.json`. Keep the annotated
dataset version and tolerance definition with every released model.
