# Floorplan Constraint Engine

Production map matching for the fixed Kerama Marazzi 2025 floor plan.

## Contract

- The plan is immutable and identified as `kerama_marazzi_2025`.
- The operator supplies only `reference_point` and `direction_point`, both in
  image percentages (`0..100`).
- R3 remains the source of relative motion and turn semantics.
- The engine may add `map_trajectory`; it never overwrites
  `plan_trajectory` or `raw_trajectory_3d`.
- A rejected quality gate produces diagnostics and no map overlay. The UI
  must not silently auto-fit a rejected path onto the plan.

## Metric calibration

The interior rectangle of the office below the medical room is measured at
`70 x 58` pixels in the canonical 160 dpi render. Its known area is `10 m²`:

```text
meters_per_pixel = sqrt(10 / (70 * 58)) = 0.0496291667
```

The calibration and source rectangle are stored in
`backend/assets/floorplans/kerama_marazzi_2025.json`.

## Algorithm

1. Convert the R3 floor trajectory from physical `x-forward, y-left` into
   plan-image `x-right, y-down` coordinates.
2. Anchor it at the operator's start point and initial direction.
3. Generate global scale and small yaw hypotheses. Metric scale is seeded by
   active walking time, not full video duration, so stationary intervals do
   not stretch the path.
4. Score every hypothesis by restricted-area intersections, plan bounds,
   clearance, walking-speed plausibility, and yaw deviation.
5. Inflate the no-go mask by a `0.28 m` person radius.
6. Repair only collision runs with local eight-connected A*. Its objective
   combines path length, wall clearance, and deviation from the visual path.
7. Simplify A* output only through verified collision-free line of sight.
   This prevents sparse resampling from cutting through equipment again.
8. Apply quality gates for residual intersections, outside-plan samples,
   route-length distortion, correction magnitude, and walking speed.

Turn angles and left/right labels stay sourced from R3. Their positions are
transferred to the constrained route by arc-length fraction.

## Static assets

- `public/floorplans/kerama-marazzi-2025.pdf` — canonical marked PDF.
- `public/floorplans/kerama-marazzi-2025.png` — exact 160 dpi display render.
- `backend/assets/floorplans/kerama_marazzi_2025_obstacles.png` — isolated
  red no-go mask.
- `backend/assets/floorplans/kerama_marazzi_2025.json` — calibration and
  engine settings.

Rebuild them deterministically with:

```bash
python backend/tools/prepare_kerama_floorplan.py ORIGINAL.pdf MARKED.pdf
```

The original PDF is used only to distinguish user-added red constraints from
red CAD content that was already present.

## Response diagnostics

`floorplan_constraint` and `processing_stats.floorplan_constraint` include:

- selected scale and yaw;
- active motion duration and estimated speed;
- raw/corrected collision ratios;
- rerouted segment count;
- correction median and p95 in meters;
- route-length ratio;
- confidence and a machine-readable rejection reason.

The map result is accepted only when every hard gate passes.
