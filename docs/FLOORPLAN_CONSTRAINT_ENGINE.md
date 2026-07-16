# Floorplan Constraint Engine

Production map matching for the fixed Kerama Marazzi 2025 floor plan.

## Contract

- The plan is immutable and identified as `kerama_marazzi_2025`.
- The operator supplies only `reference_point` and `direction_point`, both in
  image percentages (`0..100`).
- R3 remains the primary source of relative motion and turn semantics. A
  geometrically compatible LingBot observation may supply a guarded fused
  candidate; the fixed plan selects between primary R3 and that candidate.
- The engine may add `map_trajectory`; it never overwrites
  `plan_trajectory` or `raw_trajectory_3d`.
- The walkable mask is a hard production constraint. Scale, speed and
  correction-size inconsistencies lower confidence but do not suppress a
  collision-free map route.
- No map overlay is returned only when the walkable graph contains no feasible
  route for any tested scale/yaw hypothesis.

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
2. When available, robustly align LingBot to R3 with a non-reflecting similarity
   transform. Reject it on residual or signed-turn disagreement; otherwise
   construct a confidence-weighted fusion candidate.
3. Anchor each eligible observation candidate at the operator's start point and
   initial direction.
4. Generate global scale and small yaw hypotheses. Metric scale is seeded by
   active walking time, not full video duration, so stationary intervals do
   not stretch the path.
5. Pre-score every hypothesis by restricted-area intersections, plan bounds,
   clearance, walking-speed plausibility, and yaw deviation.
6. Inflate the no-go mask by a `0.28 m` person radius.
7. Constrain the best hypotheses to the walkable mask. Points outside the
   plan and start markers inside no-go cells are projected to the nearest
   walkable cell.
8. Repair collision runs with local eight-connected A*. Its objective
   combines path length, wall clearance, and deviation from the visual path.
9. Simplify A* output only through verified collision-free line of sight.
   This prevents sparse resampling from cutting through equipment again.
10. Select the feasible constrained hypothesis and observation source with the
    smallest combined visual, correction and route-distortion cost.
11. Hard-accept only zero-intersection, inside-plan routes. Report route-length,
    correction and speed inconsistencies as `quality_warnings`.

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
- confidence, `quality_warnings`, and a machine-readable rejection reason when
  the constrained graph truly has no feasible route.

The map result is accepted when every rendered segment is inside the fixed
walkable mask. Monocular-quality priors affect ranking and confidence only.
