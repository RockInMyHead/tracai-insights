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
- The walkable mask, metric scale, forward time and final route connectivity
  are hard production constraints. A collision-free but physically implausible
  route is not publishable.
- Independent LingBot is a guarded rescue source. It requires a monotonic time
  base and a speed within `0.72..1.80` of the calibrated walking-speed prior;
  fusion support never grants it an unrestricted shape fallback.
- No map overlay is returned when no candidate satisfies both the visual/map
  topology and the source-specific metric policy.

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
5. Reject hypotheses outside the source-specific metric envelope before the
   repair beam is built. Pre-score the survivors by restricted-area
   intersections, plan bounds, clearance, walking speed and yaw deviation.
6. Inflate the no-go mask by a `0.28 m` person radius.
7. Constrain the best hypotheses to the walkable mask. A start marker may only
   be projected locally (`<= 1.5 m`); larger snaps are rejected.
8. Repair collision runs with local eight-connected A*. Its objective
   combines path length, wall clearance, and deviation from the visual path.
9. Simplify A* output only through verified collision-free line of sight.
   This prevents sparse resampling from cutting through equipment again.
10. Recompute length and speed after A*/Viterbi. A repair that breaks the metric
    envelope is rejected even when its collision count is zero.
11. Select authoritative R3/fusion first. Evaluate independent LingBot only
    after all authoritative candidates fail, without inheriting their relaxed
    shape policy.
12. Before publication, densify the polyline to at most `0.75 m` per segment,
    reconstruct timestamps by source arc fraction, and revalidate every sample,
    segment, plan bound and connected-component ID.
13. Hard-accept only a zero-intersection, single-component final polyline.

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
- active motion duration, final speed and speed/prior ratio;
- raw/corrected collision ratios;
- rerouted segment count;
- correction median and p95 in meters;
- route-length ratio;
- published length and maximum published segment length;
- observation policy and timestamp provenance;
- confidence, `quality_warnings`, and machine-readable metric, topology or
  publication rejection reasons.

The map result is accepted when every rendered segment is inside the fixed
walkable mask, belongs to one walkable component and satisfies the applicable
metric policy. Independent monocular scale is never inferred from mask fit
alone.
