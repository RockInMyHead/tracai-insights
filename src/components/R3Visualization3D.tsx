import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { apiClient } from "@/lib/api";

interface PoseData {
  frame: number;
  pose: number[][];
  intrinsics?: number[][];
}

interface Props {
  videoId?: string;
  points: number[][];
  poses: PoseData[];
  pointCloud: number[][] | null;  // real point cloud from R³ depth maps
  totalFrames: number;
  distance: number;
}

// ─── Terrain noise for synthetic fallback ──────────────────────────────
function terrainHeight(x: number, z: number): number {
  return (
    Math.sin(x * 0.08) * Math.cos(z * 0.06) * 2.5 +
    Math.sin(x * 0.15 + z * 0.1) * 1.5 +
    Math.cos(x * 0.04 - z * 0.12) * 2.0 +
    Math.sin(x * 0.3 + z * 0.25) * 0.8
  );
}

// ─── Generate synthetic cloud from poses (fallback when no real point cloud) ─
function generatePointCloudFromPoses(
  poses: PoseData[],
  pointsPerCam: number = 600,
  maxTotal: number = 30000,
  scale: number = 1.0,
): Float32Array {
  if (poses.length < 1) return new Float32Array(0);

  const step = Math.max(1, Math.floor(poses.length / Math.min(poses.length, 60)));
  const sampledPoses: PoseData[] = [];
  for (let i = 0; i < poses.length; i += step) sampledPoses.push(poses[i]);
  if (sampledPoses.length < 1) sampledPoses.push(poses[0]);

  const actualPerCam = Math.max(20, Math.floor(maxTotal / Math.max(sampledPoses.length, 1)));
  const totalPoints = Math.min(sampledPoses.length * actualPerCam, maxTotal);
  const positions = new Float32Array(totalPoints * 3);
  const colors = new Float32Array(totalPoints * 3);

  let idx = 0;
  for (let pi = 0; pi < sampledPoses.length && idx < totalPoints; pi++) {
    const p = sampledPoses[pi];
    const mat = p.pose;

    const R = [
      [mat[0][0], mat[0][1], mat[0][2]],
      [mat[1][0], mat[1][1], mat[1][2]],
      [mat[2][0], mat[2][1], mat[2][2]],
    ];
    const tx = mat[0][3] * scale;
    const ty = mat[1][3] * scale;
    const tz = mat[2][3] * scale;

    const viewDir = [-R[0][2], -R[1][2], -R[2][2]];

    const ptsThisCam = Math.min(actualPerCam, totalPoints - idx);
    for (let j = 0; j < ptsThisCam && idx < totalPoints; j++) {
      const u = (Math.random() - 0.5) * 2;
      const v = (Math.random() - 0.5) * 2;
      const depthBase = 3 + Math.random() * 8;
      const approxWorldX = tx + viewDir[0] * depthBase + u * 2;
      const approxWorldZ = tz + viewDir[2] * depthBase + v * 2;
      const terrainOffset = terrainHeight(approxWorldX, approxWorldZ);
      const depth = Math.max(1, depthBase + terrainOffset * 0.3 + (Math.random() - 0.5) * 0.5);

      const xCam = u * depth * 0.8;
      const yCam = v * depth * 0.6;
      const zCam = depth;

      const wx = (R[0][0] * xCam + R[0][1] * yCam + R[0][2] * zCam) * scale + tx;
      const wy = (R[1][0] * xCam + R[1][1] * yCam + R[1][2] * zCam) * scale + ty;
      const wz = (R[2][0] * xCam + R[2][1] * yCam + R[2][2] * zCam) * scale + tz;

      positions[idx * 3] = wx;
      positions[idx * 3 + 1] = wy;
      positions[idx * 3 + 2] = wz;
      idx++;
    }
  }

  // Colors by height
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < idx; i++) {
    const y = positions[i * 3 + 1];
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const rangeY = Math.max(maxY - minY, 0.1);
  for (let i = 0; i < idx; i++) {
    const t = (positions[i * 3 + 1] - minY) / rangeY;
    colors[i * 3] = Math.min(1, 1.5 * t);
    colors[i * 3 + 1] = Math.min(1, Math.max(0, 1.5 * t - 0.3));
    colors[i * 3 + 2] = Math.min(1, Math.max(0, 2.5 * t - 1.5)) * 0.7 + 0.3;
  }

  const finalPos = new Float32Array(idx * 3);
  for (let i = 0; i < idx * 3; i++) finalPos[i] = positions[i];
  return finalPos;
}

// ─── Build camera frustum from pose matrix ─────────────────────────────
function buildCameraFrustum(
  pose: number[][],
  color: number,
  scale: number = 0.18,
): THREE.LineSegments {
  const matrix = new THREE.Matrix4();
  matrix.set(
    pose[0][0], pose[0][1], pose[0][2], pose[0][3],
    pose[1][0], pose[1][1], pose[1][2], pose[1][3],
    pose[2][0], pose[2][1], pose[2][2], pose[2][3],
    0, 0, 0, 1,
  );

  const f = 0.8 * scale;
  const n = 0.15 * scale;
  const hw = 0.4 * scale;
  const hh = 0.3 * scale;

  const pts = [
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(-hw, -hh, -f), new THREE.Vector3(hw, -hh, -f),
    new THREE.Vector3(hw, hh, -f), new THREE.Vector3(-hw, hh, -f),
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(-hw * n, -hh * n, -n), new THREE.Vector3(hw * n, -hh * n, -n),
    new THREE.Vector3(hw * n, hh * n, -n), new THREE.Vector3(-hw * n, hh * n, -n),
    new THREE.Vector3(-hw, -hh, -f), new THREE.Vector3(-hw * n, -hh * n, -n),
    new THREE.Vector3(hw, -hh, -f), new THREE.Vector3(hw * n, -hh * n, -n),
    new THREE.Vector3(hw, hh, -f), new THREE.Vector3(hw * n, hh * n, -n),
    new THREE.Vector3(-hw, hh, -f), new THREE.Vector3(-hw * n, hh * n, -n),
  ];

  const indices = [
    0, 1, 0, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8,
    1, 2, 2, 3, 3, 4, 4, 1, 5, 6, 6, 7, 7, 8, 8, 5,
  ];

  const positions = new Float32Array(indices.length * 3);
  for (let i = 0; i < indices.length; i++) {
    const p = pts[indices[i]].clone().applyMatrix4(matrix);
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z;
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.22 });
  return new THREE.LineSegments(geom, mat);
}

// ─── Color helpers ────────────────────────────────────────────────────

/** Jet colormap: blue → cyan → green → yellow → red */
function jetColor(t: number): [number, number, number] {
  const tt = Math.max(0, Math.min(1, t));
  if (tt < 0.25) {
    const s = tt / 0.25;
    return [0, s, 0.5 + s * 0.5];
  } else if (tt < 0.5) {
    const s = (tt - 0.25) / 0.25;
    return [0, 1, 1 - s * 0.5];
  } else if (tt < 0.75) {
    const s = (tt - 0.5) / 0.25;
    return [s, 1, 0];
  } else {
    const s = (tt - 0.75) / 0.25;
    return [1, 1 - s, 0];
  }
}

function heightColor(t: number): [number, number, number] {
  return jetColor(t);
}

function createPointSprite(): THREE.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 31);
    gradient.addColorStop(0, "rgba(255,255,255,1)");
    gradient.addColorStop(0.55, "rgba(255,255,255,0.95)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(32, 32, 31, 0, Math.PI * 2);
    ctx.fill();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function percentile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

function finitePoint3(p: number[]): [number, number, number] | null {
  if (!Array.isArray(p) || p.length < 2) return null;
  const x = Number(p[0]);
  const y = Number(p[1]);
  const z = Number(p[2] || 0);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
  return [x, y, z];
}

function normalizeRgbChannel(value: number, sourceIsByteRgb: boolean): number {
  const normalized = sourceIsByteRgb ? value / 255 : value;
  return Math.max(0, Math.min(1, normalized));
}

function normalizeRgbTriplet(r: number, g: number, b: number): [number, number, number] | null {
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) return null;

  // R3 currently returns RGB as floats in 0..1. Older/exported clouds may carry 0..255 bytes.
  const sourceIsByteRgb = r > 1 || g > 1 || b > 1;
  return [
    normalizeRgbChannel(r, sourceIsByteRgb),
    normalizeRgbChannel(g, sourceIsByteRgb),
    normalizeRgbChannel(b, sourceIsByteRgb),
  ];
}

function pointRgb(p: number[]): [number, number, number] | null {
  if (!Array.isArray(p) || p.length < 6) return null;
  const r = Number(p[3]);
  const g = Number(p[4]);
  const b = Number(p[5]);
  return normalizeRgbTriplet(r, g, b);
}

function pointConf(p: number[]): number {
  if (!Array.isArray(p) || p.length < 7) return 2.0;
  const conf = Number(p[6]);
  return Number.isFinite(conf) ? conf : 0;
}

function buildDisplayTransform(points: number[][]) {
  const finite = points.map(finitePoint3).filter((p): p is [number, number, number] => Boolean(p));
  const toThree = (x: number, y: number, z: number) => new THREE.Vector3(x, z, -y);
  if (finite.length === 0) {
    return {
      center: new THREE.Vector3(),
      scale: 1,
      apply: (p: number[]) => toThree(p[0] || 0, p[1] || 0, p[2] || 0),
    };
  }

  const xs = finite.map(p => p[0]);
  const ys = finite.map(p => p[1]);
  const zs = finite.map(p => p[2]);
  const center = new THREE.Vector3(
    percentile(xs, 0.5),
    percentile(ys, 0.5),
    percentile(zs, 0.5),
  );

  const radius = percentile(
    finite.map(p => new THREE.Vector3(p[0], p[1], p[2]).distanceTo(center)),
    0.95,
  );
  const scale = radius > 1e-6 ? 18 / radius : 1;

  return {
    center,
    scale,
    apply: (p: number[]) => toThree(
      ((p[0] || 0) - center.x) * scale,
      ((p[1] || 0) - center.y) * scale,
      ((p[2] || 0) - center.z) * scale,
    ),
  };
}

/**
 * Apply the same similarity transform to a complete c2w pose as to cloud
 * points.  Moving only the translation made camera frustums face the wrong
 * way after the [x,y,z] → [x,z,-y] viewer conversion.
 */
function transformPoseForDisplay(
  pose: number[][],
  transform: ReturnType<typeof buildDisplayTransform>,
): number[][] {
  if (!Array.isArray(pose) || pose.length < 3 || !Array.isArray(pose[0]) || pose[0].length < 4) {
    return pose;
  }
  const raw = new THREE.Matrix4();
  raw.set(
    pose[0][0], pose[0][1], pose[0][2], pose[0][3],
    pose[1][0], pose[1][1], pose[1][2], pose[1][3],
    pose[2][0], pose[2][1], pose[2][2], pose[2][3],
    0, 0, 0, 1,
  );
  const axis = new THREE.Matrix4().set(
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, -1, 0, 0,
    0, 0, 0, 1,
  );
  const translate = new THREE.Matrix4().makeTranslation(-transform.center.x, -transform.center.y, -transform.center.z);
  const scale = new THREE.Matrix4().makeScale(transform.scale, transform.scale, transform.scale);
  const rawToDisplay = axis.clone().multiply(scale).multiply(translate);
  const display = rawToDisplay.clone().multiply(raw).multiply(rawToDisplay.clone().invert());
  const e = display.elements;
  return [
    [e[0], e[4], e[8], e[12]],
    [e[1], e[5], e[9], e[13]],
    [e[2], e[6], e[10], e[14]],
    [0, 0, 0, 1],
  ];
}

function cleanPointCloudForDisplay(
  cloud: number[][],
  trajectory: number[][],
  transform: ReturnType<typeof buildDisplayTransform>,
  minConfidence: number,
  colorMode: R3ColorMode,
  frameRange: [number, number],
): { positions: Float32Array; colors: Float32Array | null; hasRgb: boolean } {
  const finite = cloud
    .map((raw) => ({ xyz: finitePoint3(raw), rgb: pointRgb(raw), conf: pointConf(raw), frame: pointFrameIdx(raw) }))
    .filter((p): p is { xyz: [number, number, number]; rgb: [number, number, number] | null; conf: number; frame: number | null } => Boolean(p.xyz) && p.conf >= minConfidence);
  if (finite.length < 50) {
    return { positions: new Float32Array(0), colors: null, hasRgb: false };
  }

  const xs = finite.map(p => p.xyz[0]);
  const ys = finite.map(p => p.xyz[1]);
  const zs = finite.map(p => p.xyz[2]);
  const bounds = {
    x0: percentile(xs, 0.02), x1: percentile(xs, 0.98),
    y0: percentile(ys, 0.02), y1: percentile(ys, 0.98),
    z0: percentile(zs, 0.02), z1: percentile(zs, 0.98),
  };

  const transformed: number[] = [];
  const colors: number[] = [];
  let rgbCount = 0;
  const confVals = finite.map(p => p.conf);
  const confLow = percentile(confVals, 0.05);
  const confHigh = Math.max(confLow + 1e-6, percentile(confVals, 0.98));
  const depthLow = bounds.z0;
  const depthHigh = Math.max(depthLow + 1e-6, bounds.z1);
  const frameLow = frameRange[0];
  const frameHigh = Math.max(frameLow + 1, frameRange[1]);
  for (const p of finite) {
    const xyz = p.xyz;
    if (xyz[0] < bounds.x0 || xyz[0] > bounds.x1 || xyz[1] < bounds.y0 || xyz[1] > bounds.y1 || xyz[2] < bounds.z0 || xyz[2] > bounds.z1) {
      continue;
    }
    const q = transform.apply(xyz);
    transformed.push(q.x, q.y, q.z);

    if (colorMode === "confidence") {
      const [cr, cg, cb] = heightColor((p.conf - confLow) / (confHigh - confLow));
      colors.push(cr, cg, cb);
      rgbCount++;
    } else if (colorMode === "frame" && p.frame !== null) {
      const [cr, cg, cb] = heightColor((p.frame - frameLow) / (frameHigh - frameLow));
      colors.push(cr, cg, cb);
      rgbCount++;
    } else if (colorMode === "depth") {
      const [cr, cg, cb] = heightColor((xyz[2] - depthLow) / (depthHigh - depthLow));
      colors.push(cr, cg, cb);
      rgbCount++;
    } else if (p.rgb) {
      colors.push(p.rgb[0], p.rgb[1], p.rgb[2]);
      rgbCount++;
    } else {
      colors.push(0, 0, 0);
    }
  }

  return {
    positions: new Float32Array(transformed),
    colors: rgbCount > transformed.length / 9 ? new Float32Array(colors) : null,
    hasRgb: rgbCount > transformed.length / 9,
  };
}

function downsamplePointBuffers(
  positions: Float32Array,
  colors: Float32Array | null,
  maxPoints: number,
): { positions: Float32Array; colors: Float32Array | null; count: number } {
  const total = Math.floor(positions.length / 3);
  const target = Math.max(100, Math.min(total, maxPoints));
  if (total <= target) return { positions, colors, count: total };

  const outPos = new Float32Array(target * 3);
  const outCol = colors ? new Float32Array(target * 3) : null;
  const step = total / target;

  for (let i = 0; i < target; i++) {
    const src = Math.min(total - 1, Math.floor(i * step)) * 3;
    const dst = i * 3;
    outPos[dst] = positions[src];
    outPos[dst + 1] = positions[src + 1];
    outPos[dst + 2] = positions[src + 2];
    if (colors && outCol) {
      outCol[dst] = colors[src];
      outCol[dst + 1] = colors[src + 1];
      outCol[dst + 2] = colors[src + 2];
    }
  }

  return { positions: outPos, colors: outCol, count: target };
}

type ViewMode = "orbit" | "top" | "front" | "right";
type SamplingStrategy = "confidence_top" | "random" | "voxel" | "per_frame_uniform";
type R3ColorMode = "rgb" | "confidence" | "frame" | "depth";
type R3Diagnostics = Awaited<ReturnType<typeof apiClient.getR3Diagnostics>>;
type R3TrajectoryQuality = NonNullable<
  NonNullable<Awaited<ReturnType<typeof apiClient.getR3PointCloudFiltered>>["stats"]>["trajectory_quality"]
>;
type R3PresetId = "clean" | "balanced" | "dense";

const R3_PRESETS: Array<{
  id: R3PresetId;
  label: string;
  maxRenderPoints: number;
  minConfidence: number;
  samplingStrategy: SamplingStrategy;
}> = [
  { id: "clean", label: "Чисто", minConfidence: 1.8, samplingStrategy: "confidence_top", maxRenderPoints: 50000 },
  { id: "balanced", label: "Баланс", minConfidence: 1.4, samplingStrategy: "confidence_top", maxRenderPoints: 100000 },
  { id: "dense", label: "Плотно", minConfidence: 1.0, samplingStrategy: "voxel", maxRenderPoints: 200000 },
];

function formatCompactNumber(value: number | undefined | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toLocaleString();
}

function formatPercentile(value: number | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value >= 10 ? value.toFixed(1) : value.toFixed(3);
}

function pointFrameIdx(p: number[]): number | null {
  if (!Array.isArray(p) || p.length < 8) return null;
  const frame = Number(p[7]);
  return Number.isFinite(frame) ? frame : null;
}

// ─── Main Component ────────────────────────────────────────────────────
export default function R3Visualization3D({ videoId, points, poses, pointCloud, totalFrames, distance }: Props) {
  const [maxRenderPoints, setMaxRenderPoints] = useState(100000);
  const [pointSize, setPointSize] = useState(0.75);
  const [pointOpacity, setPointOpacity] = useState(0.72);
  const [minConfidence, setMinConfidence] = useState(1.4);
  const [frameStart, setFrameStart] = useState(0);
  const [frameEnd, setFrameEnd] = useState(Math.max(0, totalFrames - 1));
  const [r3FrameCount, setR3FrameCount] = useState(0);
  const [samplingStrategy, setSamplingStrategy] = useState<SamplingStrategy>("per_frame_uniform");
  const [colorMode, setColorMode] = useState<R3ColorMode>("rgb");
  const [filteredPointCloud, setFilteredPointCloud] = useState<number[][] | null>(null);
  const [filteredTrajectory, setFilteredTrajectory] = useState<number[][] | null>(null);
  const [filteredStats, setFilteredStats] = useState<{
    source_points: number;
    filtered_points: number;
    returned_points: number;
    sampling_strategy: string;
    trajectory_quality?: R3TrajectoryQuality | null;
  } | null>(null);
  const [filteredDiagnostics, setFilteredDiagnostics] = useState<{
    pointcloud_file: string;
    pointcloud_shape: number[];
    has_conf: boolean;
    has_frame_idx: boolean;
    run_params?: Record<string, unknown>;
    stale_run?: boolean;
  } | null>(null);
  const [showDebug, setShowDebug] = useState(false);
  const [diagnostics, setDiagnostics] = useState<R3Diagnostics | null>(null);
  const [diagnosticsError, setDiagnosticsError] = useState<string | null>(null);
  const [isRefetchingCloud, setIsRefetchingCloud] = useState(false);
  const [cloudFetchError, setCloudFetchError] = useState<string | null>(null);
  const [cloudBuildStatus, setCloudBuildStatus] = useState<{ progress: number; message: string } | null>(null);
  const [cloudRetryToken, setCloudRetryToken] = useState(0);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [showCameras, setShowCameras] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("top");
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    animationId: number;
    pointCloudObj: THREE.Points | null;
    trajectoryLine: THREE.Line | null;
    positionSpheres: THREE.Mesh[];
    frustums: THREE.LineSegments[];
    grid: THREE.GridHelper;
    axes: THREE.AxesHelper;
  } | null>(null);
  const initRef = useRef(false);

  // Debug clouds use sequential R3 frame ids, not original video frame ids.
  const r3FrameMax = Math.max(0, (r3FrameCount || totalFrames) - 1);
  const effectivePointCloud = filteredPointCloud ?? pointCloud;
  const effectiveTrajectory = filteredTrajectory && filteredTrajectory.length >= 2 ? filteredTrajectory : points;

  useEffect(() => {
    setR3FrameCount(0);
    setCloudBuildStatus(null);
    setCloudRetryToken(0);
  }, [videoId]);

  useEffect(() => {
    setFrameStart(0);
    setFrameEnd(r3FrameMax);
  }, [r3FrameMax]);

  useEffect(() => {
    if (!videoId || points.length < 2) return;
    const controller = new AbortController();
    let retryTimeout: number | null = null;
    const timeout = window.setTimeout(() => {
      setIsRefetchingCloud(true);
      setCloudFetchError(null);
      apiClient.getR3PointCloudFiltered(videoId, {
        maxPoints: maxRenderPoints,
        minConf: minConfidence,
        frameStart: r3FrameMax > 0 ? frameStart : undefined,
        frameEnd: r3FrameMax > 0 ? frameEnd : undefined,
        samplingStrategy,
        includeTrajectory: true,
        includeCameras: false,
      }).then((resp) => {
        if (controller.signal.aborted) return;
        const valid = Array.isArray(resp.points)
          ? resp.points.filter(p => Array.isArray(p) && p.length >= 3)
          : [];
        if (resp.success && valid.length >= 50) {
          setCloudBuildStatus(null);
          setFilteredPointCloud(valid);
          // Keep the Three.js scene in raw R3 world coordinates.  The map API
          // returns its floor-plane path separately as `plan_trajectory`.
          const rawTrajectory = resp.raw_trajectory_3d ?? resp.trajectory;
          if (Array.isArray(rawTrajectory) && rawTrajectory.length >= 2) {
            setFilteredTrajectory(rawTrajectory.filter(p => Array.isArray(p) && p.length >= 3));
            setR3FrameCount(rawTrajectory.length);
          }
          setFilteredStats(resp.stats ?? null);
          setFilteredDiagnostics(resp.diagnostics ?? null);
        }
      }).catch(async (err) => {
        if (controller.signal.aborted) return;
        try {
          const status = await apiClient.getR3PointCloudStatus(videoId);
          if (controller.signal.aborted) return;
          if (["not_started", "queued", "processing"].includes(status.status)) {
            setCloudFetchError(null);
            setCloudBuildStatus({
              progress: typeof status.progress === "number" ? status.progress : 0,
              message: status.message || "Строится 3D-облако",
            });
            retryTimeout = window.setTimeout(() => setCloudRetryToken(value => value + 1), 2000);
            return;
          }
          setCloudBuildStatus(null);
          setCloudFetchError(status.error || status.message || (err instanceof Error ? err.message : String(err)));
        } catch {
          console.warn("Failed to fetch filtered R3 point cloud:", err);
          setCloudFetchError(err instanceof Error ? err.message : String(err));
        }
      }).finally(() => {
        if (!controller.signal.aborted) setIsRefetchingCloud(false);
      });
    }, 350);

    return () => {
      controller.abort();
      window.clearTimeout(timeout);
      if (retryTimeout !== null) window.clearTimeout(retryTimeout);
    };
  }, [videoId, points.length, maxRenderPoints, minConfidence, frameStart, frameEnd, samplingStrategy, r3FrameMax, cloudRetryToken]);

  useEffect(() => {
    if (!showDebug || !videoId) return;
    let cancelled = false;
    setDiagnosticsError(null);
    apiClient.getR3Diagnostics(videoId).then((resp) => {
      if (!cancelled) setDiagnostics(resp);
    }).catch((err) => {
      if (!cancelled) {
        console.warn("Failed to fetch R3 diagnostics:", err);
        setDiagnosticsError(err instanceof Error ? err.message : String(err));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [showDebug, videoId, filteredStats?.returned_points]);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    const container = containerRef.current;
    if (!container) return;

    const w = container.clientWidth;
    const h = Math.max(400, container.clientHeight || 500);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080816);

    const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 2000);
    camera.position.set(10, 8, 12);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.minDistance = 0.5;
    controls.maxDistance = 500;
    controls.target.set(0, 0, 0);
    controls.update();

    scene.fog = new THREE.Fog(0x080816, 50, 150);

    // Lights
    const ambient = new THREE.AmbientLight(0x404060, 0.6);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);
    const dirLight2 = new THREE.DirectionalLight(0x4488ff, 0.3);
    dirLight2.position.set(-10, -5, -10);
    scene.add(dirLight2);
    const hemiLight = new THREE.HemisphereLight(0x4444ff, 0x444422, 0.4);
    scene.add(hemiLight);

    // Grid
    const gridHelper = new THREE.GridHelper(40, 40, 0x333388, 0x222255);
    gridHelper.position.y = -2;
    scene.add(gridHelper);
    const axesHelper = new THREE.AxesHelper(3);
    scene.add(axesHelper);

    // Stars
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(3000);
    for (let i = 0; i < 3000; i++) starPos[i] = (Math.random() - 0.5) * 300;
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({ color: 0x555599, size: 0.08, transparent: true, opacity: 0.4 });
    scene.add(new THREE.Points(starGeo, starMat));

    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      sceneRef.current!.animationId = requestAnimationFrame(animate);
    };

    sceneRef.current = {
      scene, camera, renderer, controls,
      animationId: requestAnimationFrame(animate),
      pointCloudObj: null,
      trajectoryLine: null,
      positionSpheres: [],
      frustums: [],
      grid: gridHelper,
      axes: axesHelper,
    };

    const handleResize = () => {
      const w2 = container.clientWidth;
      const h2 = Math.max(400, container.clientHeight || 500);
      camera.aspect = w2 / h2;
      camera.updateProjectionMatrix();
      renderer.setSize(w2, h2);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(sceneRef.current!.animationId);
      renderer.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
      sceneRef.current = null;
      initRef.current = false;
    };
  }, []);

  // Update visualization when points/poses/pointCloud change
  useEffect(() => {
    const ctx = sceneRef.current;
    if (!ctx) return;

    const { scene } = ctx;

    try {
      ctx.grid.visible = showGrid;
      ctx.axes.visible = showGrid;

      // ─── 1. Clean up old objects ─────────────────────────────
      const cleanup = () => {
        if (ctx.pointCloudObj) {
          scene.remove(ctx.pointCloudObj);
          ctx.pointCloudObj.geometry.dispose();
          const mat = ctx.pointCloudObj.material as THREE.PointsMaterial;
          mat.map?.dispose();
          mat.dispose();
          ctx.pointCloudObj = null;
        }
        if (ctx.trajectoryLine) {
          scene.remove(ctx.trajectoryLine);
          ctx.trajectoryLine.geometry.dispose();
          (ctx.trajectoryLine.material as THREE.Material).dispose();
          ctx.trajectoryLine = null;
        }
        for (const s of ctx.positionSpheres) {
          scene.remove(s);
          s.geometry.dispose();
          (s.material as THREE.Material).dispose();
        }
        ctx.positionSpheres = [];
        for (const f of ctx.frustums) {
          scene.remove(f);
          f.geometry.dispose();
          (f.material as THREE.Material).dispose();
        }
        ctx.frustums = [];
      };
      cleanup();

      // Validate trajectory points
      const validPoints = effectiveTrajectory.filter(
        p => Array.isArray(p) && p.length >= 2 && p.every(v => typeof v === 'number')
      );
      if (validPoints.length < 2) return;
      const displayTransform = buildDisplayTransform(validPoints);
      const displayPoints = validPoints.map(p => {
        const q = displayTransform.apply(p);
        return [q.x, q.y, q.z];
      });

      // ─── 2. Point cloud (REAL from R³ when available) ─────────
      const hasRealPointCloud = effectivePointCloud !== null && effectivePointCloud.length >= 50;
      let cloudMat: THREE.PointsMaterial | null = null;
      let cloudGeo: THREE.BufferGeometry | null = null;

      if (hasRealPointCloud) {
        // ── Render the REAL depth-projected point cloud ──
        const cleanedCloud = cleanPointCloudForDisplay(
          effectivePointCloud,
          validPoints,
          displayTransform,
          minConfidence,
          colorMode,
          [frameStart, frameEnd],
        );
        const sampledCloud = downsamplePointBuffers(cleanedCloud.positions, cleanedCloud.colors, maxRenderPoints);
        const positions = sampledCloud.positions;
        const numPts = positions.length / 3;
        let minY = Infinity, maxY = -Infinity;

        for (let i = 0; i < numPts; i++) {
          const y = positions[i * 3 + 1];
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }

        const rangeY = Math.max(maxY - minY, 0.1);
        let colors = sampledCloud.colors;
        if (!colors) {
          colors = new Float32Array(numPts * 3);
          for (let i = 0; i < numPts; i++) {
            const t = (positions[i * 3 + 1] - minY) / rangeY;
            const [cr, cg, cb] = heightColor(Math.min(1, Math.max(0, t)));
            colors[i * 3] = cr;
            colors[i * 3 + 1] = cg;
            colors[i * 3 + 2] = cb;
          }
        }

        cloudGeo = new THREE.BufferGeometry();
        cloudGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        cloudGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        cloudMat = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          map: createPointSprite(),
          transparent: true,
          opacity: pointOpacity,
          alphaTest: 0.08,
          sizeAttenuation: false,
          blending: THREE.NormalBlending,
          depthWrite: false,
        });
      } else {
        // ── Fallback: generate synthetic cloud from poses or trajectory ──
        const validPoses = poses.filter(
          p => p && Array.isArray(p.pose) && p.pose.length >= 3 &&
               Array.isArray(p.pose[0]) && p.pose[0].length >= 4,
        );

        let cloudPositions: Float32Array;
        if (validPoses.length >= 2) {
          cloudPositions = generatePointCloudFromPoses(validPoses, 500, 25000);
        } else {
          cloudPositions = generateFallbackCloud(validPoints, 20000);
        }

        if (cloudPositions.length >= 3 && cloudPositions.length / 3 >= 50) {
          const numPts = cloudPositions.length / 3;
          let minY = Infinity, maxY = -Infinity;
          for (let i = 0; i < numPts; i++) {
            const y = cloudPositions[i * 3 + 1];
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
          const rangeY = Math.max(maxY - minY, 0.1);

          const colors = new Float32Array(numPts * 3);
          for (let i = 0; i < numPts; i++) {
            const t = (cloudPositions[i * 3 + 1] - minY) / rangeY;
            const [cr, cg, cb] = heightColor(Math.min(1, Math.max(0, t)));
            colors[i * 3] = cr;
            colors[i * 3 + 1] = cg;
            colors[i * 3 + 2] = cb;
          }

          cloudGeo = new THREE.BufferGeometry();
          cloudGeo.setAttribute("position", new THREE.BufferAttribute(cloudPositions, 3));
          cloudGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

          cloudMat = new THREE.PointsMaterial({
            size: 1.35,
            vertexColors: true,
            map: createPointSprite(),
            transparent: true,
            opacity: 0.8,
            alphaTest: 0.08,
            sizeAttenuation: false,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
          });
        }
      }

      if (cloudGeo && cloudMat) {
        const cloud = new THREE.Points(cloudGeo, cloudMat);
        scene.add(cloud);
        ctx.pointCloudObj = cloud;
      }

      // ─── 3. Trajectory line ─────────────────────────────────
      const curvePts = displayPoints.map(p => new THREE.Vector3(p[0], p[1], p[2] || 0));
      if (showTrajectory) {
        const curveGeo = new THREE.BufferGeometry().setFromPoints(curvePts);
        const pc = curvePts.length;
        const lineColors = new Float32Array(pc * 3);
        for (let i = 0; i < pc; i++) {
          const t = pc > 1 ? i / (pc - 1) : 0;
          const [cr, cg, cb] = heightColor(t);
          lineColors[i * 3] = cr;
          lineColors[i * 3 + 1] = cg * 0.6 + 0.4;
          lineColors[i * 3 + 2] = cb * 0.5 + 0.5;
        }
        curveGeo.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));

        const line = new THREE.Line(
          curveGeo,
          new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.82 }),
        );
        scene.add(line);
        ctx.trajectoryLine = line;
      }

      // ─── 4. Camera position spheres ─────────────────────────
      if (showTrajectory) {
        const sphereStep = Math.max(1, Math.floor(validPoints.length / 36));
        const sphereGeo = new THREE.SphereGeometry(0.035, 8, 8);
        for (let i = 0; i < displayPoints.length; i += sphereStep) {
          const p = displayPoints[i];
          const t = displayPoints.length > 1 ? i / (displayPoints.length - 1) : 0;
          const [cr, cg, cb] = heightColor(t);
          const sphereMat = new THREE.MeshBasicMaterial({
            color: new THREE.Color(cr, cg * 0.6 + 0.4, cb * 0.5 + 0.5),
          });
          const sphere = new THREE.Mesh(sphereGeo.clone(), sphereMat);
          sphere.position.set(p[0], p[1], p[2] || 0);
          scene.add(sphere);
          ctx.positionSpheres.push(sphere);
        }
      }

      // ─── 5. Camera frustums ─────────────────────────────────
      const validPoses = poses.filter(
        p => p && Array.isArray(p.pose) && p.pose.length >= 3 &&
             Array.isArray(p.pose[0]) && p.pose[0].length >= 4,
      );

      let frustumPoses: PoseData[];
      if (validPoses.length >= 2) {
        frustumPoses = validPoses.map(p => {
          return { ...p, pose: transformPoseForDisplay(p.pose, displayTransform) };
        });
      } else {
        frustumPoses = [];
        const fstep = Math.max(1, Math.floor(displayPoints.length / 24));
        for (let i = 0; i < displayPoints.length; i += fstep) {
          const pos = displayPoints[i];
          const nextPos = displayPoints[Math.min(i + 1, displayPoints.length - 1)];
          const prevPos = displayPoints[Math.max(i - 1, 0)];
          const dir = [
            nextPos[0] - prevPos[0],
            nextPos[1] - prevPos[1],
            (nextPos[2] || 0) - (prevPos[2] || 0),
          ];
          const len = Math.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) || 1;
          const zAxis = [-dir[0] / len, -dir[1] / len, -dir[2] / len];
          const up = [0, 1, 0];
          const xAxis = [
            up[1] * zAxis[2] - up[2] * zAxis[1],
            up[2] * zAxis[0] - up[0] * zAxis[2],
            up[0] * zAxis[1] - up[1] * zAxis[0],
          ];
          const xLen = Math.sqrt(xAxis[0] * xAxis[0] + xAxis[1] * xAxis[1] + xAxis[2] * xAxis[2]) || 1;
          const yAxis = [
            zAxis[1] * xAxis[0] / xLen - zAxis[2] * xAxis[1] / xLen,
            zAxis[2] * xAxis[0] / xLen - zAxis[0] * xAxis[2] / xLen,
            zAxis[0] * xAxis[1] / xLen - zAxis[1] * xAxis[0] / xLen,
          ];
          frustumPoses.push({
            frame: i,
            pose: [
              [xAxis[0] / xLen, yAxis[0], zAxis[0], pos[0]],
              [xAxis[1] / xLen, yAxis[1], zAxis[1], pos[1]],
              [xAxis[2] / xLen, yAxis[2], zAxis[2], pos[2] || 0],
              [0, 0, 0, 1],
            ],
          });
        }
      }

      if (showCameras) {
        const FRUSTUM_COLORS = [0x3b82f6, 0x8b5cf6, 0x10b981, 0xf59e0b, 0xef4444, 0x06b6d4, 0xec4899];
        const fstep = Math.max(1, Math.floor(frustumPoses.length / 14));
        let fidx = 0;
        for (let i = 0; i < frustumPoses.length; i += fstep) {
          const frustum = buildCameraFrustum(frustumPoses[i].pose, FRUSTUM_COLORS[fidx % FRUSTUM_COLORS.length]);
          scene.add(frustum);
          ctx.frustums.push(frustum);
          fidx++;
        }
      }

      // ─── 6. Auto-center camera ──────────────────────────────
      let cx = 0, cy = 0, cz = 0;
      for (const p of displayPoints) {
        cx += p[0]; cy += p[1]; cz += p[2] || 0;
      }
      const n = displayPoints.length;
      cx /= n; cy /= n; cz /= n;

      // If we have a real point cloud, center on it for better framing
      let sceneRadius = 0;
      let pcCx = cx, pcCy = cy, pcCz = cz;
      if (hasRealPointCloud && cloudGeo) {
        const posAttr = cloudGeo.getAttribute("position") as THREE.BufferAttribute | undefined;
        let _cx = 0, _cy = 0, _cz = 0;
        const validCount = posAttr?.count || 0;
        for (let i = 0; i < validCount; i++) {
          _cx += posAttr.getX(i); _cy += posAttr.getY(i); _cz += posAttr.getZ(i);
        }
        if (validCount > 0) {
          pcCx = _cx / validCount; pcCy = _cy / validCount; pcCz = _cz / validCount;
          for (let i = 0; i < validCount; i++) {
            const d = Math.sqrt((posAttr.getX(i) - pcCx) ** 2 + (posAttr.getY(i) - pcCy) ** 2 + (posAttr.getZ(i) - pcCz) ** 2);
            if (d > sceneRadius) sceneRadius = d;
          }
        }
      } else {
        for (const p of displayPoints) {
          const d = Math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + ((p[2] || 0) - cz) ** 2);
          if (d > sceneRadius) sceneRadius = d;
        }
      }

      ctx.controls.target.set(pcCx, pcCy, pcCz);
      const camDist = Math.max(18, sceneRadius * 3.2);
      if (viewMode === "top") {
        ctx.camera.position.set(pcCx, pcCy + camDist, pcCz + camDist * 0.02);
        ctx.camera.up.set(0, 0, -1);
      } else if (viewMode === "front") {
        ctx.camera.position.set(pcCx, pcCy + camDist * 0.15, pcCz + camDist);
        ctx.camera.up.set(0, 1, 0);
      } else if (viewMode === "right") {
        ctx.camera.position.set(pcCx + camDist, pcCy + camDist * 0.15, pcCz);
        ctx.camera.up.set(0, 1, 0);
      } else {
        ctx.camera.position.set(pcCx + camDist * 0.55, pcCy + camDist * 0.38, pcCz + camDist);
        ctx.camera.up.set(0, 1, 0);
      }
      ctx.controls.update();

    } catch (err) {
      console.warn("R3Visualization3D render error:", err);
    }
  }, [effectiveTrajectory, poses, effectivePointCloud, maxRenderPoints, pointSize, pointOpacity, minConfidence, colorMode, frameStart, frameEnd, showTrajectory, showCameras, showGrid, viewMode]);

  const hasRealCloud = effectivePointCloud !== null && effectivePointCloud.length >= 50;
  const hasRgbCloud = hasRealCloud && effectivePointCloud.some(p => Array.isArray(p) && p.length >= 6);
  const debugPointcloud = diagnostics?.pointcloud;
  const debugPercentiles = diagnostics?.conf_stats?.percentiles;
  const debugFile = filteredDiagnostics?.pointcloud_file || debugPointcloud?.file || "—";
  const debugHasFrameIdx = filteredDiagnostics?.has_frame_idx ?? debugPointcloud?.has_frame_idx;
  const debugSourcePoints = filteredStats?.source_points;
  const debugFilteredPoints = filteredStats?.filtered_points;
  const debugReturnedPoints = filteredStats?.returned_points ?? effectivePointCloud?.length;
  const trajectoryQuality = filteredStats?.trajectory_quality ?? null;
  const runMode = typeof filteredDiagnostics?.run_params?.mode === "string"
    ? filteredDiagnostics.run_params.mode
    : null;
  const isStaleRun = filteredDiagnostics?.stale_run === true;
  const displayDistance =
    typeof trajectoryQuality?.cleaned_distance === "number" && Number.isFinite(trajectoryQuality.cleaned_distance)
      ? trajectoryQuality.cleaned_distance
      : distance;
  const isUnstablePose = trajectoryQuality?.quality === "unstable_pose";
  const clippedSteps = trajectoryQuality?.clipped_steps ?? 0;
  const cleanedPoints = trajectoryQuality?.cleaned_points ?? effectiveTrajectory.length;
  const activePreset = R3_PRESETS.find(
    preset =>
      preset.maxRenderPoints === maxRenderPoints &&
      preset.minConfidence === minConfidence &&
      preset.samplingStrategy === samplingStrategy,
  )?.id;
  const applyPreset = (presetId: R3PresetId) => {
    const preset = R3_PRESETS.find(p => p.id === presetId);
    if (!preset) return;
    setMaxRenderPoints(preset.maxRenderPoints);
    setMinConfidence(preset.minConfidence);
    setSamplingStrategy(preset.samplingStrategy);
  };

  return (
    <div className="relative w-full h-full min-h-[400px] rounded-lg overflow-hidden border border-border/30">
      <div ref={containerRef} className="w-full h-full" style={{ minHeight: 500 }} />

      {/* Info overlay */}
      <div className="absolute top-3 left-3 flex flex-wrap gap-2">
        <div className="px-2.5 py-1 rounded-md bg-background/80 backdrop-blur-sm border border-border/30 text-xs space-y-0.5">
          <div className="flex items-center gap-3">
            <span className="text-muted-foreground">Точек траектории</span>
            <span className="font-bold font-mono text-primary">{cleanedPoints}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-muted-foreground">Кадров</span>
            <span className="font-bold font-mono text-amber-400">{totalFrames}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-muted-foreground">Дистанция</span>
            <span className="font-bold font-mono text-green-400">{displayDistance.toFixed(2)} м</span>
          </div>
          {hasRealCloud && (
            <div className="flex items-center gap-3">
              <span className="text-muted-foreground">Облако точек (R³)</span>
              <span className="font-bold font-mono text-cyan-400">{effectivePointCloud.length.toLocaleString()}</span>
            </div>
          )}
          {filteredStats && (
            <div className="flex items-center gap-3">
              <span className="text-muted-foreground">Фильтр R³</span>
              <span className="font-bold font-mono text-cyan-400">
                {filteredStats.returned_points.toLocaleString()}/{filteredStats.filtered_points.toLocaleString()}
              </span>
            </div>
          )}
          {isUnstablePose && (
            <div className="max-w-[260px] rounded border border-amber-400/30 bg-amber-500/10 px-2 py-1 text-[10px] leading-snug text-amber-200">
              R³ pose нестабилен: исправлено скачков {clippedSteps}. Используйте эту 3D-карту как облако сцены, не как точный маршрут.
            </div>
          )}
          {isStaleRun && (
            <div className="max-w-[260px] rounded border border-red-400/30 bg-red-500/10 px-2 py-1 text-[10px] leading-snug text-red-200">
              Это старый R³ output ({runMode || "unknown"}). Перезапустите R³-анализ, чтобы получить strided+fallback+metric результат.
            </div>
          )}
        </div>
      </div>

      <div
        className="absolute top-3 right-3 w-56 rounded-md bg-background/85 backdrop-blur-sm border border-border/30 p-3 text-xs text-foreground shadow-lg space-y-2"
        onPointerDown={(e) => e.stopPropagation()}
        onWheel={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="font-medium text-cyan-300">R³ настройки</span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className={["text-[10px]", showDebug ? "text-cyan-300" : "text-muted-foreground hover:text-foreground"].join(" ")}
              onClick={() => setShowDebug(v => !v)}
            >
              debug
            </button>
            <button
              type="button"
              className="text-[10px] text-muted-foreground hover:text-foreground"
              onClick={() => {
                setMaxRenderPoints(100000);
                setPointSize(0.75);
                setPointOpacity(0.72);
                setMinConfidence(1.4);
                setFrameStart(0);
                setFrameEnd(r3FrameMax);
                setSamplingStrategy("per_frame_uniform");
                setColorMode("rgb");
                setShowTrajectory(true);
                setShowCameras(false);
                setShowGrid(true);
                setViewMode("top");
              }}
            >
              reset
            </button>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-1">
          {R3_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={[
                "rounded border px-1.5 py-1",
                activePreset === preset.id
                  ? "border-cyan-400/60 bg-cyan-400/20 text-cyan-100"
                  : "border-border/40 text-muted-foreground hover:text-foreground",
              ].join(" ")}
              onClick={() => applyPreset(preset.id)}
              title={`min_conf=${preset.minConfidence}, ${preset.samplingStrategy}, ${preset.maxRenderPoints.toLocaleString()} точек`}
            >
              {preset.label}
            </button>
          ))}
        </div>

        <label className="block space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Точек</span>
            <span className="font-mono text-cyan-300">{maxRenderPoints.toLocaleString()}</span>
          </div>
          <input
            className="w-full accent-cyan-400"
            type="range"
            min={5000}
            max={200000}
            step={5000}
            value={maxRenderPoints}
            onChange={(e) => setMaxRenderPoints(Number(e.target.value))}
          />
        </label>

        <label className="block space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Кадры</span>
            <span className="font-mono text-cyan-300">{frameStart}-{frameEnd}</span>
          </div>
          <input
            className="w-full accent-cyan-400"
            type="range"
            min={0}
            max={r3FrameMax}
            step={1}
            value={Math.min(frameStart, r3FrameMax)}
            disabled={r3FrameMax <= 0}
            onChange={(e) => {
              const next = Number(e.target.value);
              setFrameStart(Math.min(next, frameEnd));
            }}
          />
          <input
            className="w-full accent-cyan-400"
            type="range"
            min={0}
            max={r3FrameMax}
            step={1}
            value={Math.min(frameEnd, r3FrameMax)}
            disabled={r3FrameMax <= 0}
            onChange={(e) => {
              const next = Number(e.target.value);
              setFrameEnd(Math.max(next, frameStart));
            }}
          />
        </label>

        <label className="block space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Сэмплинг</span>
            <span className="font-mono text-cyan-300">{samplingStrategy}</span>
          </div>
          <select
            className="w-full rounded border border-border/40 bg-background/80 px-2 py-1 text-xs"
            value={samplingStrategy}
            onChange={(e) => setSamplingStrategy(e.target.value as SamplingStrategy)}
          >
            <option value="confidence_top">confidence_top</option>
            <option value="per_frame_uniform">per_frame_uniform</option>
            <option value="voxel">voxel</option>
            <option value="random">random</option>
          </select>
        </label>

        <label className="block space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Цвет</span>
            <span className="font-mono text-cyan-300">{colorMode}</span>
          </div>
          <select
            className="w-full rounded border border-border/40 bg-background/80 px-2 py-1 text-xs"
            value={colorMode}
            onChange={(e) => setColorMode(e.target.value as R3ColorMode)}
          >
            <option value="rgb">RGB</option>
            <option value="confidence">confidence</option>
            <option value="frame">frame_idx</option>
            <option value="depth">depth</option>
          </select>
        </label>

        <label className="block space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Размер</span>
            <span className="font-mono text-cyan-300">{pointSize.toFixed(1)} px</span>
          </div>
          <input
            className="w-full accent-cyan-400"
            type="range"
            min={0.4}
            max={3.0}
            step={0.1}
            value={pointSize}
            onChange={(e) => setPointSize(Number(e.target.value))}
          />
        </label>

        <label className="block space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Прозрачность</span>
            <span className="font-mono text-cyan-300">{Math.round(pointOpacity * 100)}%</span>
          </div>
          <input
            className="w-full accent-cyan-400"
            type="range"
            min={0.2}
            max={1}
            step={0.05}
            value={pointOpacity}
            onChange={(e) => setPointOpacity(Number(e.target.value))}
          />
        </label>

        <label className="block space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Доверие</span>
            <span className="font-mono text-cyan-300">{minConfidence.toFixed(1)}</span>
          </div>
          <input
            className="w-full accent-cyan-400"
            type="range"
            min={1.0}
            max={3.0}
            step={0.1}
            value={minConfidence}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
          />
        </label>

        <div className="grid grid-cols-3 gap-1 pt-1">
          <button
            type="button"
            className={["rounded border px-1.5 py-1", showTrajectory ? "border-cyan-400/50 bg-cyan-400/15 text-cyan-200" : "border-border/40 text-muted-foreground"].join(" ")}
            onClick={() => setShowTrajectory(v => !v)}
          >
            путь
          </button>
          <button
            type="button"
            className={["rounded border px-1.5 py-1", showCameras ? "border-cyan-400/50 bg-cyan-400/15 text-cyan-200" : "border-border/40 text-muted-foreground"].join(" ")}
            onClick={() => setShowCameras(v => !v)}
          >
            камеры
          </button>
          <button
            type="button"
            className={["rounded border px-1.5 py-1", showGrid ? "border-cyan-400/50 bg-cyan-400/15 text-cyan-200" : "border-border/40 text-muted-foreground"].join(" ")}
            onClick={() => setShowGrid(v => !v)}
          >
            сетка
          </button>
        </div>

        <div className="grid grid-cols-4 gap-1 pt-1">
          {([
            ["top", "верх"],
            ["orbit", "3D"],
            ["front", "фронт"],
            ["right", "сбоку"],
          ] as [ViewMode, string][]).map(([mode, label]) => (
            <button
              key={mode}
              type="button"
              className={["rounded border px-1.5 py-1", viewMode === mode ? "border-cyan-400/50 bg-cyan-400/15 text-cyan-200" : "border-border/40 text-muted-foreground"].join(" ")}
              onClick={() => setViewMode(mode)}
            >
              {label}
            </button>
          ))}
        </div>

        {showDebug && (
          <div className="mt-2 rounded border border-cyan-400/20 bg-black/35 p-2 text-[10px] leading-relaxed">
            <div className="mb-1 font-medium text-cyan-300">R³ debug</div>
            <div className="grid grid-cols-[auto_1fr] gap-x-2">
              <span className="text-muted-foreground">file</span>
              <span className="truncate font-mono text-cyan-100" title={debugFile}>{debugFile}</span>
              <span className="text-muted-foreground">frame_idx</span>
              <span className="font-mono text-cyan-100">{debugHasFrameIdx === undefined ? "—" : debugHasFrameIdx ? "yes" : "no"}</span>
              <span className="text-muted-foreground">points</span>
              <span className="font-mono text-cyan-100">
                {formatCompactNumber(debugSourcePoints)} / {formatCompactNumber(debugFilteredPoints)} / {formatCompactNumber(debugReturnedPoints)}
              </span>
              <span className="text-muted-foreground">frames</span>
              <span className="font-mono text-cyan-100">{frameStart}-{frameEnd}</span>
              <span className="text-muted-foreground">strategy</span>
              <span className="font-mono text-cyan-100">{samplingStrategy}</span>
              <span className="text-muted-foreground">color</span>
              <span className="font-mono text-cyan-100">{colorMode}</span>
              <span className="text-muted-foreground">shape</span>
              <span className="font-mono text-cyan-100">
                {(filteredDiagnostics?.pointcloud_shape || debugPointcloud?.shape || []).join("×") || "—"}
              </span>
              <span className="text-muted-foreground">pose</span>
              <span className={["font-mono", isUnstablePose ? "text-amber-200" : "text-cyan-100"].join(" ")}>
                {trajectoryQuality?.quality || "—"}
              </span>
              <span className="text-muted-foreground">run</span>
              <span className={["font-mono", isStaleRun ? "text-red-200" : "text-cyan-100"].join(" ")}>
                {runMode || "—"}
              </span>
              <span className="text-muted-foreground">distance</span>
              <span className="font-mono text-cyan-100">{displayDistance.toFixed(2)} m</span>
            </div>

            <div className="mt-2 border-t border-cyan-400/10 pt-1">
              <div className="text-muted-foreground">confidence percentiles</div>
              <div className="grid grid-cols-4 gap-1 font-mono text-cyan-100">
                <span>p50 {formatPercentile(debugPercentiles?.p50)}</span>
                <span>p90 {formatPercentile(debugPercentiles?.p90)}</span>
                <span>p95 {formatPercentile(debugPercentiles?.p95)}</span>
                <span>p99 {formatPercentile(debugPercentiles?.p99)}</span>
              </div>
            </div>

            {diagnosticsError && (
              <div className="mt-1 text-red-300">diagnostics error: {diagnosticsError}</div>
            )}
            {cloudFetchError && (
              <div className="mt-1 text-red-300">filtered cloud error: {cloudFetchError}</div>
            )}
          </div>
        )}
      </div>

      <div className={[
        "absolute bottom-3 left-3 text-[10px]",
        hasRealCloud ? "text-cyan-400/70" : "text-muted-foreground/50",
      ].join(" ")}>
        {hasRealCloud
          ? `${hasRgbCloud ? "✓ REAL R³ RGB filtered point cloud" : "✓ REAL R³ filtered depth point cloud"}${isRefetchingCloud ? " • loading..." : ""} • Drag to rotate • Scroll to zoom`
          : cloudBuildStatus
            ? `${cloudBuildStatus.message} (${cloudBuildStatus.progress}%) • Траектория уже готова`
            : `Drag to rotate • Scroll to zoom • Right-click to pan${cloudFetchError ? " • pointcloud fetch error" : ""}`}
      </div>
    </div>
  );
}

// ─── Fallback: terrain-structured cloud from trajectory ────────────────
function generateFallbackCloud(
  trajectory: number[][],
  count: number = 20000,
): Float32Array {
  const validTraj = trajectory.filter(p => Array.isArray(p) && p.length >= 2);
  if (validTraj.length < 3) return new Float32Array(0);

  const positions = new Float32Array(count * 3);
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const p of validTraj) {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    const z = p[2] || 0;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const rangeX = Math.max(maxX - minX, 1);
  const rangeZ = Math.max(maxZ - minZ, 1);

  let idx = 0;
  for (let i = 0; i < count && idx < count; i++) {
    const segIdx = Math.floor(Math.random() * (validTraj.length - 1));
    const t = Math.random();
    const p0 = validTraj[segIdx];
    const p1 = validTraj[Math.min(segIdx + 1, validTraj.length - 1)];

    const baseX = p0[0] + (p1[0] - p0[0]) * t;
    const baseY = p0[1] + (p1[1] - p0[1]) * t;
    const baseZ = (p0[2] || 0) + ((p1[2] || 0) - (p0[2] || 0)) * t;

    const scatterX = (Math.random() - 0.5) * rangeX * 0.08;
    const scatterZ = (Math.random() - 0.5) * rangeZ * 0.08;
    const worldX = baseX + scatterX;
    const worldZ = baseZ + scatterZ;
    const terrainH = terrainHeight(worldX, worldZ) * 0.5;
    const worldY = baseY + terrainH + (Math.random() - 0.5) * 0.3;

    positions[idx * 3] = worldX;
    positions[idx * 3 + 1] = worldY;
    positions[idx * 3 + 2] = worldZ;
    idx++;
  }

  const finalPos = new Float32Array(idx * 3);
  for (let i = 0; i < idx * 3; i++) finalPos[i] = positions[i];
  return finalPos;
}
