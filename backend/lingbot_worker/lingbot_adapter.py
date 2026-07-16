from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from . import config


@dataclass
class LingBotRunOptions:
    session_id: str
    input_video: Path
    output_dir: Path
    log_path: Path
    fps: int
    target_frames: int
    keyframe_interval: int
    use_sdpa: bool
    mask_sky: bool


class LingBotMapAdapter:
    """Thin wrapper around LingBot-Map inference/demo code.

    The worker owns session/storage/API concerns. This adapter owns only the model
    invocation and normalization of LingBot-Map outputs into stable TrackAI files.
    """

    def __init__(
        self,
        repo_path: Path = config.LINGBOT_REPO_PATH,
        model_path: Path = config.LINGBOT_MODEL_PATH,
        python_bin: str = config.LINGBOT_PYTHON,
        timeout_seconds: int = config.LINGBOT_TIMEOUT_SECONDS,
    ):
        self.repo_path = repo_path
        self.model_path = model_path
        self.python_bin = python_bin or sys.executable
        self.timeout_seconds = timeout_seconds

    def validate_environment(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"LingBot-Map repo not found: {self.repo_path}. "
                "Set LINGBOT_REPO_PATH or clone https://github.com/Robbyant/lingbot-map."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LingBot-Map checkpoint not found: {self.model_path}. "
                "Set LINGBOT_MODEL_PATH or download the HuggingFace model."
            )

    def run(self, options: LingBotRunOptions) -> Dict[str, Any]:
        self.validate_environment()
        started = time.time()
        options.output_dir.mkdir(parents=True, exist_ok=True)

        with options.log_path.open("a", encoding="utf-8") as log:
            log.write(f"[lingbot] start session={options.session_id}\n")
            log.write(f"[lingbot] repo={self.repo_path}\n")
            log.write(f"[lingbot] model={self.model_path}\n")
            log.write(f"[lingbot] input={options.input_video}\n")

            self._run_subprocess(options, log)

        artifacts = self._normalize_outputs(options.output_dir)
        artifacts["timings"] = {"total_seconds": round(time.time() - started, 3)}
        return artifacts

    def _run_subprocess(self, options: LingBotRunOptions, log) -> None:
        batch_demo_py = self.repo_path / "demo_render" / "batch_demo.py"
        if not batch_demo_py.exists():
            raise FileNotFoundError(f"LingBot-Map batch demo not found at {batch_demo_py}")

        cmd = [
            self.python_bin,
            str(batch_demo_py),
            "--video_path",
            str(options.input_video),
            "--output_folder",
            str(options.output_dir),
            "--model_path",
            str(self.model_path),
            "--image_size",
            str(config.DEFAULT_IMAGE_SIZE),
            "--keyframe_interval",
            str(options.keyframe_interval),
            "--mode",
            config.DEFAULT_MODE,
            "--point_cloud_stride",
            "3",
            "--save_predictions",
            "--no_render",
            "--video_suffix",
            "_lingbot_preview",
        ]
        if options.target_frames > 0:
            cmd.extend(["--target_frames", str(options.target_frames)])
        else:
            cmd.extend(["--fps", str(options.fps)])

        # RTX 3090 deployment currently has no FlashInfer package. LingBot-Map
        # falls back safely when --use_sdpa is passed, so keep it unconditional.
        cmd.append("--use_sdpa")
        if options.mask_sky:
            cmd.append("--mask_sky")

        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0"))
        env["PYTHONPATH"] = f"{self.repo_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

        log.write("[lingbot] command=" + " ".join(cmd) + "\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
            return_code = proc.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise TimeoutError(f"LingBot-Map timed out after {self.timeout_seconds}s")

        if return_code != 0:
            raise RuntimeError(f"LingBot-Map failed with exit code {return_code}")

    def _normalize_outputs(self, output_dir: Path) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {
            "trajectory": None,
            "pointcloud": None,
            "raw_files": [],
        }

        raw_files = [p for p in output_dir.rglob("*") if p.is_file()]
        artifacts["raw_file_count"] = len(raw_files)
        artifacts["raw_files"] = [str(p.relative_to(output_dir)) for p in raw_files[:500]]

        trajectory = self._discover_trajectory(raw_files)
        trajectory_path = output_dir / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["trajectory"] = str(trajectory_path)

        pointcloud_path = self._discover_or_create_pointcloud(raw_files, output_dir)
        artifacts["pointcloud"] = str(pointcloud_path) if pointcloud_path else None
        return artifacts

    def _discover_trajectory(self, files: Iterable[Path]) -> Dict[str, Any]:
        candidates = list(files)
        for path in candidates:
            if path.suffix.lower() == ".json" and any(k in path.name.lower() for k in ("traj", "pose", "camera")):
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue

        for path in candidates:
            if path.suffix.lower() == ".npz":
                try:
                    data = np.load(path)
                    for key in (
                        "extrinsic",
                        "trajectory",
                        "poses",
                        "camera_poses",
                        "extrinsics",
                        "pred_cam",
                        "camera",
                        "cam_T_world",
                        "world_T_cam",
                    ):
                        if key in data:
                            pose_array = np.asarray(data[key])
                            if (
                                pose_array.ndim >= 3
                                and pose_array.shape[-2:] in {(3, 4), (4, 4)}
                            ):
                                confidence = data.get("depth_conf")
                                poses = self._poses_from_extrinsics(
                                    pose_array,
                                    confidence=confidence,
                                    input_is_w2c=(key == "cam_T_world"),
                                    source_file=path.name,
                                )
                                if poses:
                                    return {
                                        "poses": poses,
                                        "source_file": path.name,
                                        "source_key": key,
                                        "extrinsic_convention": "c2w",
                                    }
                            if (
                                key == "extrinsic"
                                and pose_array.ndim == 2
                                and pose_array.shape in {(3, 4), (4, 4)}
                            ):
                                # Per-frame prediction; normalized together
                                # with the other frame_*.npz files below.
                                continue
                            return {"poses": pose_array.tolist(), "source_file": path.name, "source_key": key}
                except Exception:
                    continue

        frame_files = self._prediction_frame_files(candidates)
        poses = []
        for idx, path in enumerate(frame_files):
            try:
                data = np.load(path)
                if "extrinsic" not in data:
                    continue
                # Upstream demo.postprocess explicitly converts the camera
                # prediction from w2c to c2w before --save_predictions.  Do
                # not invert it a second time here.
                c2w = self._as_4x4(data["extrinsic"].astype(np.float32))
                confidence = self._frame_confidence(data.get("depth_conf"))
                poses.append(
                    {
                        "frame_idx": idx,
                        "source_file": path.name,
                        "position": c2w[:3, 3].astype(float).tolist(),
                        "c2w": c2w.astype(float).tolist(),
                        "confidence": confidence,
                    }
                )
            except Exception:
                continue
        if poses:
            return {"poses": poses, "source": "lingbot_per_frame_npz"}

        return {"poses": [], "source_file": None}

    def _frame_confidence(self, value: Optional[np.ndarray]) -> Optional[float]:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return None
        return float(np.median(finite))

    def _poses_from_extrinsics(
        self,
        extrinsics: np.ndarray,
        *,
        confidence: Optional[np.ndarray],
        input_is_w2c: bool,
        source_file: str,
    ) -> list[Dict[str, Any]]:
        matrices = np.asarray(extrinsics)
        while matrices.ndim > 3 and matrices.shape[0] == 1:
            matrices = matrices[0]
        if matrices.ndim != 3:
            return []
        confidence_array = np.asarray(confidence) if confidence is not None else None
        poses: list[Dict[str, Any]] = []
        for index, matrix in enumerate(matrices):
            try:
                c2w = self._as_4x4(np.asarray(matrix, dtype=np.float32))
                if input_is_w2c:
                    c2w = np.linalg.inv(c2w)
                frame_confidence = None
                if confidence_array is not None and confidence_array.ndim >= 1:
                    conf_index = min(index, confidence_array.shape[0] - 1)
                    frame_confidence = self._frame_confidence(confidence_array[conf_index])
                poses.append({
                    "frame_idx": index,
                    "source_file": source_file,
                    "position": c2w[:3, 3].astype(float).tolist(),
                    "c2w": c2w.astype(float).tolist(),
                    "confidence": frame_confidence,
                })
            except Exception:
                continue
        return poses

    def _discover_or_create_pointcloud(self, files: Iterable[Path], output_dir: Path) -> Optional[Path]:
        for path in files:
            name = path.name.lower()
            if path.suffix.lower() in {".ply", ".glb"} and any(k in name for k in ("point", "cloud", "pcd", "depth", "lingbot")):
                return path

        for path in files:
            if path.suffix.lower() != ".npz":
                continue
            try:
                data = np.load(path)
                xyz_key = next(
                    (
                        k
                        for k in (
                            "points",
                            "xyz",
                            "pointcloud",
                            "pts3d",
                            "world_points",
                            "points3d",
                            "point_map",
                        )
                        if k in data
                    ),
                    None,
                )
                if not xyz_key:
                    continue
                out = output_dir / "pointcloud.npz"
                arrays = {"xyz": data[xyz_key]}
                for key in ("rgb", "colors", "confidence", "conf"):
                    if key in data:
                        arrays[key] = data[key]
                np.savez_compressed(out, **arrays)
                return out
            except Exception:
                continue
        generated = self._create_pointcloud_from_predictions(files, output_dir)
        if generated:
            return generated
        return None

    def _prediction_frame_files(self, files: Iterable[Path]) -> list[Path]:
        frame_files = [
            p
            for p in files
            if p.suffix.lower() == ".npz"
            and p.name.startswith("frame_")
            and p.parent.name not in {"input_frames", "_incoming"}
        ]
        return sorted(frame_files)

    def _as_4x4(self, extrinsic: np.ndarray) -> np.ndarray:
        if extrinsic.shape == (4, 4):
            return extrinsic
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = extrinsic.reshape(3, 4)
        return out

    def _create_pointcloud_from_predictions(self, files: Iterable[Path], output_dir: Path) -> Optional[Path]:
        frame_files = self._prediction_frame_files(files)
        if not frame_files:
            return None

        max_points = int(os.getenv("LINGBOT_POINTCLOUD_MAX_POINTS", "150000"))
        max_frames = int(os.getenv("LINGBOT_POINTCLOUD_MAX_FRAMES", "240"))
        if len(frame_files) > max_frames:
            indices = np.linspace(0, len(frame_files) - 1, max_frames).round().astype(int)
            frame_files = [frame_files[int(i)] for i in indices]

        per_frame_budget = max(128, max_points // max(1, len(frame_files)))
        xyz_parts = []
        rgb_parts = []
        conf_parts = []
        frame_parts = []

        for frame_idx, path in enumerate(frame_files):
            try:
                data = np.load(path)
                required = {"depth", "extrinsic", "intrinsic"}
                if not required.issubset(set(data.keys())):
                    continue
                depth = np.asarray(data["depth"], dtype=np.float32)
                if depth.ndim == 3:
                    depth = depth[..., 0]
                h, w = depth.shape
                conf = np.asarray(data["depth_conf"], dtype=np.float32) if "depth_conf" in data else np.ones((h, w), dtype=np.float32)
                image = np.asarray(data["images"], dtype=np.float32) if "images" in data else None
                if image is not None and image.ndim == 3 and image.shape[0] == 3:
                    image = np.moveaxis(image, 0, -1)

                valid = np.isfinite(depth) & (depth > 0)
                valid_idx = np.flatnonzero(valid.reshape(-1))
                if valid_idx.size == 0:
                    continue
                if valid_idx.size > per_frame_budget:
                    pick = np.linspace(0, valid_idx.size - 1, per_frame_budget).round().astype(int)
                    valid_idx = valid_idx[pick]

                vv, uu = np.unravel_index(valid_idx, (h, w))
                z = depth[vv, uu]
                k = np.asarray(data["intrinsic"], dtype=np.float32).reshape(3, 3)
                fx = float(k[0, 0]) if abs(float(k[0, 0])) > 1e-6 else 1.0
                fy = float(k[1, 1]) if abs(float(k[1, 1])) > 1e-6 else 1.0
                cx = float(k[0, 2])
                cy = float(k[1, 2])
                x = (uu.astype(np.float32) - cx) / fx * z
                y = (vv.astype(np.float32) - cy) / fy * z
                cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
                # Saved LingBot predictions already use c2w extrinsics.
                c2w = self._as_4x4(np.asarray(data["extrinsic"], dtype=np.float32))
                world = (c2w @ cam.T).T[:, :3].astype(np.float32)

                if image is not None:
                    colors = image[vv, uu, :3].astype(np.float32)
                    if colors.max(initial=0.0) > 1.5:
                        colors = colors / 255.0
                    colors = np.clip(colors, 0.0, 1.0)
                else:
                    colors = np.ones_like(world, dtype=np.float32)

                xyz_parts.append(world)
                rgb_parts.append(colors)
                conf_parts.append(conf[vv, uu].astype(np.float32))
                frame_parts.append(np.full(world.shape[0], frame_idx, dtype=np.int32))
            except Exception:
                continue

        if not xyz_parts:
            return None

        xyz = np.concatenate(xyz_parts, axis=0)
        rgb = np.concatenate(rgb_parts, axis=0)
        conf = np.concatenate(conf_parts, axis=0)
        frame_idx = np.concatenate(frame_parts, axis=0)
        if xyz.shape[0] > max_points:
            pick = np.linspace(0, xyz.shape[0] - 1, max_points).round().astype(int)
            xyz = xyz[pick]
            rgb = rgb[pick]
            conf = conf[pick]
            frame_idx = frame_idx[pick]

        out = output_dir / "pointcloud.npz"
        np.savez_compressed(out, xyz=xyz, rgb=rgb, conf=conf, frame_idx=frame_idx)
        return out
