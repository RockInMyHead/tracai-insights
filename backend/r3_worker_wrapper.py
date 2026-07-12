"""
R³ Worker Wrapper — запускает R³ инференс через subprocess.
Используется GPU Worker (порт 8003) для запуска R³ (отдельное conda окружение r3).

Режимы:
  --live       выводит frame_processed после каждого нового .npz кадра (для real-time SSE)
  без --live   обычный режим: один complete event в конце

Output:
    JSON в stdout с результатами: camera poses, depth maps, point cloud
    В --live режиме: поток frame_processed событий, затем complete
"""
import os, sys, json, time, shutil, subprocess, tempfile, math, threading
from pathlib import Path

try:
    from r3_pointcloud import build_sampled_pointcloud
except ImportError:  # pragma: no cover - package-style startup
    from backend.r3_pointcloud import build_sampled_pointcloud

try:
    from r3_long_video import align_segment_poses, plan_segment_windows
except ImportError:  # pragma: no cover - package-style startup
    from backend.r3_long_video import align_segment_poses, plan_segment_windows

R3_DIR = Path("/home/artem/trackai/R3")
CONDA_RUN = ["/home/artem/miniconda3/bin/conda", "run", "-n", "r3", "--cwd", str(R3_DIR)]


def emit(event_type, data=None):
    """Print JSON event to stdout with flush."""
    payload = {"event": event_type}
    if data is not None:
        payload["data"] = data
    print(json.dumps(payload), flush=True)


CONF_THRESHOLDS = [0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def conf_stats(conf_values):
    import numpy as np

    if conf_values.size == 0:
        return {"percentiles": {}, "counts_by_threshold": {}}
    p = np.percentile(conf_values, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    keys = ["p0", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100"]
    return {
        "percentiles": {k: float(v) for k, v in zip(keys, p)},
        "counts_by_threshold": {str(t): int((conf_values >= t).sum()) for t in CONF_THRESHOLDS},
    }


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_stride: int = 5,
    max_frames: int = 0,
    segmented_long: bool = False,
    segment_min_duration: float = 600.0,
    long_target_fps: float = 5.0,
):
    """Extract frames from video using OpenCV.

    If max_frames is lower than the number of stride-selected frames, sample
    evenly across the full video. Long routes are otherwise truncated to the
    first N frames, which breaks R3 trajectory scale and coverage.
    """
    import cv2
    import numpy as np

    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    emit("video_info", {
        "frames": total_frames, "fps": fps,
        "width": width, "height": height, "duration": duration,
    })

    requested_frame_stride = max(1, int(frame_stride or 1))
    requested_max_frames = max(0, int(max_frames or 0))
    segmented_selection = bool(segmented_long and duration >= max(60.0, segment_min_duration))
    if segmented_selection and fps > 0 and long_target_fps > 0:
        frame_stride = max(requested_frame_stride, int(round(fps / long_target_fps)))
    else:
        frame_stride = requested_frame_stride
    # The max_frames limit becomes a per-segment GPU budget for long videos.
    # Short videos keep the existing uniform full-video cap.
    max_frames = 0 if segmented_selection else requested_max_frames

    candidate_indices = list(range(0, total_frames, frame_stride)) if total_frames > 0 else []
    if max_frames > 0 and len(candidate_indices) > max_frames:
        pick = np.linspace(0, len(candidate_indices) - 1, max_frames).round().astype(int)
        selected_indices = {candidate_indices[int(i)] for i in pick}
        sampling_mode = "uniform_full_video"
    else:
        selected_indices = set(candidate_indices)
        sampling_mode = "stride"

    emit("frame_sampling", {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "frame_stride": frame_stride,
        "requested_frame_stride": requested_frame_stride,
        "max_frames": max_frames,
        "requested_max_frames": requested_max_frames,
        "segmented_long": segmented_selection,
        "long_target_fps": long_target_fps if segmented_selection else None,
        "candidate_frames": len(candidate_indices),
        "selected_frames": len(selected_indices),
        "sampling_mode": sampling_mode,
    })

    selected_list = sorted(selected_indices)
    saved_source_indices = []
    saved_count = 0

    # For long MJPEG AVI files, sequential OpenCV decoding can stop early on
    # some FFmpeg builds. Seeking the selected source frames preserves full
    # video coverage when we intentionally sample across the whole route.
    use_seek_sampling = sampling_mode == "uniform_full_video"
    if use_seek_sampling:
        for source_idx in selected_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_idx))
            ret, frame = cap.read()
            if not ret:
                continue
            out_path = frames_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved_source_indices.append(int(source_idx))
            saved_count += 1
    else:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in selected_indices:
                out_path = frames_dir / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved_source_indices.append(int(frame_idx))
                saved_count += 1
                if max_frames > 0 and saved_count >= len(selected_indices):
                    break
            frame_idx += 1

    cap.release()

    frame_selection = {
        "video_path": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "frame_stride": frame_stride,
        "requested_frame_stride": requested_frame_stride,
        "max_frames": max_frames,
        "requested_max_frames": requested_max_frames,
        "segmented_long": segmented_selection,
        "long_target_fps": long_target_fps if segmented_selection else None,
        "candidate_frames": len(candidate_indices),
        "selected_frames_requested": len(selected_list),
        "saved_frames": saved_count,
        "sampling_mode": sampling_mode,
        "extraction_mode": "seek_selected" if use_seek_sampling else "sequential",
        "source_frame_min": min(saved_source_indices) if saved_source_indices else None,
        "source_frame_max": max(saved_source_indices) if saved_source_indices else None,
        "source_indices": saved_source_indices,
    }
    try:
        (Path(output_dir) / "frame_selection.json").write_text(
            json.dumps(frame_selection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    emit("frames_extracted", frame_selection)
    return str(frames_dir), saved_count


def _build_r3_infer_cmd(frames_dir: str, output_dir: str, ckpt_name: str = "r3.safetensors",
                        mode: str = "test", size: int = 392,
                        max_frames: int = 0):
    """Build infer.py command using the release presets from R3 demo.py."""
    mode = (mode or "test").lower()
    if mode in {"short"}:
        mode = "local"
    if mode in {"sampled", "sparse"}:
        mode = "strided"

    if mode in {"long", "strided"} and ckpt_name == "r3.safetensors":
        ckpt_name = "r3_long.safetensors"

    kv_cache_mode = "all" if mode in {"test", "strided"} else "dynamic"

    def env_enabled(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    def env_choice(name: str, default: str, choices: set[str]) -> str:
        value = (os.getenv(name) or default).strip().lower()
        return value if value in choices else default

    def env_number(name: str, default: str, minimum: float) -> str:
        raw = (os.getenv(name) or default).strip()
        try:
            value = max(minimum, float(raw))
        except ValueError:
            value = float(default)
        return str(int(value)) if value.is_integer() else str(value)

    # Greedy is useful for a fast preview.  The completed long-video path now
    # defaults to R3's global PGO reconstruction; it can be rolled back per
    # worker with R3_REL_POSE_METHOD=greedy.
    pose_method = env_choice(
        "R3_REL_POSE_METHOD",
        "pgo" if mode in {"long", "strided"} else "greedy",
        {"greedy", "pgo"},
    )
    ckpt_path = str(R3_DIR / "ckpt" / ckpt_name)

    cmd = CONDA_RUN + [
        "python3", str(R3_DIR / "infer.py"),
        "--seq_path", frames_dir,
        "--output_dir", output_dir,
        "--ckpt", ckpt_path,
        "--size", str(size),
        "--max_frames", str(max_frames),
        "--frame_stride", "1",
        "--online_kv_backend", "dense",
        "--online_kv_cache_mode", kv_cache_mode,
        "--keyframe_mode", "novelty",
        "--keyframe_novelty_threshold", env_number("R3_KEYFRAME_NOVELTY_THRESHOLD", "0.985", 0.0),
        "--keyframe_max_interval", env_number(
            "R3_KEYFRAME_MAX_INTERVAL", "15" if mode in {"long", "strided"} else "30", 1.0,
        ),
        "--keyframe_max_keyframes", env_number(
            "R3_KEYFRAME_MAX_KEYFRAMES", "160" if mode in {"long", "strided"} else "100", 16.0,
        ),
        "--rel_pose_reconstruction_method", pose_method,
    ]

    # Match R3 demo.py release presets: long/strided need confidence fallback
    # re-anchoring. Without it, long indoor videos often produce pose teleports.
    if mode in {"long", "strided"} and env_enabled("R3_ENABLE_FALLBACK", True):
        max_segment_frames = "300" if mode == "long" else "100"
        cmd += [
            "--online_fallback_enabled",
            "--fallback_drought_length", "3",
            "--fallback_drought_threshold", "0",
            "--fallback_drought_threshold_pct", "45.0",
            "--fallback_num_bridge_frames", "10",
            "--evict_low_conf_threshold", "0",
            "--fallback_ref_mode", "bridge",
            "--min_segment_frames", "16",
            "--max_segment_frames", max_segment_frames,
            "--fallback_replay_attention", "full",
        ]
        # Long segments now use their own PGO by default. It remains possible
        # to disable this on a constrained worker without changing code.
        if env_enabled("R3_DISABLE_SEGMENT_PGO", False):
            cmd.append("--disable_segment_pgo")

    # DA3 metric model is cached on the 3090 host; keep an env kill-switch for
    # emergency rollback without changing code.
    if mode in {"long", "strided"} and env_enabled("R3_ENABLE_METRIC_SCALE", True):
        cmd += [
            "--metric_scale_enabled",
            "--metric_bootstrap_frames", "5",
        ]

    return cmd, mode, ckpt_name


def run_r3_inference(frames_dir: str, output_dir: str, ckpt_name: str = "r3.safetensors",
                     mode: str = "test", size: int = 392,
                     max_frames: int = 0):
    """Run R³ inference via subprocess using conda r3 env."""
    cmd, resolved_mode, resolved_ckpt = _build_r3_infer_cmd(frames_dir, output_dir, ckpt_name, mode, size, max_frames)

    emit("r3_start", {"cmd": " ".join(str(c) for c in cmd[-12:])})
    emit("r3_preset", {"mode": resolved_mode, "ckpt": resolved_ckpt})

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=7200, cwd=str(R3_DIR), env=env
    )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise Exception(f"R³ inference failed (exit {result.returncode}): {detail[-6000:]}")

    params_path = Path(output_dir) / "run_params.json"
    if params_path.exists():
        try:
            params = json.loads(params_path.read_text())
            params["mode"] = resolved_mode
            params["wrapper_requested_mode"] = mode
            params["wrapper_resolved_ckpt"] = resolved_ckpt
            params_path.write_text(json.dumps(sanitize_json(params), indent=2), encoding="utf-8")
        except Exception as exc:
            emit("warning", {"message": f"failed to update run_params mode: {exc}"})

    return output_dir


def _merge_segment_artifacts(
    segment_output: Path,
    combined_output: Path,
    global_indices: list[int],
    merged_poses: dict[int, object],
    merged_confidence,
) -> dict:
    """Align and copy one R3 segment into the combined output namespace."""
    import numpy as np

    resolved = find_r3_output_dir(segment_output)
    camera_files = sorted((resolved / "camera").glob("*.npz"))
    local_poses: dict[int, np.ndarray] = {}
    camera_by_local: dict[int, Path] = {}
    for order, camera_file in enumerate(camera_files):
        try:
            stem_index = int(camera_file.stem)
        except ValueError:
            stem_index = order
        local_index = stem_index if 0 <= stem_index < len(global_indices) else order
        if local_index >= len(global_indices):
            continue
        camera = np.load(str(camera_file))
        local_poses[local_index] = np.asarray(camera["pose"], dtype=np.float64)
        camera_by_local[local_index] = camera_file

    if not local_poses:
        raise RuntimeError(f"No camera poses produced for segment {segment_output.name}")
    aligned, scale, diagnostics = align_segment_poses(
        local_poses,
        global_indices,
        merged_poses,
    )

    for name in ("camera", "depth", "conf", "color"):
        (combined_output / name).mkdir(parents=True, exist_ok=True)

    segment_confidence = None
    confidence_path = resolved / "pose_conf.npy"
    if confidence_path.exists():
        try:
            segment_confidence = np.asarray(np.load(str(confidence_path)), dtype=np.float64).reshape(-1)
        except Exception:
            segment_confidence = None

    copied = 0
    overlap = 0
    for local_index, global_index in enumerate(global_indices):
        camera_file = camera_by_local.get(local_index)
        transformed_pose = aligned.get(global_index)
        if camera_file is None or transformed_pose is None:
            continue

        if segment_confidence is not None and local_index < len(segment_confidence):
            value = float(segment_confidence[local_index])
            if np.isfinite(value):
                current = merged_confidence[global_index]
                merged_confidence[global_index] = value if not np.isfinite(current) else max(current, value)

        if global_index in merged_poses:
            overlap += 1
            continue

        camera = np.load(str(camera_file))
        camera_payload = {key: camera[key] for key in camera.files}
        camera_payload["pose"] = transformed_pose.astype(np.float32)
        np.savez_compressed(
            combined_output / "camera" / f"{global_index:06d}.npz",
            **camera_payload,
        )
        merged_poses[global_index] = transformed_pose

        local_stem = camera_file.stem
        depth_path = resolved / "depth" / f"{local_stem}.npy"
        if depth_path.exists():
            depth = np.asarray(np.load(str(depth_path)), dtype=np.float32)
            np.save(combined_output / "depth" / f"{global_index:06d}.npy", depth * float(scale))
        conf_path = resolved / "conf" / f"{local_stem}.npy"
        if conf_path.exists():
            shutil.copy2(conf_path, combined_output / "conf" / f"{global_index:06d}.npy")
        color_path = resolved / "color" / f"{local_stem}.png"
        if color_path.exists():
            shutil.copy2(color_path, combined_output / "color" / f"{global_index:06d}.png")
        copied += 1

    return {
        **diagnostics,
        "input_poses": len(local_poses),
        "copied_new_poses": copied,
        "overlap_poses": overlap,
        "resolved_output": str(resolved),
    }


def run_r3_inference_segmented(
    frames_dir: str,
    output_dir: str,
    ckpt_name: str,
    mode: str,
    size: int,
    segment_frames: int,
    overlap_frames: int,
) -> str:
    """Run overlapping R3 blocks and stitch their c2w products with Sim(3)."""
    import numpy as np

    source_frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    windows = plan_segment_windows(len(source_frames), segment_frames, overlap_frames)
    if len(windows) <= 1:
        return run_r3_inference(frames_dir, output_dir, ckpt_name, mode, size, 0)

    combined_output = Path(output_dir)
    segments_root = combined_output / "segments"
    segments_root.mkdir(parents=True, exist_ok=True)
    merged_poses: dict[int, np.ndarray] = {}
    merged_confidence = np.full(len(source_frames), np.nan, dtype=np.float64)
    manifest: dict = {
        "enabled": True,
        "total_selected_frames": len(source_frames),
        "segment_frames": segment_frames,
        "overlap_frames": overlap_frames,
        "segments": [],
    }
    first_run_params: dict = {}
    keep_segments = (os.getenv("R3_KEEP_SEGMENT_OUTPUTS") or "false").lower() in {"1", "true", "yes", "on"}

    for window in windows:
        segment_dir = segments_root / f"segment_{window.index:03d}"
        segment_frames_dir = segment_dir / "frames"
        segment_output = segment_dir / "output"
        if segment_dir.exists():
            shutil.rmtree(segment_dir)
        segment_frames_dir.mkdir(parents=True, exist_ok=True)
        segment_output.mkdir(parents=True, exist_ok=True)

        global_indices = list(range(window.start, window.end))
        for local_index, global_index in enumerate(global_indices):
            source = source_frames[global_index]
            target = segment_frames_dir / f"frame_{local_index:06d}.jpg"
            try:
                target.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, target)

        cmd, resolved_mode, resolved_ckpt = _build_r3_infer_cmd(
            str(segment_frames_dir),
            str(segment_output),
            ckpt_name,
            mode,
            size,
            0,
        )
        emit("r3_segment_start", {
            "segment": window.index + 1,
            "segments_total": len(windows),
            "selected_start": window.start,
            "selected_end": window.end - 1,
            "frames": window.frame_count,
        })
        started = time.time()
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=str(R3_DIR),
            env=env,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"R³ segment {window.index + 1}/{len(windows)} failed: {detail[-6000:]}"
            )

        resolved_output = find_r3_output_dir(segment_output)
        params_path = resolved_output / "run_params.json"
        if not first_run_params and params_path.exists():
            try:
                first_run_params = json.loads(params_path.read_text())
            except Exception:
                first_run_params = {}
        previously_merged = set(merged_poses)
        stitch = _merge_segment_artifacts(
            segment_output,
            combined_output,
            global_indices,
            merged_poses,
            merged_confidence,
        )
        new_global_indices = sorted(set(merged_poses) - previously_merged)
        for batch_start in range(0, len(new_global_indices), 100):
            batch_indices = new_global_indices[batch_start:batch_start + 100]
            emit("frame_processed", {
                "num_processed": len(merged_poses),
                "num_total": len(source_frames),
                "new_trajectory_points": [
                    [float(value) for value in merged_poses[index][:3, 3]]
                    for index in batch_indices
                ],
            })
        item = {
            "index": window.index,
            "selected_start": window.start,
            "selected_end": window.end - 1,
            "frames": window.frame_count,
            "inference_seconds": round(time.time() - started, 2),
            "mode": resolved_mode,
            "checkpoint": resolved_ckpt,
            "stitch": stitch,
        }
        manifest["segments"].append(item)
        emit("r3_segment_complete", item)

        if not keep_segments:
            shutil.rmtree(segment_dir, ignore_errors=True)

    if np.isfinite(merged_confidence).any():
        np.save(combined_output / "pose_conf.npy", merged_confidence.astype(np.float32))
    manifest["merged_poses"] = len(merged_poses)
    (combined_output / "segment_manifest.json").write_text(
        json.dumps(sanitize_json(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    first_run_params.update({
        "mode": mode,
        "wrapper_requested_mode": mode,
        "wrapper_resolved_ckpt": ckpt_name,
        "segmented_long": True,
        "segment_count": len(windows),
        "segment_frames": segment_frames,
        "segment_overlap_frames": overlap_frames,
        "max_frames": segment_frames,
        "inference_time_s": round(sum(item["inference_seconds"] for item in manifest["segments"]), 2),
        "rel_pose_reconstruction_method": os.getenv("R3_REL_POSE_METHOD", "pgo"),
    })
    (combined_output / "run_params.json").write_text(
        json.dumps(sanitize_json(first_run_params), indent=2),
        encoding="utf-8",
    )
    emit("r3_segmented_complete", {
        "segments": len(windows),
        "merged_poses": len(merged_poses),
    })
    return output_dir


def run_r3_inference_live(frames_dir: str, output_dir: str, camera_dir: Path,
                          ckpt_name: str = "r3.safetensors",
                          mode: str = "test", size: int = 392,
                          max_frames: int = 0):
    """Run R³ inference and emit frame_processed events as .npz files appear."""
    import numpy as np

    cmd, resolved_mode, resolved_ckpt = _build_r3_infer_cmd(frames_dir, output_dir, ckpt_name, mode, size, max_frames)

    emit("r3_start", {"cmd": " ".join(str(c) for c in cmd[-12:])})
    emit("r3_preset", {"mode": resolved_mode, "ckpt": resolved_ckpt})

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Start process
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(R3_DIR), env=env, text=True,
    )

    # Poll camera directory for new .npz files
    known_files = set()
    if camera_dir.exists():
        known_files = {f.name for f in camera_dir.glob("*.npz")}

    last_emit_count = 0

    while True:
        ret = proc.poll()
        # Check for new camera pose files
        if camera_dir.exists():
            current_files = sorted(camera_dir.glob("*.npz"))
            new_poses = []
            for cf in current_files:
                if cf.name not in known_files:
                    try:
                        data = np.load(str(cf))
                        pose = data["pose"].tolist()
                        intrinsics = data["intrinsics"].tolist() if "intrinsics" in data else None
                        new_poses.append({
                            "frame": int(cf.stem),
                            "pose": pose,
                            "intrinsics": intrinsics,
                        })
                        known_files.add(cf.name)
                    except Exception:
                        pass  # file still being written

            if new_poses:
                # Convert pose to trajectory point
                traj_points = []
                for p in new_poses:
                    po = p["pose"]
                    if po and len(po) >= 3 and len(po[0]) >= 4:
                        traj_points.append([po[0][3], po[1][3], po[2][3]])

                emit("frame_processed", {
                    "num_processed": len(current_files),
                    "num_total": "?",
                    "new_poses": sanitize_json(new_poses),
                    "new_trajectory_points": traj_points,
                })
                last_emit_count = len(current_files)

        if ret is not None:
            break
        time.sleep(1.0)

    # Process finished — check return code
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        detail = (stderr or stdout or "").strip()
        raise Exception(f"R³ inference failed (exit {proc.returncode}): {detail[-6000:]}")

    # Collect any remaining poses
    if camera_dir.exists():
        remaining = sorted(camera_dir.glob("*.npz"))
        if len(remaining) > last_emit_count and last_emit_count > 0:
            # Emit remaining
            remaining_poses = []
            for cf in remaining[last_emit_count:]:
                try:
                    data = np.load(str(cf))
                    rem_pose = data["pose"].tolist()
                    rem_intr = data["intrinsics"].tolist() if "intrinsics" in data else None
                    remaining_poses.append({
                        "frame": int(cf.stem),
                        "pose": rem_pose,
                        "intrinsics": rem_intr,
                    })
                except Exception:
                    pass
            if remaining_poses:
                emit("frame_processed", {
                    "num_processed": len(remaining),
                    "new_poses": sanitize_json(remaining_poses),
                })

    return output_dir


def find_r3_output_dir(output_path: Path) -> Path:
    """Find the R³ output directory."""
    if (output_path / "run_params.json").exists():
        return output_path
    subdirs = sorted([
        d for d in output_path.iterdir()
        if d.is_dir() and d.name != "frames"
    ], key=lambda p: p.stat().st_mtime)
    for sd in reversed(subdirs):
        if (sd / "run_params.json").exists():
            return sd
    return output_path


def sanitize_json(obj):
    """Convert NaN/Infinity values to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# ─── NEW: Depth back-projection → point cloud ──────────────────────────

def backproject_depth_pointcloud(
    output_dir: Path,
    stride: int = 4,
    max_points: int = 100000,
    min_conf: float = 1.0,
) -> list:
    """Read depth maps and camera poses, back-project to 3D world points.

    Returns a list of [x, y, z, r, g, b, conf] points (downsampled to max_points).
    """
    import numpy as np
    import cv2

    r3_output = find_r3_output_dir(output_dir)
    camera_dir = r3_output / "camera"
    depth_dir = r3_output / "depth"
    conf_dir = r3_output / "conf"
    frames_dir = r3_output / "frames"
    color_dir = r3_output / "color"

    if not camera_dir.exists() or not depth_dir.exists():
        emit("pointcloud_status", {"error": "camera or depth dir missing"})
        return []

    camera_files = sorted(camera_dir.glob("*.npz"))
    emit("pointcloud_status", {"frames": len(camera_files), "status": "reading"})

    all_points = []
    frames_used = 0

    for cf in camera_files:
        try:
            # Load camera pose + intrinsics
            cam_data = np.load(str(cf))
            pose = cam_data["pose"]  # 4x4 c2w
            if "intrinsics" in cam_data:
                intrinsics = cam_data["intrinsics"]
            else:
                intrinsics = None

            # Load depth map
            depth_path = depth_dir / f"{cf.stem}.npy"
            if not depth_path.exists():
                continue
            depth = np.load(str(depth_path))
            H, W = depth.shape

            rgb = None
            frame_path = color_dir / f"{cf.stem}.png"
            if not frame_path.exists():
                frame_path = frames_dir / f"frame_{int(cf.stem):06d}.jpg"
            if frame_path.exists():
                bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (H, W):
                        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

            # Load confidence (if available)
            conf = None
            conf_path = conf_dir / f"{cf.stem}.npy"
            if conf_path.exists():
                conf = np.load(str(conf_path))

            # Default intrinsics if missing
            if intrinsics is None:
                fx = fy = float(max(H, W)) * 1.2
                cx = W / 2.0
                cy = H / 2.0
            else:
                fx = float(intrinsics[0, 0])
                fy = float(intrinsics[1, 1])
                cx = float(intrinsics[0, 2])
                cy = float(intrinsics[1, 2])

            # 4x4 c2w matrix
            if pose.shape == (3, 4):
                pose_mat = np.eye(4)
                pose_mat[:3, :] = pose
            else:
                pose_mat = pose

            # Pixel grid with stride
            ys, xs = np.meshgrid(
                np.arange(0, H, stride),
                np.arange(0, W, stride),
                indexing='ij',
            )
            ys = ys.ravel()
            xs = xs.ravel()
            depth_vals = depth[ys, xs]

            # Filter valid depths
            valid = np.isfinite(depth_vals) & (depth_vals > 0)

            conf_vals = None
            if conf is not None:
                conf_vals = conf[ys, xs]
                valid = valid & np.isfinite(conf_vals) & (conf_vals > min_conf)

            if not valid.any():
                continue

            xs_v = xs[valid]
            ys_v = ys[valid]
            z_vals = depth_vals[valid]
            conf_out = conf_vals[valid].astype(np.float32) if conf_vals is not None else np.full(z_vals.shape, 2.0, dtype=np.float32)
            frame_out = np.full(z_vals.shape, int(cf.stem), dtype=np.float32)

            # Back-project to camera space
            x_cam = (xs_v - cx) * z_vals / fx
            y_cam = (ys_v - cy) * z_vals / fy

            # Camera space → world space (c2w)
            ones = np.ones_like(x_cam)
            cam_pts = np.stack([x_cam, y_cam, z_vals, ones], axis=-1)  # N x 4
            world_pts = (pose_mat @ cam_pts.T).T  # N x 4
            world_pts = world_pts[:, :3]  # N x 3

            if rgb is not None:
                rgb_vals = rgb[ys_v, xs_v].astype(np.float32) / 255.0
            else:
                rgb_vals = np.full((world_pts.shape[0], 3), 0.75, dtype=np.float32)

            all_points.append(np.concatenate([world_pts, rgb_vals, conf_out[:, None], frame_out[:, None]], axis=1))
            frames_used += 1

        except Exception as e:
            emit("pointcloud_status", {"frame": cf.stem, "error": str(e)})
            continue

    if not all_points:
        emit("pointcloud_status", {"error": "no points generated"})
        return []

    combined = np.concatenate(all_points, axis=0)
    emit("pointcloud_status", {
        "frames_used": frames_used,
        "total_points": int(combined.shape[0]),
    })

    # Filter finite attributes
    xyz = combined[:, :3]
    rgb = combined[:, 3:6]
    conf = combined[:, 6] if combined.shape[1] > 6 else np.ones((combined.shape[0],), dtype=np.float32) * 2.0
    finite = np.isfinite(xyz).all(axis=1)
    valid_attrs = np.isfinite(rgb).all(axis=1) & np.isfinite(conf)
    combined = combined[finite & valid_attrs]

    # Save full debug cloud with frame_idx for diagnostics
    debug_npz = r3_output / "pointcloud_full_debug.npz"
    np.savez_compressed(str(debug_npz), points=combined)
    diag_dir = r3_output / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    stats = conf_stats(combined[:, 6] if combined.shape[1] > 6 else np.array([], dtype=np.float32))
    stats["total_points_before_filter"] = int(combined.shape[0])
    (diag_dir / "conf_stats.json").write_text(json.dumps(stats, indent=2))

    # Downsample production cloud
    if combined.shape[0] > max_points:
        idx = np.random.RandomState(42).choice(combined.shape[0], max_points, replace=False)
        combined = combined[idx]

    # Save as .npz for later retrieval
    output_npz = r3_output / "pointcloud.npz"
    np.savez_compressed(str(output_npz), points=combined[:, :7])
    emit("pointcloud_status", {"saved": str(output_npz), "num_points": int(combined.shape[0])})

    return combined[:, :7].tolist()


# ─── NEW: Collect results with point cloud ─────────────────────────────

def collect_results(output_dir: str, export_pointcloud: bool = True):
    """Collect all R³ results into a JSON-serializable dict, with optional point cloud."""
    import numpy as np

    output_path = Path(output_dir)
    r3_output = find_r3_output_dir(output_path)

    run_params = {}
    params_path = r3_output / "run_params.json"
    if params_path.exists():
        with open(params_path) as f:
            run_params = json.load(f)

    frame_selection = {}
    frame_selection_path = r3_output / "frame_selection.json"
    if frame_selection_path.exists():
        try:
            with open(frame_selection_path) as f:
                frame_selection = json.load(f)
        except Exception:
            frame_selection = {}

    camera_dir = r3_output / "camera"
    poses = []
    if camera_dir.exists():
        camera_files = sorted(camera_dir.glob("*.npz"))
        for cf in camera_files:
            data = np.load(str(cf))
            poses.append({
                "frame": int(cf.stem),
                "pose": data["pose"].tolist(),
                "intrinsics": data["intrinsics"].tolist() if "intrinsics" in data else None,
            })

    pose_conf = None
    pose_conf_path = r3_output / "pose_conf.npy"
    if pose_conf_path.exists():
        pose_conf = np.load(str(pose_conf_path)).tolist()

    result = {
        "success": True,
        "num_frames": len(poses),
        "run_params": {
            "config_name": run_params.get("config_name"),
            "ckpt": run_params.get("ckpt"),
            "wrapper_mode": run_params.get("wrapper_mode"),
            "mode": run_params.get("mode"),
            "inference_time_s": run_params.get("inference_time_s"),
            "segmented_long": run_params.get("segmented_long", False),
            "segment_count": run_params.get("segment_count"),
            "segment_frames": run_params.get("segment_frames"),
            "segment_overlap_frames": run_params.get("segment_overlap_frames"),
        },
        "camera_poses": sanitize_json(poses),
        "num_poses_total": len(poses),
        "pose_confidence": sanitize_json(pose_conf if pose_conf else None),
        "output_dir": str(r3_output),
        "frame_selection": sanitize_json(frame_selection),
    }

    # Generate point cloud from depth maps
    if export_pointcloud and (r3_output / "depth").exists():
        try:
            cloud_result = build_sampled_pointcloud(
                r3_output,
                stride=max(1, int(os.getenv("R3_POINTCLOUD_STRIDE", "4"))),
                max_points=max(1_000, int(os.getenv("R3_POINTCLOUD_MAX_POINTS", "200000"))),
            )
            pcloud = cloud_result.get("points") or []
            if pcloud and len(pcloud) > 0:
                # Send only a small sample in the complete event (full cloud available via API)
                sample = pcloud[:2000] if len(pcloud) > 2000 else pcloud
                result["pointcloud_sample"] = sample
                result["pointcloud_count"] = len(pcloud)
        except Exception as e:
            emit("pointcloud_status", {"error": str(e)})

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="R³ Worker Wrapper")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", default="/home/artem/trackai/gpu_worker_data/r3_output")
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--ckpt", default="r3.safetensors")
    parser.add_argument("--size", type=int, default=392)
    parser.add_argument("--mode", default="test")
    parser.add_argument("--live", action="store_true", help="Emit progress events as frames are processed")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    camera_dir = output_path / "camera"
    camera_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    emit("start", {"video": args.video_path, "live": args.live})

    segmented_enabled = (
        args.mode.lower() in {"long", "strided", "sampled", "sparse"}
        and (os.getenv("R3_SEGMENTED_LONG") or "true").lower() in {"1", "true", "yes", "on"}
    )
    segment_min_duration = float(os.getenv("R3_SEGMENT_MIN_DURATION_SECONDS", "600"))
    long_target_fps = float(os.getenv("R3_LONG_TARGET_FPS", "5"))

    # Step 1: Extract a dense-enough long-video sequence. max_frames remains
    # the per-segment budget rather than thinning the entire recording.
    frames_dir, num_frames = extract_frames(
        args.video_path,
        args.output_dir,
        args.frame_stride,
        args.max_frames,
        segmented_long=segmented_enabled,
        segment_min_duration=segment_min_duration,
        long_target_fps=long_target_fps,
    )
    frame_selection = {}
    selection_path = output_path / "frame_selection.json"
    if selection_path.exists():
        try:
            frame_selection = json.loads(selection_path.read_text())
        except Exception:
            frame_selection = {}
    segment_frames = max(256, int(os.getenv("R3_SEGMENT_FRAMES", str(args.max_frames or 1500))))
    overlap_frames = max(16, int(os.getenv("R3_SEGMENT_OVERLAP_FRAMES", "90")))
    use_segmented = bool(frame_selection.get("segmented_long") and num_frames > segment_frames)

    # Step 2: Run R³ inference (live or standard)
    if use_segmented:
        run_r3_inference_segmented(
            frames_dir,
            args.output_dir,
            args.ckpt,
            args.mode,
            args.size,
            segment_frames,
            overlap_frames,
        )
    elif args.live:
        run_r3_inference_live(frames_dir, args.output_dir, camera_dir,
                              args.ckpt, args.mode, args.size, args.max_frames)
    else:
        run_r3_inference(frames_dir, args.output_dir, args.ckpt, args.mode, args.size, args.max_frames)

    # Step 3: Return trajectory immediately.  Production point-cloud export
    # is scheduled by GPU Worker after this wrapper exits, so the user no
    # longer waits for CPU back-projection and NPZ compression.
    inline_pointcloud = (os.getenv("R3_INLINE_POINTCLOUD") or "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    result = collect_results(args.output_dir, export_pointcloud=inline_pointcloud)
    result["pointcloud_deferred"] = bool(
        not inline_pointcloud and (find_r3_output_dir(output_path) / "depth").exists()
    )
    result["total_time_s"] = round(time.time() - start, 1)
    emit("complete", {"result": result})


if __name__ == "__main__":
    main()
