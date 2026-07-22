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

try:
    from r3_trajectory import summarize_fallback_edges
except ImportError:  # pragma: no cover - package-style startup
    from backend.r3_trajectory import summarize_fallback_edges

try:
    from r3_pose_graph import (
        R3_ABSOLUTE_POSE_SPACE,
        R3_CONFIDENCE_SEMANTICS,
        R3_POSE_ENCODING,
        R3_POSE_GRAPH_SCHEMA_VERSION,
        R3_RELATIVE_TRANSFORM_CONVENTION,
        load_pose_graph_summary,
    )
except ImportError:  # pragma: no cover - package-style startup
    from backend.r3_pose_graph import (
        R3_ABSOLUTE_POSE_SPACE,
        R3_CONFIDENCE_SEMANTICS,
        R3_POSE_ENCODING,
        R3_POSE_GRAPH_SCHEMA_VERSION,
        R3_RELATIVE_TRANSFORM_CONVENTION,
        load_pose_graph_summary,
    )

try:
    from r3_pose_graph_optimizer import (
        load_pose_graph_candidate_c2w,
        load_pose_graph_candidate_summary,
        run_pose_graph_shadow,
    )
except ImportError:  # pragma: no cover - package-style startup
    from backend.r3_pose_graph_optimizer import (
        load_pose_graph_candidate_c2w,
        load_pose_graph_candidate_summary,
        run_pose_graph_shadow,
    )

try:
    from r3_scale_aware import (
        build_scale_aware_candidate,
        estimate_floor_height_observations,
        load_scale_aware_candidate_summary,
        save_scale_aware_candidate,
    )
except ImportError:  # pragma: no cover - package-style startup
    from backend.r3_scale_aware import (
        build_scale_aware_candidate,
        estimate_floor_height_observations,
        load_scale_aware_candidate_summary,
        save_scale_aware_candidate,
    )

R3_DIR = Path("/home/artem/trackai/R3")
CONDA_RUN = ["/home/artem/miniconda3/bin/conda", "run", "-n", "r3", "--cwd", str(R3_DIR)]

R3_POSE_GRAPH_EXPORT_MARKER = "# TRACKAI_R3_POSE_GRAPH_EXPORT_V1"
R3_POSE_GRAPH_EXPORT_ANCHOR = '''                with open(os.path.join(output_dir, "pose_edge_log.json"), "w") as f:
                    json.dump(edge_records, f)
'''
R3_POSE_GRAPH_EXPORT_INSERTION = '''                # TRACKAI_R3_POSE_GRAPH_EXPORT_V1
                pose_graph_capacity = len(edges)
                pose_graph_edge_sequence = np.empty(pose_graph_capacity, dtype=np.int64)
                pose_graph_frame_i = np.empty(pose_graph_capacity, dtype=np.int32)
                pose_graph_frame_j = np.empty(pose_graph_capacity, dtype=np.int32)
                pose_graph_model_frame_i = np.empty(pose_graph_capacity, dtype=np.int64)
                pose_graph_model_frame_j = np.empty(pose_graph_capacity, dtype=np.int64)
                pose_graph_rel_pose = np.empty((pose_graph_capacity, 9), dtype=np.float32)
                pose_graph_confidence = np.empty(pose_graph_capacity, dtype=np.float32)
                pose_graph_confidence_t = np.full(pose_graph_capacity, np.nan, dtype=np.float32)
                pose_graph_confidence_r = np.full(pose_graph_capacity, np.nan, dtype=np.float32)
                pose_graph_edge_type = np.full(pose_graph_capacity, 255, dtype=np.uint8)
                pose_graph_count = 0
                for edge_sequence, edge in enumerate(edges):
                    frame_i = int(edge.frame_i)
                    frame_j = int(edge.frame_j)
                    if frame_i not in frame_id_to_output_idx or frame_j not in frame_id_to_output_idx:
                        continue
                    rel_pose = getattr(edge, "rel_pose_enc", None)
                    if rel_pose is None:
                        continue
                    rel_pose_values = rel_pose.detach().cpu().float().numpy().reshape(-1)
                    if rel_pose_values.size != 9 or not np.isfinite(rel_pose_values).all():
                        continue
                    confidence_t = getattr(edge, "confidence_t", None)
                    confidence_r = getattr(edge, "confidence_r", None)
                    record_index = pose_graph_count
                    pose_graph_edge_sequence[record_index] = edge_sequence
                    pose_graph_frame_i[record_index] = frame_id_to_output_idx[frame_i]
                    pose_graph_frame_j[record_index] = frame_id_to_output_idx[frame_j]
                    pose_graph_model_frame_i[record_index] = frame_i
                    pose_graph_model_frame_j[record_index] = frame_j
                    pose_graph_rel_pose[record_index] = rel_pose_values
                    pose_graph_confidence[record_index] = float(edge.confidence)
                    if confidence_t is not None:
                        pose_graph_confidence_t[record_index] = float(confidence_t)
                    if confidence_r is not None:
                        pose_graph_confidence_r[record_index] = float(confidence_r)
                    pose_graph_edge_type[record_index] = {
                        "normal": 0,
                        "bridge": 1,
                        "anchor": 2,
                    }.get(edge.edge_type, 255)
                    pose_graph_count += 1
                if pose_graph_count:
                    np.savez_compressed(
                        os.path.join(output_dir, "pose_graph_edges.npz"),
                        schema_version=np.asarray([1], dtype=np.int32),
                        pose_encoding=np.asarray("txyz_qxyzw_fovxy"),
                        transform_convention=np.asarray("target_hmat=relative_hmat@reference_hmat"),
                        frame_index_space=np.asarray("exported_camera_index"),
                        absolute_pose_space=np.asarray("world_to_camera"),
                        confidence_semantics=np.asarray("softplus_positive_weight_not_covariance"),
                        edge_type_names=np.asarray(["normal", "bridge", "anchor", "unknown"]),
                        edge_sequence=pose_graph_edge_sequence[:pose_graph_count],
                        frame_i=pose_graph_frame_i[:pose_graph_count],
                        frame_j=pose_graph_frame_j[:pose_graph_count],
                        model_frame_i=pose_graph_model_frame_i[:pose_graph_count],
                        model_frame_j=pose_graph_model_frame_j[:pose_graph_count],
                        rel_pose_enc=pose_graph_rel_pose[:pose_graph_count],
                        confidence=pose_graph_confidence[:pose_graph_count],
                        confidence_t=pose_graph_confidence_t[:pose_graph_count],
                        confidence_r=pose_graph_confidence_r[:pose_graph_count],
                        edge_type=pose_graph_edge_type[:pose_graph_count],
                    )
'''


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


def _patch_r3_infer_source(source: str) -> tuple[str, dict]:
    """Inject a versioned full-edge sidecar export into supported R3 infer.py."""
    if R3_POSE_GRAPH_EXPORT_MARKER in source:
        return source, {"status": "already_available", "changed": False}
    if R3_POSE_GRAPH_EXPORT_ANCHOR not in source:
        return source, {
            "status": "unsupported_infer_source",
            "changed": False,
            "reason": "pose_edge_log export anchor not found",
        }
    patched = source.replace(
        R3_POSE_GRAPH_EXPORT_ANCHOR,
        R3_POSE_GRAPH_EXPORT_ANCHOR + R3_POSE_GRAPH_EXPORT_INSERTION,
        1,
    )
    try:
        compile(patched, "infer.py", "exec")
    except SyntaxError as exc:
        return source, {
            "status": "patch_compile_failed",
            "changed": False,
            "reason": f"{exc.msg} at line {exc.lineno}",
        }
    return patched, {"status": "patched", "changed": True}


def _ensure_r3_pose_graph_export(r3_dir: str | Path = R3_DIR) -> dict:
    """Patch the external R3 exporter atomically, or report why it was skipped."""
    enabled = (os.getenv("R3_EXPORT_POSE_GRAPH_EDGES") or "true").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return {"status": "disabled", "changed": False}
    infer_path = Path(r3_dir) / "infer.py"
    try:
        source = infer_path.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "status": "infer_read_failed",
            "changed": False,
            "path": str(infer_path),
            "reason": f"{type(exc).__name__}: {exc}",
        }
    patched, diagnostics = _patch_r3_infer_source(source)
    diagnostics = {"path": str(infer_path), **diagnostics}
    if not diagnostics["changed"]:
        return diagnostics
    temporary_path = infer_path.with_name(
        f".{infer_path.name}.trackai.{os.getpid()}.tmp"
    )
    try:
        temporary_path.write_text(patched, encoding="utf-8")
        os.replace(temporary_path, infer_path)
    except Exception as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "status": "infer_write_failed",
            "changed": False,
            "path": str(infer_path),
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return diagnostics


def _prepare_r3_pose_graph_export() -> dict:
    diagnostics = _ensure_r3_pose_graph_export(R3_DIR)
    emit("r3_pose_graph_export", diagnostics)
    return diagnostics


def _probe_video_frame_timestamps(video_path: str) -> tuple[list[float | None], dict]:
    """Read presentation timestamps in decode order without decoding pixels.

    Source-frame index divided by nominal FPS is wrong for variable-frame-rate
    recordings and after dropped frames.  R3 still consumes numbered JPEGs,
    but retaining their original PTS lets downstream motion constraints use
    the real time delta instead of silently assuming a constant cadence.
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return [], {
            "available": False,
            "source": "ffprobe_best_effort_timestamp_time",
            "error": type(exc).__name__,
        }
    if result.returncode != 0:
        return [], {
            "available": False,
            "source": "ffprobe_best_effort_timestamp_time",
            "error": (result.stderr or "ffprobe_failed").strip()[-500:],
        }
    try:
        frames = json.loads(result.stdout).get("frames", [])
    except (AttributeError, json.JSONDecodeError) as exc:
        return [], {
            "available": False,
            "source": "ffprobe_best_effort_timestamp_time",
            "error": type(exc).__name__,
        }

    timestamps: list[float | None] = []
    finite_count = 0
    for frame in frames if isinstance(frames, list) else []:
        raw = frame.get("best_effort_timestamp_time") if isinstance(frame, dict) else None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            timestamps.append(value)
            finite_count += 1
        else:
            timestamps.append(None)
    return timestamps, {
        "available": finite_count > 0,
        "source": "ffprobe_best_effort_timestamp_time",
        "decoded_frames": len(timestamps),
        "finite_timestamps": finite_count,
    }


def resolve_extraction_stride(
    source_fps: float,
    requested_frame_stride: int,
    *,
    long_video: bool,
    long_target_fps: float,
) -> int:
    requested = max(1, int(requested_frame_stride or 1))
    if long_video and source_fps > 0 and long_target_fps > 0:
        return max(1, int(round(source_fps / long_target_fps)))
    return requested


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_stride: int = 5,
    max_frames: int = 0,
    continuous_long: bool = False,
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
    long_video_selection = bool(
        (continuous_long or segmented_long)
        and duration >= max(60.0, segment_min_duration)
    )
    segmented_selection = bool(segmented_long and long_video_selection)
    continuous_selection = bool(continuous_long and long_video_selection and not segmented_selection)
    if long_video_selection and fps > 0 and long_target_fps > 0:
        # Long-video quality is governed by the explicit target FPS. The API's
        # historical frame_stride=5 default would otherwise cap a 30 FPS video
        # at 6 FPS even when production asks for 8 FPS, dropping useful turn
        # overlap. Keep the request in diagnostics but do not let that stale
        # short-video throttle reduce long-route temporal coverage.
        frame_stride = resolve_extraction_stride(
            fps,
            requested_frame_stride,
            long_video=True,
            long_target_fps=long_target_fps,
        )
    else:
        frame_stride = requested_frame_stride
    # R3's native long mode has bounded memory and needs one continuous frame
    # stream to retain its keyframe bank and reconnect loops.  Keep max_frames
    # only for short videos; otherwise it silently lowers effective FPS as the
    # video becomes longer.
    max_frames = 0 if long_video_selection else requested_max_frames

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
        "long_video_sampling": long_video_selection,
        "continuous_long": continuous_selection,
        "segmented_long": segmented_selection,
        "long_target_fps": long_target_fps if long_video_selection else None,
        "candidate_frames": len(candidate_indices),
        "selected_frames": len(selected_indices),
        "sampling_mode": sampling_mode,
    })

    selected_list = sorted(selected_indices)
    selected_order = {source_idx: order for order, source_idx in enumerate(selected_list)}
    saved_records: dict[int, float | None] = {}
    actual_source_by_target: dict[int, int] = {}
    recovered_source_indices: list[int] = []
    probed_timestamps, timestamp_probe = _probe_video_frame_timestamps(video_path)

    def source_timestamp(source_idx: int) -> float | None:
        if 0 <= source_idx < len(probed_timestamps):
            value = probed_timestamps[source_idx]
            if value is not None and math.isfinite(value):
                return float(value)
        if fps > 0:
            return float(source_idx / fps)
        return None

    def save_selected_frame(
        target_source_idx: int,
        frame,
        *,
        actual_source_idx: int | None = None,
    ) -> None:
        order = selected_order[target_source_idx]
        out_path = frames_dir / f"frame_{order:06d}.jpg"
        if not cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            return
        actual = int(target_source_idx if actual_source_idx is None else actual_source_idx)
        saved_records[int(target_source_idx)] = source_timestamp(actual)
        actual_source_by_target[int(target_source_idx)] = actual

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
            save_selected_frame(int(source_idx), frame)
    else:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in selected_indices:
                save_selected_frame(int(frame_idx), frame)
                if max_frames > 0 and len(saved_records) >= len(selected_indices):
                    break
            frame_idx += 1

    cap.release()

    # A long AVI can report its complete frame count and still stop sequential
    # decoding hundreds of seconds before EOF. Recover only the missing
    # selected frames by index, keeping the output numbering in source order.
    missing_source_indices = [
        source_idx for source_idx in selected_list if source_idx not in saved_records
    ]
    if missing_source_indices and long_video_selection:
        recovery_cap = cv2.VideoCapture(video_path)
        if recovery_cap.isOpened():
            for source_idx in missing_source_indices:
                lookback = max(8, int(round(fps * 2.0))) if fps > 0 else 60
                lower_bound = max(0, source_idx - lookback)
                ret = False
                frame = None
                actual_source_idx = source_idx
                for candidate in range(source_idx, lower_bound - 1, -1):
                    recovery_cap.set(cv2.CAP_PROP_POS_FRAMES, int(candidate))
                    ret, frame = recovery_cap.read()
                    if ret:
                        actual_source_idx = candidate
                        break
                if not ret:
                    continue
                save_selected_frame(
                    int(source_idx),
                    frame,
                    actual_source_idx=int(actual_source_idx),
                )
                if source_idx in saved_records:
                    recovered_source_indices.append(int(source_idx))
        recovery_cap.release()

    saved_target_indices = [
        source_idx for source_idx in selected_list if source_idx in saved_records
    ]
    saved_source_indices = [actual_source_by_target[source_idx] for source_idx in saved_target_indices]
    saved_source_timestamps = [saved_records[source_idx] for source_idx in saved_target_indices]
    saved_count = len(saved_target_indices)

    exact_timestamp_count = sum(
        1
        for source_idx in saved_source_indices
        if 0 <= source_idx < len(probed_timestamps)
        and probed_timestamps[source_idx] is not None
        and math.isfinite(float(probed_timestamps[source_idx]))
    )
    if saved_count > 0 and exact_timestamp_count == saved_count:
        timestamp_source = "ffprobe_best_effort_timestamp_time"
    elif exact_timestamp_count > 0:
        timestamp_source = "mixed_ffprobe_and_nominal_fps"
    else:
        timestamp_source = "nominal_fps_fallback"

    frame_selection = {
        "video_path": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "frame_stride": frame_stride,
        "requested_frame_stride": requested_frame_stride,
        "max_frames": max_frames,
        "requested_max_frames": requested_max_frames,
        "long_video_sampling": long_video_selection,
        "continuous_long": continuous_selection,
        "segmented_long": segmented_selection,
        "long_target_fps": long_target_fps if long_video_selection else None,
        "candidate_frames": len(candidate_indices),
        "selected_frames_requested": len(selected_list),
        "saved_frames": saved_count,
        "sampling_mode": sampling_mode,
        "extraction_mode": (
            "seek_selected"
            if use_seek_sampling
            else "sequential_plus_seek_recovery"
            if recovered_source_indices
            else "sequential"
        ),
        "recovered_frames": len(recovered_source_indices),
        "missing_frames": len(selected_list) - saved_count,
        "source_frame_min": min(saved_source_indices) if saved_source_indices else None,
        "source_frame_max": max(saved_source_indices) if saved_source_indices else None,
        "requested_source_indices": selected_list,
        "source_indices": saved_source_indices,
        "source_timestamps_seconds": saved_source_timestamps,
        "timestamp_source": timestamp_source,
        "exact_timestamp_count": exact_timestamp_count,
        "fallback_timestamp_count": saved_count - exact_timestamp_count,
        "timestamp_probe": timestamp_probe,
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

    # Match the official R3 release presets by default.  Greedy reconstruction
    # plus disabled fallback-segment PGO is deliberate upstream behaviour;
    # forcing both PGO layers made physical corners much smoother and added a
    # long post-inference optimization phase on CPU.
    release_preset = env_enabled("R3_USE_RELEASE_PRESET", True)
    pose_method = (
        "greedy"
        if release_preset
        else env_choice("R3_REL_POSE_METHOD", "greedy", {"greedy", "pgo"})
    )
    keyframe_novelty = (
        "0.985"
        if release_preset
        else env_number("R3_KEYFRAME_NOVELTY_THRESHOLD", "0.985", 0.0)
    )
    keyframe_interval = (
        "30"
        if release_preset
        else env_number("R3_KEYFRAME_MAX_INTERVAL", "30", 1.0)
    )
    keyframe_count = (
        "100"
        if release_preset
        else env_number("R3_KEYFRAME_MAX_KEYFRAMES", "100", 16.0)
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
        "--keyframe_novelty_threshold", keyframe_novelty,
        "--keyframe_max_interval", keyframe_interval,
        "--keyframe_max_keyframes", keyframe_count,
        "--rel_pose_reconstruction_method", pose_method,
    ]

    # Match R3 demo.py release presets: long/strided need confidence fallback
    # re-anchoring. Without it, long indoor videos often produce pose teleports.
    if mode in {"long", "strided"} and env_enabled("R3_ENABLE_FALLBACK", True):
        max_segment_frames = "300" if mode == "long" else "100"
        bridge_baseline_ratio = (
            "0.35"
            if release_preset
            else env_number("R3_FALLBACK_MIN_BRIDGE_BASELINE_RATIO", "0.35", 0.0)
        )
        bridge_lookback = (
            "40"
            if release_preset
            else env_number("R3_FALLBACK_MAX_BRIDGE_LOOKBACK", "40", 10.0)
        )
        cmd += [
            "--online_fallback_enabled",
            "--fallback_drought_length", "3",
            "--fallback_drought_threshold", "0",
            "--fallback_drought_threshold_pct", "45.0",
            "--fallback_num_bridge_frames", "10",
            "--fallback_min_bridge_baseline_ratio", bridge_baseline_ratio,
            "--fallback_max_bridge_lookback", bridge_lookback,
            "--evict_low_conf_threshold", "0",
            "--fallback_ref_mode", "bridge",
            "--min_segment_frames", "16",
            "--max_segment_frames", max_segment_frames,
            "--fallback_replay_attention", "full",
        ]
        # The official demo explicitly disables fallback-segment PGO.  The
        # new opt-in variable also neutralizes stale deployments that still
        # contain R3_DISABLE_SEGMENT_PGO=false from the regressed version.
        if release_preset or not env_enabled("R3_ENABLE_SEGMENT_PGO", False):
            cmd.append("--disable_segment_pgo")

    # A floor-plan overlay needs continuity, not a fresh absolute metric guess
    # after every fallback.  Upstream accepts even a >2x metric re-anchor; on a
    # long indoor route that makes later corridors visibly expand/shrink.  The
    # bridge-depth fallback already maps every new segment into the previous
    # scale, so keep that continuity policy by default.  Metric re-anchoring is
    # still available as an explicit experiment with the new policy variable;
    # this intentionally neutralizes stale R3_ENABLE_METRIC_SCALE=true values.
    scale_policy = env_choice(
        "R3_SCALE_POLICY",
        "bridge_continuity",
        {"bridge_continuity", "metric_reanchor"},
    )
    if mode in {"long", "strided"} and scale_policy == "metric_reanchor":
        cmd += [
            "--metric_scale_enabled",
            "--metric_bootstrap_frames", "5",
        ]

    return cmd, mode, ckpt_name


def run_r3_inference(frames_dir: str, output_dir: str, ckpt_name: str = "r3.safetensors",
                     mode: str = "test", size: int = 392,
                     max_frames: int = 0):
    """Run R³ inference via subprocess using conda r3 env."""
    _prepare_r3_pose_graph_export()
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
            params["wrapper_input_size"] = int(size)
            params["wrapper_max_frames"] = int(max_frames)
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
    *,
    scale_prior: float | None = None,
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
        scale_prior=scale_prior,
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


def _read_segment_pose_graph(
    segment_output: Path,
    global_indices: list[int],
    scale: float,
) -> dict[str, object] | None:
    """Map one segment's measured relative edges into global frame indices."""
    import numpy as np

    path = segment_output / "pose_graph_edges.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            frame_i = np.asarray(payload["frame_i"], dtype=np.int64).reshape(-1)
            frame_j = np.asarray(payload["frame_j"], dtype=np.int64).reshape(-1)
            rel_pose = np.asarray(payload["rel_pose_enc"], dtype=np.float32)
            confidence = np.asarray(payload["confidence"], dtype=np.float32).reshape(-1)
            confidence_t = np.asarray(payload["confidence_t"], dtype=np.float32).reshape(-1)
            confidence_r = np.asarray(payload["confidence_r"], dtype=np.float32).reshape(-1)
            edge_type = np.asarray(payload["edge_type"], dtype=np.uint8).reshape(-1)
    except Exception:
        return None

    count = len(frame_i)
    if (
        rel_pose.shape != (count, 9)
        or any(len(values) != count for values in (frame_j, confidence, confidence_t, confidence_r, edge_type))
    ):
        return None
    valid = (
        (frame_i >= 0)
        & (frame_j >= 0)
        & (frame_i < len(global_indices))
        & (frame_j < len(global_indices))
        & (frame_i != frame_j)
        & np.isfinite(rel_pose).all(axis=1)
    )
    if not valid.any():
        return None
    frame_i = frame_i[valid]
    frame_j = frame_j[valid]
    scaled_rel_pose = rel_pose[valid].copy()
    # A world Sim(3) changes relative translation scale but its common world
    # rotation/translation cancels between the two camera frames.
    scaled_rel_pose[:, :3] *= float(scale)
    quaternion_norm = np.linalg.norm(scaled_rel_pose[:, 3:7], axis=1)
    usable_quaternion = np.isfinite(quaternion_norm) & (quaternion_norm > 1e-8)
    if not usable_quaternion.all():
        frame_i = frame_i[usable_quaternion]
        frame_j = frame_j[usable_quaternion]
        scaled_rel_pose = scaled_rel_pose[usable_quaternion]
        confidence = confidence[valid][usable_quaternion]
        confidence_t = confidence_t[valid][usable_quaternion]
        confidence_r = confidence_r[valid][usable_quaternion]
        edge_type = edge_type[valid][usable_quaternion]
        quaternion_norm = quaternion_norm[usable_quaternion]
    else:
        confidence = confidence[valid]
        confidence_t = confidence_t[valid]
        confidence_r = confidence_r[valid]
        edge_type = edge_type[valid]
    scaled_rel_pose[:, 3:7] /= quaternion_norm[:, None]
    index_array = np.asarray(global_indices, dtype=np.int32)
    return {
        "frame_i": index_array[frame_i],
        "frame_j": index_array[frame_j],
        "rel_pose_enc": scaled_rel_pose,
        "confidence": confidence,
        "confidence_t": confidence_t,
        "confidence_r": confidence_r,
        "edge_type": edge_type,
    }


def _save_merged_pose_graph(
    combined_output: Path,
    parts: list[dict[str, object]],
) -> int:
    """Persist a connected pose graph spanning every overlapping segment."""
    import numpy as np

    if not parts:
        return 0
    names = (
        "frame_i",
        "frame_j",
        "rel_pose_enc",
        "confidence",
        "confidence_t",
        "confidence_r",
        "edge_type",
    )
    merged = {
        name: np.concatenate([np.asarray(part[name]) for part in parts], axis=0)
        for name in names
    }
    # Overlap windows export some identical measurements twice. Keep the
    # highest-confidence copy so overlap does not receive artificial weight.
    best_by_key: dict[tuple[int, int, int], int] = {}
    for index, (left, right, kind, confidence) in enumerate(zip(
        merged["frame_i"],
        merged["frame_j"],
        merged["edge_type"],
        merged["confidence"],
    )):
        key = (int(left), int(right), int(kind))
        previous = best_by_key.get(key)
        if previous is None or float(confidence) > float(merged["confidence"][previous]):
            best_by_key[key] = index
    keep = np.asarray(sorted(best_by_key.values()), dtype=np.int64)
    for name in names:
        merged[name] = merged[name][keep]
    count = len(keep)
    np.savez_compressed(
        combined_output / "pose_graph_edges.npz",
        schema_version=np.asarray([R3_POSE_GRAPH_SCHEMA_VERSION], dtype=np.int32),
        pose_encoding=np.asarray(R3_POSE_ENCODING),
        transform_convention=np.asarray(R3_RELATIVE_TRANSFORM_CONVENTION),
        frame_index_space=np.asarray("exported_camera_index"),
        absolute_pose_space=np.asarray(R3_ABSOLUTE_POSE_SPACE),
        confidence_semantics=np.asarray(R3_CONFIDENCE_SEMANTICS),
        edge_type_names=np.asarray(["normal", "bridge", "anchor", "unknown"]),
        edge_sequence=np.arange(count, dtype=np.int64),
        model_frame_i=merged["frame_i"].astype(np.int64),
        model_frame_j=merged["frame_j"].astype(np.int64),
        **merged,
    )
    return count


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
    _prepare_r3_pose_graph_export()

    combined_output = Path(output_dir)
    segments_root = combined_output / "segments"
    segments_root.mkdir(parents=True, exist_ok=True)
    merged_poses: dict[int, np.ndarray] = {}
    merged_confidence = np.full(len(source_frames), np.nan, dtype=np.float64)
    segment_scales: list[float] = []
    manifest: dict = {
        "enabled": True,
        "total_selected_frames": len(source_frames),
        "segment_frames": segment_frames,
        "overlap_frames": overlap_frames,
        "segments": [],
    }
    first_run_params: dict = {}
    pose_graph_parts: list[dict[str, object]] = []
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
            scale_prior=(
                float(np.median(segment_scales[-3:]))
                if segment_scales else None
            ),
        )
        segment_scales.append(float(stitch.get("scale") or 1.0))
        pose_graph_part = _read_segment_pose_graph(
            resolved_output,
            global_indices,
            float(stitch.get("scale") or 1.0),
        )
        if pose_graph_part is not None:
            pose_graph_parts.append(pose_graph_part)
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
    manifest["pose_graph_edges"] = _save_merged_pose_graph(
        combined_output,
        pose_graph_parts,
    )
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

    _prepare_r3_pose_graph_export()
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

    params_path = Path(output_dir) / "run_params.json"
    if params_path.exists():
        try:
            params = json.loads(params_path.read_text())
            params["mode"] = resolved_mode
            params["wrapper_requested_mode"] = mode
            params["wrapper_resolved_ckpt"] = resolved_ckpt
            params["wrapper_input_size"] = int(size)
            params["wrapper_max_frames"] = int(max_frames)
            params_path.write_text(json.dumps(sanitize_json(params), indent=2), encoding="utf-8")
        except Exception as exc:
            emit("warning", {"message": f"failed to stamp live run params: {exc}"})

    return output_dir


def resolve_long_execution_policy(mode: str) -> tuple[bool, bool, str]:
    """Resolve the production long-video execution strategy.

    R3's own long mode keeps one global keyframe bank and pose graph. The old
    R3_ENABLE_EXTERNAL_SEGMENTATION flag ran independent models and then joined
    them with pairwise Sim(3), which discards global loop edges. Keep that
    implementation available only behind the new, explicit experimental
    policy so stale service environments cannot silently re-enable it.
    """
    long_mode = (mode or "").lower() in {"long", "strided", "sampled", "sparse"}
    if not long_mode:
        return False, False, "short"
    requested = (
        os.getenv("R3_LONG_EXECUTION_POLICY") or "segmented_pose_graph"
    ).strip().lower()
    if requested in {"continuous", "continuous_experimental"}:
        return True, False, "continuous_experimental"
    return False, True, "segmented_pose_graph"


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

    pose_edges = []
    pose_edges_path = r3_output / "pose_edge_log.json"
    if pose_edges_path.exists():
        try:
            loaded_edges = json.loads(pose_edges_path.read_text())
            if isinstance(loaded_edges, list):
                pose_edges = loaded_edges
        except Exception:
            pose_edges = []
    try:
        bridge_window = int(run_params.get("fallback_num_bridge_frames") or 10)
    except (TypeError, ValueError):
        bridge_window = 10
    fallback_summary = summarize_fallback_edges(
        pose_edges,
        point_count=len(poses),
        bridge_window=bridge_window,
    )
    pose_graph_summary = load_pose_graph_summary(
        r3_output / "pose_graph_edges.npz",
        point_count=len(poses),
    )
    optimizer_mode = (os.getenv("R3_POSE_GRAPH_OPTIMIZER_MODE") or "shadow").strip().lower()
    pose_graph_candidate = load_pose_graph_candidate_summary(r3_output)
    if (
        optimizer_mode == "shadow"
        and pose_graph_summary.get("optimizer_ready", False)
        and len(poses) >= 2
    ):
        emit("r3_pose_graph_optimizer_start", {
            "mode": "shadow",
            "poses": len(poses),
            "edges": pose_graph_summary.get("edge_count", 0),
        })
        pose_graph_candidate = run_pose_graph_shadow(
            r3_output,
            np.asarray([pose["pose"] for pose in poses], dtype=np.float64),
        )
        emit("r3_pose_graph_optimizer_complete", {
            "accepted": pose_graph_candidate.get("accepted", False),
            "runtime_seconds": pose_graph_candidate.get("runtime_seconds"),
            "objective_improvement": pose_graph_candidate.get("objective_improvement"),
            "rejection_reasons": pose_graph_candidate.get("rejection_reasons", []),
        })
    elif optimizer_mode not in {"off", "shadow"}:
        pose_graph_candidate = {
            "available": False,
            "accepted": False,
            "error": f"unsupported optimizer mode: {optimizer_mode}",
        }

    scale_aware_mode = (os.getenv("R3_SCALE_AWARE_MODE") or "shadow").strip().lower()
    scale_aware_candidate = load_scale_aware_candidate_summary(r3_output)
    floor_scale_diagnostics: dict = {"available": False, "reason": "disabled"}
    if scale_aware_mode == "shadow" and len(poses) >= 2 and (r3_output / "depth").exists():
        raw_c2w = np.asarray([pose["pose"] for pose in poses], dtype=np.float64)
        robust_c2w = load_pose_graph_candidate_c2w(
            r3_output,
            expected_count=len(poses),
            accepted_only=True,
        )
        scale_base = robust_c2w if robust_c2w is not None else raw_c2w
        emit("r3_scale_aware_start", {
            "mode": "shadow",
            "poses": len(poses),
            "base": "robust_candidate" if robust_c2w is not None else "raw",
        })
        observations, floor_scale_diagnostics = estimate_floor_height_observations(
            r3_output,
            scale_base,
            maximum_frames=max(24, int(os.getenv("R3_FLOOR_SCALE_MAX_FRAMES") or "180")),
        )
        scale_result = build_scale_aware_candidate(scale_base, observations)
        scale_result["diagnostics"].update({
            "base_source": "robust_candidate" if robust_c2w is not None else "raw",
            "floor_estimation": floor_scale_diagnostics,
        })
        scale_aware_candidate = save_scale_aware_candidate(r3_output, scale_result)
        emit("r3_scale_aware_complete", {
            "accepted": scale_aware_candidate.get("accepted", False),
            "observations": scale_aware_candidate.get("observation_count", 0),
            "scale_range": scale_aware_candidate.get("scale_range"),
            "rejection_reasons": scale_aware_candidate.get("rejection_reasons", []),
        })
    elif scale_aware_mode not in {"off", "shadow"}:
        scale_aware_candidate = {
            "available": False,
            "accepted": False,
            "error": f"unsupported scale-aware mode: {scale_aware_mode}",
        }

    result = {
        "success": True,
        "num_frames": len(poses),
        "run_params": {
            "config_name": run_params.get("config_name"),
            "ckpt": run_params.get("ckpt"),
            "wrapper_mode": run_params.get("wrapper_mode"),
            "mode": run_params.get("mode"),
            "inference_time_s": run_params.get("inference_time_s"),
            "online_fallback_enabled": run_params.get("online_fallback_enabled", False),
            "max_segment_frames": run_params.get("max_segment_frames"),
            "metric_scale_enabled": run_params.get("metric_scale_enabled", False),
            "metric_bootstrap_frames": run_params.get("metric_bootstrap_frames"),
            "depth_scale_mode": run_params.get("depth_scale_mode"),
            "segmented_long": run_params.get("segmented_long", False),
            "segment_count": run_params.get("segment_count"),
            "segment_frames": run_params.get("segment_frames"),
            "segment_overlap_frames": run_params.get("segment_overlap_frames"),
            "fallback_min_bridge_baseline_ratio": run_params.get("fallback_min_bridge_baseline_ratio"),
            "fallback_max_bridge_lookback": run_params.get("fallback_max_bridge_lookback"),
            "fallback_boundaries": fallback_summary["boundaries"],
            "fallback_boundary_source": fallback_summary["source"],
            "fallback_events": fallback_summary["events"],
            "pose_graph_optimizer_ready": pose_graph_summary.get("optimizer_ready", False),
            "pose_graph_optimizer_mode": optimizer_mode,
            "pose_graph_candidate_accepted": pose_graph_candidate.get("accepted", False),
            "pose_graph_optimizer_seconds": pose_graph_candidate.get("runtime_seconds"),
            "scale_aware_mode": scale_aware_mode,
            "scale_aware_candidate_accepted": scale_aware_candidate.get("accepted", False),
            "scale_aware_observations": scale_aware_candidate.get("observation_count", 0),
        },
        "camera_poses": sanitize_json(poses),
        "num_poses_total": len(poses),
        "pose_confidence": sanitize_json(pose_conf if pose_conf else None),
        "output_dir": str(r3_output),
        "frame_selection": sanitize_json(frame_selection),
        "pose_graph": sanitize_json(pose_graph_summary),
        "pose_graph_candidate": sanitize_json(pose_graph_candidate),
        "scale_aware_candidate": sanitize_json(scale_aware_candidate),
        "floor_scale_diagnostics": sanitize_json(floor_scale_diagnostics),
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

    continuous_long_enabled, segmented_enabled, long_execution_policy = (
        resolve_long_execution_policy(args.mode)
    )
    emit("r3_long_execution_policy", {
        "policy": long_execution_policy,
        "continuous_long": continuous_long_enabled,
        "external_segmentation": segmented_enabled,
    })
    segment_min_duration = float(
        os.getenv(
            "R3_LONG_MIN_DURATION_SECONDS",
            os.getenv("R3_SEGMENT_MIN_DURATION_SECONDS", "600"),
        )
    )
    # Turns made while walking often complete in about one second.  Five
    # observations per second are marginal once blur/occlusion removes even a
    # couple of frames; eight preserves temporal overlap without approaching
    # the source video's full GPU/memory cost.
    long_target_fps = float(os.getenv("R3_LONG_TARGET_FPS", "8"))

    # Step 1: preserve a dense-enough sequence across the whole route. Native
    # long mode receives every selected frame in one process.
    frames_dir, num_frames = extract_frames(
        args.video_path,
        args.output_dir,
        args.frame_stride,
        args.max_frames,
        continuous_long=continuous_long_enabled,
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
    selected_fps = 0.0
    try:
        selected_fps = float(frame_selection.get("fps") or 0.0) / max(
            1, int(frame_selection.get("frame_stride") or 1)
        )
    except (TypeError, ValueError, ZeroDivisionError):
        selected_fps = max(1.0, long_target_fps)
    segment_seconds = max(180.0, min(300.0, float(
        os.getenv("R3_LONG_SEGMENT_SECONDS", "240")
    )))
    overlap_seconds = max(15.0, min(60.0, float(
        os.getenv("R3_LONG_SEGMENT_OVERLAP_SECONDS", "30")
    )))
    segment_frames = max(256, int(round(selected_fps * segment_seconds)))
    overlap_frames = max(16, int(round(selected_fps * overlap_seconds)))
    use_segmented = bool(frame_selection.get("segmented_long") and num_frames > segment_frames)
    inference_max_frames = 0 if frame_selection.get("long_video_sampling") else args.max_frames
    inference_mode = "long" if frame_selection.get("continuous_long") else args.mode

    # Step 2: Run R³ inference (live or standard)
    if use_segmented:
        run_r3_inference_segmented(
            frames_dir,
            args.output_dir,
            args.ckpt,
            "long",
            args.size,
            segment_frames,
            overlap_frames,
        )
    elif args.live:
        run_r3_inference_live(frames_dir, args.output_dir, camera_dir,
                              args.ckpt, inference_mode, args.size, inference_max_frames)
    else:
        run_r3_inference(
            frames_dir,
            args.output_dir,
            args.ckpt,
            inference_mode,
            args.size,
            inference_max_frames,
        )

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
