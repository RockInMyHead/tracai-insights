import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Loader2, EyeOff } from "lucide-react";
import { apiClient } from "@/lib/api";
import R3Visualization3D from "./R3Visualization3D";
import TrajectoryMap, { type TrajectoryData } from "./TrajectoryMap";

interface Props {
  videoId: string;
  onComplete?: (result: Record<string, unknown>) => void;
  onClose?: () => void;
  floorPlan?: string | null;
  drawnPlan?: unknown[] | null;
  referencePoint?: { x: number; y: number } | null;
  directionPoint?: { x: number; y: number } | null;
  /** When true, follow the production analyze job (no second R³ upload). */
  followActiveAnalysis?: boolean;
  /** Parent still running analyzeVideo — keep draft live state. */
  isAnalyzing?: boolean;
  /** Live draft point count for the parent progress dashboard. */
  onLiveStats?: (stats: { points: number }) => void;
}

function pickLivePoints(payload: {
  plan_trajectory?: number[][];
  raw_plan_trajectory?: number[][];
  trajectory?: number[][];
  raw_trajectory_3d?: number[][];
}): number[][] {
  const candidates = [
    payload.plan_trajectory,
    payload.raw_plan_trajectory,
    payload.trajectory,
    payload.raw_trajectory_3d,
  ];
  for (const candidate of candidates) {
    if (!Array.isArray(candidate) || candidate.length < 1) continue;
    const valid = candidate.filter((p) => Array.isArray(p) && p.length >= 2);
    if (valid.length > 0) return valid as number[][];
  }
  return [];
}

export default function RealTimeR3Visualization({
  videoId,
  onComplete,
  onClose,
  floorPlan = null,
  drawnPlan = null,
  referencePoint = null,
  directionPoint = null,
  followActiveAnalysis = false,
  isAnalyzing = false,
  onLiveStats,
}: Props) {
  const [points, setPoints] = useState<number[][]>([]);
  const [poses, setPoses] = useState<{frame: number; pose: number[][]; intrinsics?: number[][]}[]>([]);
  const [pointCloud, setPointCloud] = useState<number[][] | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [status, setStatus] = useState<"connecting" | "processing" | "complete" | "error">("connecting");
  const [statusMessage, setStatusMessage] = useState("Подключение к GPU...");
  const [distance, setDistance] = useState(0);
  const [fps, setFps] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");
  // Keep inline so the live floor-plan panel stays visible during analysis.
  const [isFullscreen3d, setIsFullscreen3d] = useState(false);
  const unsubscribeRef = useRef<(() => void) | null>(null);
  const pointsRef = useRef<number[][]>([]);
  const posesRef = useRef<{frame: number; pose: number[][]; intrinsics?: number[][]}[]>([]);
  const [animProgress, setAnimProgress] = useState(0);
  const completedRef = useRef(false);
  const totalFramesRef = useRef(0);
  const onCompleteRef = useRef(onComplete);
  const lastPointCountRef = useRef(0);
  const stablePollsRef = useRef(0);

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const applyTrajectoryPoints = (nextPoints: number[][], processedHint?: number) => {
    if (nextPoints.length === 0) return;
    pointsRef.current = nextPoints;
    setPoints([...nextPoints]);

    let dist = 0;
    for (let i = 1; i < nextPoints.length; i++) {
      const dx = nextPoints[i][0] - nextPoints[i - 1][0];
      const dy = nextPoints[i][1] - nextPoints[i - 1][1];
      const dz = (nextPoints[i][2] || 0) - (nextPoints[i - 1][2] || 0);
      dist += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    setDistance(dist);
    setFrameCount(processedHint || nextPoints.length);
    setStatus("processing");
    setStatusMessage(
      `${nextPoints.length.toLocaleString("ru-RU")} точек · черновик из готовых кадров`
    );
    setAnimProgress(Math.min(95, (nextPoints.length / Math.max(totalFramesRef.current || 100, 1)) * 100));
    onLiveStats?.({ points: nextPoints.length });
  };

  // Poll growing plan trajectory while production analyze holds the GPU lock.
  useEffect(() => {
    if (!followActiveAnalysis) return;

    let cancelled = false;
    let inFlight = false;
    let abortController: AbortController | null = null;
    setStatus("connecting");
    setStatusMessage("Ожидание первых кадров…");
    setErrorMessage("");
    setPoints([]);
    setPointCloud(null);
    pointsRef.current = [];
    posesRef.current = [];
    setFrameCount(0);
    setDistance(0);
    setAnimProgress(0);
    completedRef.current = false;
    lastPointCountRef.current = 0;
    stablePollsRef.current = 0;
    // Ignore any leftover trajectory from a previous analyze of the same video_id
    // until we observe an empty payload (stale artifacts cleared) or slow growth.
    let armedForFreshRun = false;
    const sessionStartedAt = Date.now();

    const poll = async () => {
      if (cancelled || completedRef.current || inFlight) return;
      inFlight = true;
      abortController?.abort();
      abortController = new AbortController();
      try {
        const resp = await apiClient.getR3Trajectory(videoId, "raw", {
          livePreview: true,
          signal: abortController.signal,
          timeoutMs: 5000,
        });
        if (cancelled) return;
        const nextPoints = pickLivePoints(resp);
        const suppressed = Boolean((resp as { live_preview_suppressed?: boolean }).live_preview_suppressed);

        if (!armedForFreshRun) {
          if (suppressed || nextPoints.length === 0) {
            armedForFreshRun = true;
            setPoints([]);
            pointsRef.current = [];
            setStatus("processing");
            setStatusMessage("Ждём новые кадры текущего анализа…");
            return;
          }
          // Large instant dump right after start = previous completed run. Ignore it.
          const ageSec = (Date.now() - sessionStartedAt) / 1000;
          if (ageSec < 20 && nextPoints.length >= 50) {
            setStatus("processing");
            setStatusMessage("Отбрасываю старый маршрут — жду текущий прогон…");
            return;
          }
          armedForFreshRun = true;
        }

        if (nextPoints.length > 0) {
          applyTrajectoryPoints(nextPoints, nextPoints.length);
        } else {
          setStatus("processing");
          setStatusMessage("Обработка кадров…");
        }

        if (nextPoints.length === lastPointCountRef.current && nextPoints.length > 1) {
          stablePollsRef.current += 1;
        } else {
          stablePollsRef.current = 0;
          lastPointCountRef.current = nextPoints.length;
        }

        // After parent analyze finishes and the path stops growing, mark draft complete.
        if (!isAnalyzing && nextPoints.length >= 2 && stablePollsRef.current >= 2) {
          completedRef.current = true;
          setStatus("complete");
          setAnimProgress(100);
          setStatusMessage(
            `${nextPoints.length.toLocaleString("ru-RU")} точек · черновик готов`
          );
          onCompleteRef.current?.({ plan_trajectory: nextPoints, follow_active: true });
        }
      } catch {
        if (cancelled || completedRef.current) return;
        // 404 / timeout while GPU still starting is expected — keep waiting.
        setStatus("processing");
        setStatusMessage(
          isAnalyzing ? "Обработка на GPU…" : "Ожидание траектории…"
        );
      } finally {
        inFlight = false;
      }
    };

    void poll();
    const interval = setInterval(() => {
      void poll();
    }, 2500);

    return () => {
      cancelled = true;
      clearInterval(interval);
      abortController?.abort();
    };
  }, [videoId, followActiveAnalysis, isAnalyzing]);

  // Subscribe to SSE stream (standalone live / replay path)
  useEffect(() => {
    if (followActiveAnalysis) return;

    setStatus("connecting");
    setStatusMessage("Подключение к GPU-серверу...");
    setPoints([]);
    setPointCloud(null);
    pointsRef.current = [];
    posesRef.current = [];
    setFrameCount(0);
    setDistance(0);
    setAnimProgress(0);
    completedRef.current = false;

    const unsub = apiClient.subscribeR3Stream(videoId, {
      onStatus: (data) => {
        if (data.status === "receiving_video") {
          setStatus("processing");
          setStatusMessage("Получение видео на GPU-сервере...");
        } else if (data.event_type === "receiving") {
          const mb = Math.round((Number(data.received_bytes) || 0) / 1024 / 1024);
          setStatusMessage(`Загрузка видео на GPU: ${mb} MB...`);
        } else if (data.event_type === "video_received") {
          setStatusMessage("Видео загружено, запуск R³...");
        } else if (data.event_type === "r3_start") {
          setStatusMessage("R³ запущен, обработка кадров...");
        } else if (data.event_type === "watching") {
          setStatus("processing");
          setStatusMessage("Live: следим за активным анализом — линия растёт по кадрам (черновик)");
        } else if (data.event_type === "replay") {
          const numFiles = data.npz_files || 0;
          setStatusMessage(`Загрузка готовых результатов R³ (${numFiles} точек)...`);
        } else if (data.event_type === "pointcloud_status") {
          const progress = typeof data.progress === "number" ? ` (${data.progress}%)` : "";
          setStatusMessage(`${String(data.message || "Строится 3D-облако")}${progress}`);
        } else if (data.event_type === "r3_segment_start") {
          setStatusMessage(
            `Длинное видео: R³ блок ${String(data.segment || "?")}/${String(data.segments_total || "?")}...`
          );
        } else if (data.event_type === "r3_segment_complete") {
          setStatusMessage(
            `Готов блок ${String(data.index !== undefined ? Number(data.index) + 1 : "?")} — сшиваю траекторию...`
          );
        } else if (data.event_type === "r3_segmented_complete") {
          setStatusMessage(`Все ${String(data.segments || "")} блоки объединены`);
        }
      },
      onVideoInfo: (data) => {
        totalFramesRef.current = data.frames || 0;
        setTotalFrames(data.frames || 0);
        setStatus("processing");
        setStatusMessage(`Загружено ${data.frames || "..."} кадров, обрабатываю...`);
      },
      onFrameProcessed: (data) => {
        // Capture trajectory points
        const rawTraj = data.new_trajectory_points;
        const newTraj = Array.isArray(rawTraj) ? rawTraj.filter(p => Array.isArray(p)) : [];
        if (newTraj.length > 0) {
          pointsRef.current = [...pointsRef.current, ...newTraj];
          applyTrajectoryPoints(pointsRef.current, data.num_processed || pointsRef.current.length);
        }

        // Capture full pose data for better 3D reconstruction
        const rawPoses = data.new_poses;
        if (Array.isArray(rawPoses) && rawPoses.length > 0) {
          const validPoses = rawPoses
            .filter(p => p && Array.isArray((p as { pose?: unknown }).pose) && ((p as { pose: unknown[] }).pose.length >= 3))
            .map(p => {
              const item = p as { frame?: number; pose: number[][]; intrinsics?: number[][] };
              return {
                frame: typeof item.frame === "number" ? item.frame : 0,
                pose: item.pose,
                intrinsics: Array.isArray(item.intrinsics) ? item.intrinsics : undefined,
              };
            });
          if (validPoses.length > 0) {
            posesRef.current = [...posesRef.current, ...validPoses];
            setPoses([...posesRef.current]);
          }
        }
      },
      onComplete: (data) => {
        completedRef.current = true;
        setStatus("complete");
        setAnimProgress(100);

        // ── 1. Show sample point cloud from SSE immediately ──────
        let totalPcCount = 0;
        const resultData = (data.result || data) as Record<string, unknown>;
        if (resultData) {
          const pc = resultData.pointcloud_sample || resultData.pointcloud;
          totalPcCount = Number(resultData.pointcloud_count) || 0;
          if (Array.isArray(pc) && pc.length > 0) {
            const validPc = pc.filter(p => Array.isArray(p) && p.length >= 3);
            if (validPc.length >= 50) {
              setPointCloud(validPc);
              setStatusMessage(
                `Реконструкция завершена! Облако точек: ${(totalPcCount || validPc.length).toLocaleString()}`
              );
            } else {
              setStatusMessage("Реконструкция завершена!");
            }
          } else {
            setStatusMessage("Реконструкция завершена!");
          }
        }

        if (data.total_time_s) {
          setProcessingTime(data.total_time_s as number);
          setStatusMessage(prev => `${prev} за ${(data.total_time_s as number).toFixed(1)} с`);
        }
        if (data.processing_stats) {
          const stats = data.processing_stats as Record<string, unknown>;
          if (typeof stats.fps === "number") setFps(stats.fps);
        }
        // Fix totalFrames in case it wasn't set (e.g. replay mode)
        const numFrames = (data as Record<string, unknown>).num_frames;
        if (typeof numFrames === "number" && numFrames > 0 && totalFramesRef.current === 0) {
          totalFramesRef.current = numFrames;
          setTotalFrames(numFrames);
        }
        onCompleteRef.current?.(data);

        // ── 2. Fetch full point cloud from API (SSE only carries a small preview) ─────
        apiClient.getR3PointCloudFiltered(videoId, {
          maxPoints: 100000,
          minConf: 1.4,
          samplingStrategy: "per_frame_uniform",
          includeTrajectory: true,
          includeCameras: false,
        }).then(resp => {
          // The 3D viewer must receive c2w translations.  `trajectory` is now
          // plan-space and is deliberately reserved for the floor-map screen.
          const rawTrajectory = resp.raw_trajectory_3d ?? resp.trajectory;
          if (resp.success && Array.isArray(rawTrajectory) && rawTrajectory.length >= 2) {
            const validTrajectory = rawTrajectory.filter(p => Array.isArray(p) && p.length >= 3);
            if (validTrajectory.length >= 2) {
              setPoints(validTrajectory);
              pointsRef.current = validTrajectory;
            }
          }
          const cleanedDistance = resp.stats?.trajectory_quality?.cleaned_distance;
          if (typeof cleanedDistance === "number" && Number.isFinite(cleanedDistance)) {
            setDistance(cleanedDistance);
          }
          if (resp.success && Array.isArray(resp.points) && resp.points.length > 200) {
            const valid = resp.points.filter(p => Array.isArray(p) && p.length >= 3);
            if (valid.length > 200) {
              setPointCloud(valid);
              setStatusMessage(
                `Реконструкция завершена! Облако точек: ${(resp.stats?.filtered_points || valid.length).toLocaleString()}`
              );
            }
          }
        }).catch(err => {
          console.warn("Failed to fetch filtered point cloud:", err);
        });
      },
      onError: (err) => {
        if (completedRef.current) return;
        setStatus("error");
        setErrorMessage(err);
        setStatusMessage(`Ошибка: ${err}`);
      },
    }, { followActive: false });

    unsubscribeRef.current = unsub;

    return () => {
      unsub();
    };
  }, [videoId, followActiveAnalysis]);

  const handleClose = () => {
    unsubscribeRef.current?.();
    onClose?.();
  };

  const isLive = status === "connecting" || status === "processing";
  // During production analyze, show only the plan draft — hide the 3D debug console.
  const planFirstLive = followActiveAnalysis || isAnalyzing;

  const liveTrajectories = useMemo<TrajectoryData[]>(() => {
    if (points.length < 2) return [];
    return [{
      trajectory: points.map((point) => ({
        x: Number(point[0]) || 0,
        y: Number(point[1]) || 0,
        z: Number(point[2]) || 0,
      })),
      turnPoints: [],
      ownerName: "Черновик",
      color: "#38bdf8",
      videoId,
      method: "r3_reconstruction",
      coordinateConvention: "x_forward_y_left_z_up",
      mapAligned: false,
      r3AutoFitToPlan: true,
      mapScaleFactor: 1,
    }];
  }, [points, videoId]);

  const showFloorplanLive = Boolean(floorPlan || drawnPlan);
  const pointLabel = (frameCount || points.length).toLocaleString("ru-RU");

  return (
    <Card className={`w-full overflow-hidden border border-border/40 bg-card/60 ${isFullscreen3d ? "fixed inset-0 z-[99] rounded-none border-0" : ""}`}>
      {isFullscreen3d && (
        <Button
          variant="ghost"
          size="icon"
          onClick={handleClose}
          className="absolute top-4 left-4 z-50 h-9 w-9 rounded-full bg-background/60 backdrop-blur-sm hover:bg-background/90"
          title="Закрыть"
        >
          <EyeOff className="h-5 w-5" />
        </Button>
      )}
      {!isFullscreen3d && (
        <>
          <CardHeader className="space-y-3 pb-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 space-y-1">
                <CardTitle className="flex items-center gap-2.5 text-base font-semibold tracking-tight">
                  {isLive ? (
                    <span className="relative flex h-2.5 w-2.5 shrink-0">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/70" />
                      <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-400" />
                    </span>
                  ) : status === "complete" ? (
                    <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-emerald-400" />
                  ) : (
                    <Loader2 className="h-4 w-4 shrink-0 animate-spin text-destructive" />
                  )}
                  <span className="truncate">
                    {status === "error"
                      ? "Ошибка live-просмотра"
                      : status === "complete"
                        ? "Черновик маршрута"
                        : "Live на плане"}
                  </span>
                </CardTitle>
                <p className="text-sm text-muted-foreground">
                  {status === "error"
                    ? (errorMessage || statusMessage)
                    : isLive
                      ? "Черновик из уже посчитанных кадров · GPU ещё считает остальное видео"
                      : statusMessage}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {points.length > 0 && (
                  <Badge
                    variant="outline"
                    className="border-border/50 bg-secondary/40 font-mono text-[11px] text-muted-foreground"
                  >
                    {pointLabel}
                  </Badge>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleClose}
                  className="h-8 w-8 text-muted-foreground"
                  title="Скрыть"
                >
                  <EyeOff className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <Progress value={animProgress} className="h-1 w-full bg-secondary/60" />
          </CardHeader>

          <CardContent className="space-y-3 pt-0">
            {showFloorplanLive ? (
              <div className="overflow-hidden rounded-xl border border-border/40 bg-secondary/20">
                <div className="h-[min(58vh,560px)] min-h-[360px]">
                  {liveTrajectories.length > 0 ? (
                    <TrajectoryMap
                      trajectories={liveTrajectories}
                      floorPlan={floorPlan}
                      drawnPlan={drawnPlan}
                      referencePoint={referencePoint}
                      directionPoint={directionPoint}
                      playbackPointLimit={points.length}
                      compactMode
                      minimalChrome
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                      <div className="text-center">
                        <Loader2 className="mx-auto mb-3 h-7 w-7 animate-spin text-primary/80" />
                        <p>Ждём первые точки…</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : null}

            {/* Full 3D console only outside production live-follow */}
            {!planFirstLive && (
              <div className="relative overflow-hidden rounded-xl border border-border/30 bg-black/40">
                {points.length > 0 ? (
                  <R3Visualization3D
                    videoId={videoId}
                    points={points}
                    poses={poses}
                    pointCloud={pointCloud}
                    totalFrames={totalFrames}
                    distance={distance}
                    onFullscreenChange={(full) => setIsFullscreen3d(full)}
                  />
                ) : (
                  <div className="flex h-[420px] items-center justify-center">
                    <div className="text-center">
                      <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-primary" />
                      <p className="text-sm text-muted-foreground">Ожидание 3D…</p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {!planFirstLive && (
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <div className="flex items-center gap-2">
                  {isLive && <Loader2 className="h-3 w-3 animate-spin" />}
                  <span>{statusMessage}</span>
                </div>
                <div className="flex items-center gap-2">
                  {fps > 0 && (
                    <Badge variant="outline" className="text-[10px]">
                      {fps.toFixed(1)} кадр/с
                    </Badge>
                  )}
                  {status === "complete" && processingTime > 0 && (
                    <Badge
                      variant="outline"
                      className="border-emerald-500/30 text-[10px] text-emerald-400"
                    >
                      {processingTime.toFixed(1)} с
                    </Badge>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </>
      )}
    </Card>
  );
}
