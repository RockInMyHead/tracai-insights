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
}

export default function RealTimeR3Visualization({
  videoId,
  onComplete,
  onClose,
  floorPlan = null,
  drawnPlan = null,
  referencePoint = null,
  directionPoint = null,
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

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  // Subscribe to SSE stream
  useEffect(() => {
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
          const mb = Math.round((data.received_bytes || 0) / 1024 / 1024);
          setStatusMessage(`Загрузка видео на GPU: ${mb} MB...`);
        } else if (data.event_type === "video_received") {
          setStatusMessage("Видео загружено, запуск R³...");
        } else if (data.event_type === "r3_start") {
          setStatusMessage("R³ запущен, обработка кадров...");
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
          setPoints([...pointsRef.current]);

          // Calculate distance
          const pts = pointsRef.current;
          let dist = 0;
          for (let i = 1; i < pts.length; i++) {
            const dx = pts[i][0] - pts[i - 1][0];
            const dy = pts[i][1] - pts[i - 1][1];
            const dz = (pts[i][2] || 0) - (pts[i - 1][2] || 0);
            dist += Math.sqrt(dx * dx + dy * dy + dz * dz);
          }
          setDistance(dist);

          setFrameCount(data.num_processed || pts.length);
          setStatusMessage(
            `Обработано ${data.num_processed || pts.length} кадров, ${pts.length} точек`
          );
          setAnimProgress(Math.min(95, (pts.length / Math.max(totalFramesRef.current || 100, 1)) * 100));
        }

        // Capture full pose data for better 3D reconstruction
        const rawPoses = data.new_poses;
        if (Array.isArray(rawPoses) && rawPoses.length > 0) {
          const validPoses = rawPoses
            .filter(p => p && Array.isArray(p.pose) && p.pose.length >= 3)
            .map(p => ({
              frame: typeof p.frame === 'number' ? p.frame : 0,
              pose: p.pose as number[][],
              intrinsics: Array.isArray(p.intrinsics) ? p.intrinsics as number[][] : undefined,
            }));
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
        const resultData = data.result || data;
        if (resultData) {
          const pc = resultData.pointcloud_sample || resultData.pointcloud;
          totalPcCount = resultData.pointcloud_count || 0;
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
    });

    unsubscribeRef.current = unsub;

    return () => {
      unsub();
    };
  }, [videoId]);

  const handleClose = () => {
    unsubscribeRef.current?.();
    onClose?.();
  };

  const isLive = status === "connecting" || status === "processing";

  const liveTrajectories = useMemo<TrajectoryData[]>(() => {
    if (points.length < 2) return [];
    return [{
      trajectory: points.map((point) => ({
        x: Number(point[0]) || 0,
        y: Number(point[1]) || 0,
        z: Number(point[2]) || 0,
      })),
      turnPoints: [],
      ownerName: "Live R³",
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

  return (
    <Card className={`w-full border-2 border-primary/20 ${isFullscreen3d ? "fixed inset-0 z-[99] rounded-none border-0" : ""}`}>
      {/* Кнопка закрытия при fullscreen */}
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
      {!isFullscreen3d && (<>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            {isLive ? (
              <>
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500" />
                </span>
                Live — траектория рисуется по мере готовности
              </>
            ) : status === "complete" ? (
              <>
                <span className="h-3 w-3 rounded-full bg-green-500 inline-block" />
                R³ реконструкция завершена
              </>
            ) : (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Ошибка
              </>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            {status === "complete" && (
              <Badge variant="default" className="bg-green-500/20 text-green-500 border-green-500/30">
                {frameCount} кадров
              </Badge>
            )}
            <Button variant="ghost" size="icon" onClick={handleClose} className="h-8 w-8">
              <EyeOff className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          {statusMessage}
          {errorMessage ? ` · ${errorMessage}` : ""}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Progress */}
        <Progress value={animProgress} className="w-full h-1.5" />

        <div className={`grid gap-3 ${showFloorplanLive ? "xl:grid-cols-2" : "grid-cols-1"}`}>
          {showFloorplanLive && (
            <div className="overflow-hidden rounded-lg border border-primary/20 bg-background">
              <div className="border-b border-border/40 px-3 py-2 text-sm font-medium">
                План: live-траектория ({points.length.toLocaleString("ru-RU")} точек)
              </div>
              <div className="h-[420px]">
                {liveTrajectories.length > 0 ? (
                  <TrajectoryMap
                    trajectories={liveTrajectories}
                    floorPlan={floorPlan}
                    drawnPlan={drawnPlan}
                    referencePoint={referencePoint}
                    directionPoint={directionPoint}
                    playbackPointLimit={points.length}
                    compactMode
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    <div className="text-center">
                      <Loader2 className="mx-auto mb-2 h-8 w-8 animate-spin text-primary" />
                      Ждём первые точки R³…
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 3D Visualization */}
          <div className="relative overflow-hidden rounded-lg border border-border/30 bg-black/40">
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
              <div
                className="flex items-center justify-center"
                style={{ height: showFloorplanLive ? 420 : 500 }}
              >
                <div className="text-center">
                  <Loader2 className="h-10 w-10 animate-spin mx-auto mb-3 text-primary" />
                  <p className="text-sm text-muted-foreground">Ожидание первой 3D точки...</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Status */}
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            {isLive && <Loader2 className="h-3 w-3 animate-spin" />}
            <span>
              {isLive
                ? "Линия на плане удлиняется по мере обработки кадров"
                : statusMessage}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {fps > 0 && (
              <Badge variant="outline" className="text-[10px]">
                {fps.toFixed(1)} кадр/с
              </Badge>
            )}
            {status === "complete" && processingTime > 0 && (
              <Badge variant="default" className="bg-green-500/20 text-green-500 border-green-500/30 text-[10px]">
                {processingTime.toFixed(1)} с
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
      </>)}
    </Card>
  );
}
