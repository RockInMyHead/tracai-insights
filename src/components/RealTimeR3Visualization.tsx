import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Loader2, EyeOff } from "lucide-react";
import { apiClient } from "@/lib/api";
import R3Visualization3D from "./R3Visualization3D";

interface Props {
  videoId: string;
  onComplete?: (result: Record<string, unknown>) => void;
  onClose?: () => void;
}

export default function RealTimeR3Visualization({ videoId, onComplete, onClose }: Props) {
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
  const unsubscribeRef = useRef<(() => void) | null>(null);
  const pointsRef = useRef<number[][]>([]);
  const posesRef = useRef<{frame: number; pose: number[][]; intrinsics?: number[][]}[]>([]);
  const [animProgress, setAnimProgress] = useState(0);
  const completedRef = useRef(false);
  const totalFramesRef = useRef(0);

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
        onComplete?.(data);

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

  return (
    <Card className="w-full border-2 border-primary/20">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            {isLive ? (
              <>
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500" />
                </span>
                R³ Live — реконструкция в реальном времени
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
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Progress */}
        <Progress value={animProgress} className="w-full h-1.5" />

        {/* 3D Visualization */}
        <div className="relative rounded-lg overflow-hidden bg-black/40 border border-border/30">
          {points.length > 0 ? (
            <R3Visualization3D
              videoId={videoId}
              points={points}
              poses={poses}
              pointCloud={pointCloud}
              totalFrames={totalFrames}
              distance={distance}
            />
          ) : (
            <div
              className="flex items-center justify-center"
              style={{ height: 500 }}
            >
              <div className="text-center">
                <Loader2 className="h-10 w-10 animate-spin mx-auto mb-3 text-primary" />
                <p className="text-sm text-muted-foreground">Ожидание первой 3D точки...</p>
              </div>
            </div>
          )}
        </div>

        {/* Status */}
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
              <Badge variant="default" className="bg-green-500/20 text-green-500 border-green-500/30 text-[10px]">
                {processingTime.toFixed(1)} с
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
