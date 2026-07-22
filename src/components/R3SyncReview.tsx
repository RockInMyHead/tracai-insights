import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import TrajectoryMap, {
  type TrajectoryData,
  type TrajectoryPoint,
} from "@/components/TrajectoryMap";
import { finiteNum } from "@/lib/numbers";

type Props = {
  videoUrl: string;
  trajectories: TrajectoryData[];
  stats?: Record<string, unknown> | null;
  floorPlan?: string | null;
  drawnPlan?: unknown[] | null;
  referencePoint?: { x: number; y: number } | null;
  directionPoint?: { x: number; y: number } | null;
  setDirectionMode?: boolean;
  onSetDirectionModeChange?: (enabled: boolean) => void;
  onDirectionPointSet?: (point: { x: number; y: number }) => void;
};

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];

function normalizePoint(point: unknown): TrajectoryPoint | null {
  if (Array.isArray(point)) {
    return { x: finiteNum(point[0]), y: finiteNum(point[1]), z: finiteNum(point[2]) };
  }
  if (point && typeof point === "object") {
    const o = point as Record<string, unknown> & { 0?: unknown; 1?: unknown; 2?: unknown };
    return {
      x: finiteNum(o.x ?? o[0]),
      y: finiteNum(o.y ?? o[1]),
      z: finiteNum(o.z ?? o[2]),
    };
  }
  return null;
}

function normalizeTrajectory(points: unknown): TrajectoryPoint[] {
  if (!Array.isArray(points)) return [];
  return points.map(normalizePoint).filter((p): p is TrajectoryPoint => Boolean(p));
}

function DenseTrajectoryPanel({
  points,
  currentIndex,
}: {
  points: TrajectoryPoint[];
  currentIndex: number;
}) {
  const model = useMemo(() => {
    if (points.length === 0) return null;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const point of points) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }
    const spanX = Math.max(maxX - minX, 1e-6);
    const spanY = Math.max(maxY - minY, 1e-6);
    const pad = 6;
    const scale = Math.min((100 - pad * 2) / spanX, (100 - pad * 2) / spanY);
    const toSvg = (point: TrajectoryPoint) => ({
      x: (point.x - minX) * scale + pad,
      y: 100 - ((point.y - minY) * scale + pad),
    });
    return { toSvg };
  }, [points]);

  const visibleCount = Math.max(1, Math.min(currentIndex + 1, points.length));
  const visiblePoints = points.slice(0, visibleCount);
  const path = model
    ? visiblePoints
        .map((point, index) => {
          const p = model.toSvg(point);
          return `${index === 0 ? "M" : "L"} ${p.x.toFixed(3)} ${p.y.toFixed(3)}`;
        })
        .join(" ")
    : "";
  const current = model && points[currentIndex] ? model.toSvg(points[currentIndex]) : null;

  return (
    <div className="relative h-[318px] overflow-hidden rounded-lg bg-[#060a16]">
      <svg viewBox="0 0 100 100" className="h-full w-full">
        <rect x="0" y="0" width="100" height="100" fill="#060a16" />
        {model &&
          points.map((point, index) => {
            const p = model.toSvg(point);
            const passed = index < visibleCount;
            return (
              <circle
                key={`${index}-${p.x.toFixed(2)}-${p.y.toFixed(2)}`}
                cx={p.x}
                cy={p.y}
                r={passed ? 0.28 : 0.18}
                fill={passed ? "#22d3ee" : "#64748b"}
                opacity={passed ? 0.8 : 0.28}
              />
            );
          })}
        {path && (
          <path
            d={path}
            fill="none"
            stroke="#60a5fa"
            strokeWidth="0.75"
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity="0.95"
          />
        )}
        {current && (
          <circle cx={current.x} cy={current.y} r="1.5" fill="#f97316" stroke="#fff" strokeWidth="0.35" />
        )}
      </svg>
      <div className="absolute left-3 top-3 rounded-md border border-white/10 bg-black/50 px-3 py-2 text-xs text-white backdrop-blur">
        <div>Точек: {points.length.toLocaleString("ru-RU")}</div>
        <div>Кадр/точка: {visibleCount.toLocaleString("ru-RU")}</div>
      </div>
    </div>
  );
}

export default function R3SyncReview({
  videoUrl,
  trajectories,
  stats,
  floorPlan,
  drawnPlan,
  referencePoint,
  directionPoint,
  setDirectionMode,
  onSetDirectionModeChange,
  onDirectionPointSet,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const routeAnimRef = useRef<number | null>(null);
  const [videoReloadKey, setVideoReloadKey] = useState(0);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [routeIndex, setRouteIndex] = useState(0);
  const [routeMode, setRouteMode] = useState(true);
  const method = String(trajectories[0]?.method || stats?.method || stats?.algorithm || "").toLowerCase();
  const algorithmName = method.includes("lingbot")
    ? "LingBot-Map"
    : method.includes("r3") || method.includes("r³")
      ? "R³"
      : "траектории";

  const densePoints = useMemo(() => {
    const first = trajectories[0];
    return first ? normalizeTrajectory(first.trajectory) : [];
  }, [trajectories]);

  const trajectorySignature = useMemo(() => {
    if (densePoints.length === 0) return "empty";
    const first = densePoints[0];
    const last = densePoints[densePoints.length - 1];
    return `${densePoints.length}:${first.x.toFixed(2)},${first.y.toFixed(2)}:${last.x.toFixed(2)},${last.y.toFixed(2)}`;
  }, [densePoints]);

  const videoCanDrive = duration > 0 && !videoError && !routeMode;

  const currentIndex = useMemo(() => {
    if (densePoints.length === 0) return 0;
    if (videoCanDrive && isPlaying) {
      const ratio = duration > 0 ? currentTime / duration : 0;
      return Math.max(0, Math.min(densePoints.length - 1, Math.floor(ratio * densePoints.length)));
    }
    return Math.max(0, Math.min(densePoints.length - 1, routeIndex));
  }, [currentTime, densePoints.length, duration, isPlaying, routeIndex, videoCanDrive]);

  const stopRouteAnimation = () => {
    if (routeAnimRef.current !== null) {
      cancelAnimationFrame(routeAnimRef.current);
      routeAnimRef.current = null;
    }
  };

  const startRouteAnimation = (fromIndex = 0) => {
    stopRouteAnimation();
    if (densePoints.length < 2) {
      setRouteIndex(0);
      setIsPlaying(false);
      return;
    }
    setRouteMode(true);
    setIsPlaying(true);
    const startIndex = Math.max(0, Math.min(fromIndex, densePoints.length - 1));
    const remaining = Math.max(1, densePoints.length - 1 - startIndex);
    // ~28ms per point, clamped to a pleasant 2.5–14s reveal.
    const durationMs = Math.min(14000, Math.max(2500, remaining * 28)) / Math.max(playbackRate, 0.25);
    const startedAt = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - startedAt) / durationMs);
      const eased = 1 - (1 - t) * (1 - t);
      const next = startIndex + Math.floor(eased * remaining);
      setRouteIndex(next);
      if (t < 1) {
        routeAnimRef.current = requestAnimationFrame(tick);
      } else {
        routeAnimRef.current = null;
        setIsPlaying(false);
        setRouteIndex(densePoints.length - 1);
      }
    };
    setRouteIndex(startIndex);
    routeAnimRef.current = requestAnimationFrame(tick);
  };

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  // When a new analysis arrives, smoothly draw the route on the floor plan.
  useEffect(() => {
    stopRouteAnimation();
    videoRef.current?.pause();
    setIsPlaying(false);
    setCurrentTime(0);
    if (densePoints.length < 2) {
      setRouteIndex(0);
      return;
    }
    const timer = window.setTimeout(() => startRouteAnimation(0), 120);
    return () => {
      window.clearTimeout(timer);
      stopRouteAnimation();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trajectorySignature]);

  useEffect(() => () => stopRouteAnimation(), []);

  const togglePlay = () => {
    const video = videoRef.current;
    if (isPlaying) {
      stopRouteAnimation();
      video?.pause();
      setIsPlaying(false);
      return;
    }
    if (video && duration > 0 && !videoError && !routeMode) {
      video.play().catch(() => {
        startRouteAnimation(routeIndex);
      });
      return;
    }
    const from = routeIndex >= densePoints.length - 1 ? 0 : routeIndex;
    startRouteAnimation(from);
  };

  const seekTo = (seconds: number) => {
    const video = videoRef.current;
    if (!video || !(duration > 0)) return;
    stopRouteAnimation();
    setRouteMode(false);
    const next = Math.max(0, Math.min(seconds, duration || video.duration || 0));
    video.currentTime = next;
    setCurrentTime(next);
    if (densePoints.length > 0) {
      const ratio = duration > 0 ? next / duration : 0;
      setRouteIndex(Math.max(0, Math.min(densePoints.length - 1, Math.floor(ratio * densePoints.length))));
    }
  };

  const seekToPoint = (index: number) => {
    stopRouteAnimation();
    setRouteMode(true);
    setIsPlaying(false);
    const next = Math.max(0, Math.min(index, Math.max(densePoints.length - 1, 0)));
    setRouteIndex(next);
    if (duration > 0 && videoRef.current) {
      const ratio = densePoints.length > 1 ? next / (densePoints.length - 1) : 0;
      const seconds = ratio * duration;
      videoRef.current.currentTime = seconds;
      setCurrentTime(seconds);
    }
  };

  const reset = () => {
    stopRouteAnimation();
    setIsPlaying(false);
    setRouteMode(true);
    setRouteIndex(0);
    setCurrentTime(0);
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  };

  useEffect(() => {
    setVideoError(null);
    setCurrentTime(0);
    setDuration(0);
  }, [videoUrl]);

  return (
    <Card className="border-primary/20">
      <CardHeader className="space-y-2 pb-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <CardTitle className="text-lg">Проверка {algorithmName}: план, видео и точки</CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            <Button type="button" size="sm" onClick={togglePlay}>
              {isPlaying ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
              {isPlaying ? "Пауза" : "Рисовать маршрут"}
            </Button>
            <Button type="button" size="sm" variant="outline" onClick={reset}>
              <RotateCcw className="mr-2 h-4 w-4" />
              Сброс
            </Button>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          После загрузки анализа маршрут плавно прорисовывается на плане. Кнопка «Рисовать маршрут»
          повторяет анимацию; если preview видео доступен, можно синхронизировать ход с плеером.
        </p>

        <div className="grid gap-3 lg:grid-cols-[1fr_auto] lg:items-center">
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{videoCanDrive ? `${currentTime.toFixed(1)} c` : "анимация"}</span>
              <span>
                точка {Math.min(currentIndex + 1, densePoints.length).toLocaleString("ru-RU")} /{" "}
                {densePoints.length.toLocaleString("ru-RU")}
              </span>
              <span>
                {videoCanDrive
                  ? `${duration.toFixed(1)} c`
                  : `${Math.round(((currentIndex + 1) / Math.max(densePoints.length, 1)) * 100)}%`}
              </span>
            </div>
            <Slider
              value={[videoCanDrive && !routeMode ? currentTime : currentIndex]}
              min={0}
              max={videoCanDrive && !routeMode ? Math.max(duration, 0.01) : Math.max(densePoints.length - 1, 1)}
              step={videoCanDrive && !routeMode ? 0.05 : 1}
              onValueChange={(value) => {
                const next = value[0] ?? 0;
                if (videoCanDrive && !routeMode) seekTo(next);
                else seekToPoint(Math.round(next));
              }}
            />
          </div>
          <div className="flex flex-wrap items-center gap-1">
            <span className="mr-1 text-xs text-muted-foreground">Скорость</span>
            {SPEEDS.map((speed) => (
              <Button
                key={speed}
                type="button"
                size="sm"
                variant={playbackRate === speed ? "default" : "outline"}
                className="h-8 px-2"
                onClick={() => setPlaybackRate(speed)}
              >
                {speed}x
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <div className="grid gap-4 xl:grid-cols-3">
          <Card className="overflow-hidden">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">1. Чертеж с текущей траекторией</CardTitle>
            </CardHeader>
            <CardContent className="h-[360px] p-0">
              <TrajectoryMap
                trajectories={trajectories}
                stats={stats ?? undefined}
                floorPlan={floorPlan}
                drawnPlan={drawnPlan}
                referencePoint={referencePoint}
                directionPoint={directionPoint}
                playbackPointLimit={currentIndex + 1}
                compactMode
                setDirectionMode={setDirectionMode}
                onSetDirectionModeChange={onSetDirectionModeChange}
                onDirectionPointSet={onDirectionPointSet}
              />
            </CardContent>
          </Card>

          <Card className="overflow-hidden">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">2. Видео</CardTitle>
            </CardHeader>
            <CardContent className="p-3">
              <div className="overflow-hidden rounded-lg bg-black">
                <video
                  key={`${videoUrl}-${videoReloadKey}`}
                  ref={videoRef}
                  src={videoUrl}
                  className="h-[318px] w-full object-contain"
                  controls
                  preload="metadata"
                  onLoadedMetadata={(event) => {
                    const video = event.currentTarget;
                    setDuration(Number.isFinite(video.duration) ? video.duration : 0);
                    video.playbackRate = playbackRate;
                    setVideoError(null);
                  }}
                  onTimeUpdate={(event) => {
                    if (!routeMode) setCurrentTime(event.currentTarget.currentTime);
                  }}
                  onPlay={() => {
                    stopRouteAnimation();
                    setRouteMode(false);
                    setIsPlaying(true);
                  }}
                  onPause={() => {
                    if (!routeMode) setIsPlaying(false);
                  }}
                  onEnded={() => {
                    if (!routeMode) setIsPlaying(false);
                  }}
                  onError={() => {
                    setIsPlaying(false);
                    setVideoError("Preview видео недоступен — маршрут рисуется без синхронизации с плеером.");
                    window.setTimeout(() => setVideoReloadKey((value) => value + 1), 3000);
                  }}
                >
                  Ваш браузер не поддерживает видео.
                </video>
                {videoError && (
                  <div className="border-t border-white/10 bg-black px-3 py-2 text-xs text-white/70">
                    {videoError}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="overflow-hidden">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">3. {algorithmName} точки и след</CardTitle>
            </CardHeader>
            <CardContent className="p-3">
              <DenseTrajectoryPanel points={densePoints} currentIndex={currentIndex} />
            </CardContent>
          </Card>
        </div>
      </CardContent>
    </Card>
  );
}
