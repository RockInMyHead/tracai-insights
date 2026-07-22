import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, Clock, Cpu, MapPin, Radio, Timer } from "lucide-react";
import { cn } from "@/lib/utils";

export type ProcessingDashboardProps = {
  ownerName?: string;
  method?: "r3" | "lingbot" | "slam";
  progress: number;
  message: string;
  status?: string;
  stage?: string;
  elapsedSeconds?: number | null;
  etaSeconds?: number | null;
  batchIndex?: number;
  batchTotal?: number;
  startedAtMs?: number | null;
  /** Completed stage durations in seconds (from server). */
  stageTimings?: Record<string, number> | null;
  /** Live seconds for the currently active stage. */
  stageCurrentSeconds?: number | null;
  /** 0..1 fraction inside the active stage (from server). */
  stageFraction?: number | null;
  /** Optional live GPU frame counters for the R³ stage. */
  gpuFramesDone?: number | null;
  gpuFramesTotal?: number | null;
  /** Frames extracted from video before R³ writes camera poses. */
  gpuFramesExtracted?: number | null;
};

const STAGES = [
  { id: "upload", label: "Загрузка", span: [0, 6] as const },
  { id: "gpu", label: "R³ GPU", span: [6, 70] as const },
  { id: "trajectory", label: "Сборка", span: [70, 78] as const },
  { id: "lingbot", label: "LingBot", span: [78, 92] as const },
  { id: "map", label: "План", span: [92, 99] as const },
  { id: "done", label: "Готово", span: [100, 100] as const },
] as const;

type StageId = (typeof STAGES)[number]["id"];

function formatClock(totalSeconds: number | null | undefined): string {
  if (totalSeconds == null || !Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "--:--";
  }
  const rounded = Math.floor(totalSeconds);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const seconds = rounded % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function formatStageSeconds(totalSeconds: number | null | undefined): string {
  if (totalSeconds == null || !Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "—";
  }
  if (totalSeconds < 60) {
    return `${Math.floor(totalSeconds)}с`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function normalizeStage(
  stage: string | undefined,
  status: string | undefined,
  message: string,
  progress: number,
): StageId | "error" {
  const raw = (stage || "").toLowerCase();
  const text = (message || "").toLowerCase();
  const state = (status || "").toLowerCase();
  if (progress >= 100 || state === "completed" || raw === "done") return "done";
  if (state === "error" || state === "failed" || raw === "error") return "error";

  // Trust explicit server stage before message heuristics (GPU messages often mention "план").
  if (raw === "upload" || raw === "queued") return "upload";
  if (raw === "gpu" || state.includes("gpu")) return "gpu";
  if (raw === "trajectory") return "trajectory";
  if (raw === "lingbot" || state.includes("lingbot")) return "lingbot";
  if (raw === "map") return "map";
  if (STAGES.some((item) => item.id === raw)) return raw as StageId;

  if (text.includes("загруз") || state.includes("upload")) return "upload";
  if (text.includes("lingbot")) return "lingbot";
  if (text.includes("сборк") || text.includes("траект")) return "trajectory";
  if (text.includes("gpu") || text.includes("r³") || text.includes("r3") || text.includes("кадр")) {
    return "gpu";
  }
  if (text.includes("kerama") || (text.includes("план") && !text.includes("черновик"))) return "map";
  return "gpu";
}

function progressFromStage(stage: StageId | "error", fraction: number): number {
  if (stage === "error") return 0;
  const item = STAGES.find((entry) => entry.id === stage);
  if (!item) return Math.round(Math.max(0, Math.min(100, fraction * 100)));
  const [lo, hi] = item.span;
  const frac = Math.max(0, Math.min(1, fraction));
  return Math.round(lo + (hi - lo) * frac);
}

export default function ProcessingDashboard({
  ownerName,
  method = "r3",
  progress,
  message,
  status,
  stage,
  elapsedSeconds,
  etaSeconds,
  batchIndex,
  batchTotal,
  startedAtMs,
  stageTimings,
  stageCurrentSeconds,
  stageFraction,
  gpuFramesDone,
  gpuFramesTotal,
  gpuFramesExtracted,
}: ProcessingDashboardProps) {
  const [nowMs, setNowMs] = useState(() => Date.now());
  const stageTickRef = useRef<{ stage: string; seconds: number; at: number } | null>(null);
  const peakProgressRef = useRef(0);
  const runKeyRef = useRef<number | null>(startedAtMs ?? null);

  useEffect(() => {
    if (startedAtMs != null && startedAtMs !== runKeyRef.current) {
      runKeyRef.current = startedAtMs;
      peakProgressRef.current = 0;
      stageTickRef.current = null;
    }
  }, [startedAtMs]);

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 250);
    return () => window.clearInterval(timer);
  }, []);

  const activeStage = normalizeStage(stage, status, message, progress || 0);

  const resolvedFraction = useMemo(() => {
    if (activeStage === "done") return 1;
    if (activeStage === "gpu" && gpuFramesTotal != null && gpuFramesTotal > 0) {
      const total = gpuFramesTotal;
      const done = gpuFramesDone || 0;
      const extracted = gpuFramesExtracted || 0;
      if (done > 0 && Number.isFinite(done)) {
        return Math.max(0, Math.min(0.97, done / total));
      }
      if (extracted > 0 && Number.isFinite(extracted)) {
        const prep = Math.min(0.08, (extracted / total) * 0.08);
        const warmup = stageFraction != null && Number.isFinite(stageFraction)
          ? Math.max(0, Math.min(0.32, stageFraction))
          : 0.12;
        return Math.max(0.05, Math.min(0.4, prep + warmup));
      }
    }
    if (stageFraction != null && Number.isFinite(stageFraction)) {
      return Math.max(0, Math.min(1, stageFraction));
    }
    return 0.05;
  }, [activeStage, gpuFramesDone, gpuFramesExtracted, gpuFramesTotal, stageFraction]);

  const pipelineProgress = useMemo(() => {
    const fromStage = progressFromStage(activeStage, resolvedFraction);
    const fromServer = Math.max(0, Math.min(100, Math.round(progress || 0)));
    const next = Math.max(fromStage, fromServer);
    const state = (status || "").toLowerCase();
    const isActiveRun = state.includes("processing") || state === "queued";
    if (isActiveRun && fromServer < peakProgressRef.current - 15) {
      peakProgressRef.current = fromServer;
    } else {
      peakProgressRef.current = Math.max(peakProgressRef.current, next);
    }
    if (activeStage === "error") return fromServer;
    if (activeStage === "done") return 100;
    return isActiveRun ? Math.max(fromServer, Math.min(peakProgressRef.current, next)) : peakProgressRef.current;
  }, [activeStage, resolvedFraction, progress, status]);

  useEffect(() => {
    if (activeStage === "done" || activeStage === "error") {
      peakProgressRef.current = activeStage === "done" ? 100 : 0;
    }
  }, [activeStage]);

  useEffect(() => {
    if (
      stageCurrentSeconds != null
      && Number.isFinite(stageCurrentSeconds)
      && activeStage
      && activeStage !== "done"
      && activeStage !== "error"
    ) {
      stageTickRef.current = {
        stage: activeStage,
        seconds: stageCurrentSeconds,
        at: Date.now(),
      };
    }
  }, [stageCurrentSeconds, activeStage]);

  const liveElapsed = useMemo(() => {
    if (elapsedSeconds != null && Number.isFinite(elapsedSeconds)) {
      return elapsedSeconds;
    }
    if (startedAtMs != null && Number.isFinite(startedAtMs)) {
      return Math.max(0, (nowMs - startedAtMs) / 1000);
    }
    return null;
  }, [elapsedSeconds, startedAtMs, nowMs]);

  const liveStageSeconds = useMemo(() => {
    const tick = stageTickRef.current;
    if (
      tick
      && tick.stage === activeStage
      && activeStage !== "done"
      && activeStage !== "error"
    ) {
      return Math.max(0, tick.seconds + (nowMs - tick.at) / 1000);
    }
    if (
      stageCurrentSeconds != null
      && Number.isFinite(stageCurrentSeconds)
      && activeStage !== "done"
    ) {
      return stageCurrentSeconds;
    }
    return null;
  }, [nowMs, activeStage, stageCurrentSeconds]);

  const methodLabel =
    method === "r3" ? "R³ production" : method === "lingbot" ? "LingBot-Map" : "SLAM";

  const stageIndex = Math.max(
    0,
    STAGES.findIndex((item) => item.id === activeStage),
  );

  const displayTimings = useMemo(() => {
    const base: Record<string, number> = { ...(stageTimings || {}) };
    if (
      activeStage
      && activeStage !== "done"
      && activeStage !== "error"
      && liveStageSeconds != null
    ) {
      base[activeStage] = liveStageSeconds;
    }
    return base;
  }, [stageTimings, liveStageSeconds, activeStage]);

  const stageFill = (item: (typeof STAGES)[number], index: number): number => {
    if (pipelineProgress >= 100 || activeStage === "done") return 100;
    if (index < stageIndex) return 100;
    if (index > stageIndex) return 0;
    return Math.round(resolvedFraction * 100);
  };

  return (
    <div className="processing-dashboard relative overflow-hidden rounded-2xl border border-cyan-400/20 bg-[linear-gradient(160deg,rgba(8,18,28,0.96),rgba(6,12,20,0.98)_45%,rgba(10,24,32,0.94))] p-5 shadow-[0_0_0_1px_rgba(34,211,238,0.06),0_24px_60px_rgba(0,0,0,0.45)]">
      <div className="pointer-events-none absolute -right-16 -top-20 h-56 w-56 rounded-full bg-cyan-400/10 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-24 -left-10 h-48 w-48 rounded-full bg-teal-500/10 blur-3xl" />
      <div className="pointer-events-none absolute inset-x-8 top-0 h-px bg-gradient-to-r from-transparent via-cyan-300/50 to-transparent" />

      <div className="relative flex flex-col gap-5">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-400/5 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-cyan-200/80">
              <Radio className="h-3.5 w-3.5 animate-pulse text-cyan-300" />
              Live pipeline
            </div>
            <div>
              <h3 className="font-mono text-lg font-semibold tracking-tight text-cyan-50">
                Обработка видео
              </h3>
              <p className="mt-1 text-sm text-slate-400">
                ИИ анализирует траекторию движения
                {ownerName ? (
                  <>
                    {" · "}
                    <span className="text-cyan-200">{ownerName}</span>
                  </>
                ) : null}
              </p>
            </div>
          </div>

          <div className="text-right">
            <div className="font-mono text-4xl font-semibold leading-none tracking-tight text-cyan-300 tabular-nums">
              {pipelineProgress}
              <span className="ml-1 text-lg text-cyan-500/80">%</span>
            </div>
            <div className="mt-1 text-[11px] uppercase tracking-[0.16em] text-slate-500">
              по стадиям
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex h-3 overflow-hidden rounded-full bg-slate-800/80">
            {STAGES.filter((item) => item.id !== "done").map((item, index) => {
              const [lo, hi] = item.span;
              const weight = Math.max(hi - lo, 1);
              const fill = stageFill(item, index);
              const current = item.id === activeStage && pipelineProgress < 100;
              return (
                <div
                  key={item.id}
                  className="relative h-full border-r border-slate-900/40 last:border-r-0"
                  style={{ flexGrow: weight, flexBasis: 0 }}
                  title={`${item.label}: ${fill}%`}
                >
                  <div
                    className={cn(
                      "h-full transition-all duration-700",
                      current
                        ? "bg-[linear-gradient(90deg,#22d3ee,#67e8f9)]"
                        : fill > 0
                          ? "bg-cyan-400/80"
                          : "bg-transparent",
                    )}
                    style={{ width: `${fill}%` }}
                  />
                </div>
              );
            })}
          </div>
          <div className="flex items-center justify-between gap-3 text-[11px] text-slate-500">
            <span className="inline-flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-cyan-400/80" />
              {methodLabel}
            </span>
            {activeStage === "gpu" && gpuFramesTotal != null ? (
              <span className="font-mono text-cyan-300/90 tabular-nums">
                {(gpuFramesDone || 0) > 0
                  ? `кадры ${gpuFramesDone}/${gpuFramesTotal}`
                  : (gpuFramesExtracted || 0) > 0
                    ? `подготовлено ${gpuFramesExtracted}/${gpuFramesTotal}`
                    : `кадры 0/${gpuFramesTotal}`}
              </span>
            ) : batchTotal && batchTotal > 1 ? (
              <span>
                Видео {Math.min((batchIndex || 0) + 1, batchTotal)} / {batchTotal}
              </span>
            ) : (
              <span>Production Kerama</span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <div className="rounded-xl border border-white/5 bg-black/25 px-3 py-3">
            <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.16em] text-slate-500">
              <Clock className="h-3.5 w-3.5" />
              Прошло
            </div>
            <div className="font-mono text-xl text-cyan-100 tabular-nums">
              {formatClock(liveElapsed)}
            </div>
          </div>
          <div className="rounded-xl border border-white/5 bg-black/25 px-3 py-3">
            <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.16em] text-slate-500">
              <Timer className="h-3.5 w-3.5" />
              Осталось ≈
            </div>
            <div className="font-mono text-xl text-teal-100 tabular-nums">
              {pipelineProgress >= 100 ? "00:00" : formatClock(etaSeconds)}
            </div>
          </div>
          <div className="col-span-2 rounded-xl border border-white/5 bg-black/25 px-3 py-3 sm:col-span-1">
            <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.16em] text-slate-500">
              <MapPin className="h-3.5 w-3.5" />
              Стадия
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <div className="truncate font-mono text-sm text-slate-100">
                {STAGES.find((item) => item.id === activeStage)?.label
                  || (activeStage === "error" ? "Ошибка" : "В работе")}
              </div>
              <div className="shrink-0 font-mono text-sm text-cyan-300 tabular-nums">
                {formatStageSeconds(
                  activeStage === "done"
                    ? displayTimings.done
                    : liveStageSeconds ?? displayTimings[activeStage],
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-start gap-2 rounded-xl border border-cyan-400/10 bg-cyan-400/[0.04] px-3 py-3">
          <Activity className="mt-0.5 h-4 w-4 shrink-0 text-cyan-300" />
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
              Текущий шаг
            </div>
            <div className="mt-1 truncate text-sm text-slate-100">
              {message || "Ожидание статуса с сервера..."}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
          {STAGES.map((item, index) => {
            const fill = stageFill(item, index);
            const reached = fill >= 100 || pipelineProgress >= 100;
            const current = item.id === activeStage && pipelineProgress < 100;
            const seconds = displayTimings[item.id];
            const showTime = fill > 0 || current;
            return (
              <div
                key={item.id}
                className={cn(
                  "rounded-lg border px-1.5 py-2 text-center transition-colors duration-500",
                  fill > 0
                    ? "border-cyan-300/30 bg-cyan-400/10 text-cyan-100"
                    : "border-white/5 bg-black/20 text-slate-500",
                  current && "shadow-[0_0_20px_rgba(34,211,238,0.18)]",
                )}
              >
                <div className="mx-auto mb-1.5 h-1 w-8 overflow-hidden rounded-full bg-slate-700/80">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-700",
                      fill > 0 ? "bg-cyan-300" : "bg-transparent",
                      current && "animate-pulse",
                    )}
                    style={{ width: `${fill}%` }}
                  />
                </div>
                <div className="text-[10px] leading-tight">{item.label}</div>
                <div
                  className={cn(
                    "mt-1 font-mono text-[11px] tabular-nums",
                    current ? "text-cyan-300" : showTime ? "text-slate-300" : "text-slate-600",
                  )}
                >
                  {current ? `${Math.round(resolvedFraction * 100)}%` : showTime ? formatStageSeconds(seconds) : "—"}
                  {current ? <span className="ml-0.5 animate-pulse">·</span> : null}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
