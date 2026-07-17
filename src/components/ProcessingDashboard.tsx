import { useEffect, useMemo, useState } from "react";
import { Activity, Clock, Cpu, MapPin, Radio, Timer } from "lucide-react";
import { Progress } from "@/components/ui/progress";
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
};

const STAGES = [
  { id: "upload", label: "Загрузка" },
  { id: "gpu", label: "R³ GPU" },
  { id: "lingbot", label: "LingBot" },
  { id: "map", label: "План" },
  { id: "done", label: "Готово" },
] as const;

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

function resolveStage(
  stage: string | undefined,
  status: string | undefined,
  message: string,
  progress: number,
): string {
  if (stage) return stage;
  const text = (message || "").toLowerCase();
  const state = (status || "").toLowerCase();
  if (progress >= 100 || state === "completed") return "done";
  if (state === "error" || state === "failed") return "error";
  if (text.includes("загруз") || state.includes("upload")) return "upload";
  if (text.includes("lingbot")) return "lingbot";
  if (text.includes("план") || text.includes("map")) return "map";
  if (text.includes("gpu") || text.includes("r³") || text.includes("r3")) return "gpu";
  return "processing";
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
}: ProcessingDashboardProps) {
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const liveElapsed = useMemo(() => {
    if (elapsedSeconds != null && Number.isFinite(elapsedSeconds)) {
      return elapsedSeconds;
    }
    if (startedAtMs != null && Number.isFinite(startedAtMs)) {
      return Math.max(0, (nowMs - startedAtMs) / 1000);
    }
    return null;
  }, [elapsedSeconds, startedAtMs, nowMs]);

  const safeProgress = Math.max(0, Math.min(100, Math.round(progress || 0)));
  const activeStage = resolveStage(stage, status, message, safeProgress);
  const methodLabel =
    method === "r3" ? "R³ production" : method === "lingbot" ? "LingBot-Map" : "SLAM";

  const stageIndex = Math.max(
    0,
    STAGES.findIndex((item) => item.id === activeStage),
  );

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
              {safeProgress}
              <span className="ml-1 text-lg text-cyan-500/80">%</span>
            </div>
            <div className="mt-1 text-[11px] uppercase tracking-[0.16em] text-slate-500">
              завершено
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <Progress
            value={safeProgress}
            className="h-3 overflow-hidden rounded-full bg-slate-800/80 [&>div]:bg-[linear-gradient(90deg,#22d3ee,#2dd4bf,#67e8f9)] [&>div]:transition-[transform] [&>div]:duration-700"
          />
          <div className="flex items-center justify-between gap-3 text-[11px] text-slate-500">
            <span className="inline-flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-cyan-400/80" />
              {methodLabel}
            </span>
            {batchTotal && batchTotal > 1 ? (
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
              {safeProgress >= 100 ? "00:00" : formatClock(etaSeconds)}
            </div>
          </div>
          <div className="col-span-2 rounded-xl border border-white/5 bg-black/25 px-3 py-3 sm:col-span-1">
            <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.16em] text-slate-500">
              <MapPin className="h-3.5 w-3.5" />
              Стадия
            </div>
            <div className="truncate font-mono text-sm text-slate-100">
              {STAGES.find((item) => item.id === activeStage)?.label
                || (activeStage === "error" ? "Ошибка" : "В работе")}
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

        <div className="grid grid-cols-5 gap-2">
          {STAGES.map((item, index) => {
            const reached = index <= stageIndex || safeProgress >= 100;
            const current = item.id === activeStage && safeProgress < 100;
            return (
              <div
                key={item.id}
                className={cn(
                  "rounded-lg border px-1.5 py-2 text-center transition-colors duration-500",
                  reached
                    ? "border-cyan-300/30 bg-cyan-400/10 text-cyan-100"
                    : "border-white/5 bg-black/20 text-slate-500",
                  current && "shadow-[0_0_20px_rgba(34,211,238,0.18)]",
                )}
              >
                <div className="mx-auto mb-1.5 h-1 w-8 overflow-hidden rounded-full bg-slate-700/80">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-700",
                      reached ? "w-full bg-cyan-300" : "w-0 bg-transparent",
                      current && "animate-pulse",
                    )}
                  />
                </div>
                <div className="text-[10px] leading-tight">{item.label}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
