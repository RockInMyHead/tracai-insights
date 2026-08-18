import { type MouseEvent, type PointerEvent, type WheelEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import TrajectoryMap, { type TrajectoryData, type TurnPoint } from "@/components/TrajectoryMap";
import { apiClient, type VideoAnalysisResult, type VideoListItem } from "@/lib/api";
import { getCameraImportAPI, type CameraImportedVideo, type CameraImportProgress } from "@/lib/cameraImport";
import { Camera, ChevronDown, Download, History, Loader2, MapPinned, RotateCcw, Upload, ZoomIn, ZoomOut } from "lucide-react";
import { toast } from "sonner";

const FLOORPLAN_URL = `${import.meta.env.BASE_URL}floorplans/kerama-marazzi-2025.png`;
const CAMERA_OWNER = "Экшен-камера";
const PLAN_ZOOM_MIN = 1;
const PLAN_ZOOM_MAX = 8;
const PLAN_DRAG_THRESHOLD_PX = 6;

type DesktopState = "ready" | "looking" | "copying" | "processing" | "done" | "needs_camera" | "error";
type ProcessingMode = "online" | "local_gpu" | "local_cpu";
type ProcessingResolution = Awaited<ReturnType<NonNullable<Window["trackai"]>["processing"]["resolveMode"]>>;
type PlanPoint = { x: number; y: number };
type VideoProcessingProgress = {
  videoId: string;
  fileName: string;
  index: number;
  total: number;
  percent: number;
  status: "queued" | "processing" | "done" | "error" | "hint";
  message?: string;
};

function getDesktopBridge() {
  return (window as unknown as { trackai?: Window["trackai"] }).trackai;
}

async function getDesktopUploadedVideosList() {
  const bridge = getDesktopBridge();
  const fromMain = await bridge?.desktopApi?.getUploadedVideos?.();
  if (fromMain) return fromMain as Awaited<ReturnType<typeof apiClient.getUploadedVideosList>>;
  return apiClient.getUploadedVideosList();
}

async function getDesktopVideoAnalysis(videoId: string) {
  const bridge = getDesktopBridge();
  const fromMain = await bridge?.desktopApi?.getVideoAnalysis?.(videoId);
  if (fromMain) return fromMain as VideoAnalysisResult;
  return apiClient.getVideoAnalysis(videoId);
}

async function getDesktopProcessingStatus(videoId: string) {
  const bridge = getDesktopBridge();
  const fromMain = await bridge?.desktopApi?.getProcessingStatus?.(videoId);
  if (fromMain) return fromMain as Awaited<ReturnType<typeof apiClient.getProcessingStatus>>;
  return apiClient.getProcessingStatus(videoId);
}

type ProcessingStatusResponse = Awaited<ReturnType<typeof apiClient.getProcessingStatus>>;

function isTransientDesktopApiError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /504|502|503|500|408|Gateway Time-out|Desktop API failed|не отвечает|fetch failed|network|timed out|aborted/i.test(message);
}

async function getDesktopProcessingStatusResilient(
  videoId: string,
  onTransient?: (message: string) => void,
): Promise<ProcessingStatusResponse> {
  const maxAttempts = 8;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      return await getDesktopProcessingStatus(videoId);
    } catch (error) {
      if (!isTransientDesktopApiError(error) || attempt >= maxAttempts - 1) {
        throw error;
      }
      onTransient?.("TrackAI занят — продолжаем обработку...");
      await new Promise((resolve) => window.setTimeout(resolve, 2500 * (attempt + 1)));
    }
  }
  throw new Error("Не удалось получить статус обработки");
}

async function recoverTrajectoryFromAnalysis(videoId: string): Promise<VideoAnalysisResult["data"] | null> {
  try {
    const analysis = await getDesktopVideoAnalysis(videoId);
    return getDesktopTrajectory(analysis.data, videoId).length ? analysis.data : null;
  } catch {
    return null;
  }
}

async function waitForVideoReadyForAnalysis(
  videoId: string,
  fileName: string,
  isStopped: () => boolean,
  onTick?: (message: string) => void,
): Promise<void> {
  for (let attempt = 0; attempt < 90; attempt += 1) {
    if (isStopped()) return;
    try {
      const status = await getDesktopProcessingStatusResilient(videoId, onTick);
      const state = String(status.status || "").toLowerCase();
      const stage = String((status as { stage?: string }).stage || "").toLowerCase();
      if (["error", "failed"].includes(state)) {
        if (isDesktopGpuBusyError(status.message)) {
          onTick?.("TrackAI занят — продолжаем подготовку...");
        } else {
          throw new Error(formatDesktopUserFacingError(status.message));
        }
      } else if (
        ["uploaded", "queued", "completed", "done", "success"].includes(state)
        || stage === "queued"
        || /ожидает запуска r3/i.test(String(status.message || ""))
      ) {
        return;
      }
      onTick?.(formatDesktopProcessingMessage(status.message) || `Подготовка: ${fileName}`);
    } catch (error) {
      if (attempt >= 10 && error instanceof Error && /не принял|error|failed/i.test(error.message)) {
        throw error;
      }
      onTick?.("TrackAI перезапускается, ждём готовности...");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 2000));
  }
  throw new Error("TrackAI не подтвердил копирование за 3 минуты. Нажмите «Остановить выгрузку» и повторите «Загрузить».");
}

function getDesktopTrajectory(data: VideoAnalysisResult["data"], videoId: string): TrajectoryData[] {
  if (!data) return [];
  const graphFirst = data.graph_first_trajectory?.length
    ? data.graph_first_trajectory
    : undefined;
  const mapPoints = data.map_trajectory?.length ? data.map_trajectory : undefined;
  const planFallback = (!mapPoints && !graphFirst && data.plan_trajectory?.length)
    ? data.plan_trajectory
    : undefined;
  // Только привязка к графу / плану. Сырой R³ autofit на десктопе не показываем.
  const points = mapPoints || graphFirst || planFallback || [];
  if (points.length < 1) return [];
  const uncertaintyMarker = data.graph_first_uncertainty?.marker;
  const competingNextEdges = (
    data.graph_first_uncertainty?.competing_next_edges || []
  ).map((edge) => edge.points.map((point) => ({
    x: Number(point[0]) || 0,
    y: Number(point[1]) || 0,
    z: Number(point[2]) || 0,
  })));
  const approximate = Boolean((graphFirst && !mapPoints?.length) || planFallback);
  return [{
    trajectory: points.map((point) => ({ x: Number(point[0]) || 0, y: Number(point[1]) || 0, z: Number(point[2]) || 0 })),
    turnPoints: (data.map_turn_points || data.turn_points || []) as TurnPoint[],
    ownerName: CAMERA_OWNER,
    color: approximate ? "#f59e0b" : "#0f766e",
    videoId,
    method: data.method,
    mapAligned: Boolean(mapPoints?.length),
    r3AutoFitToPlan: false,
    uncertain: approximate,
    uncertaintyMarker: Array.isArray(uncertaintyMarker)
      ? {
          x: Number(uncertaintyMarker[0]) || 0,
          y: Number(uncertaintyMarker[1]) || 0,
          z: Number(uncertaintyMarker[2]) || 0,
        }
      : undefined,
    competingNextEdges,
  }];
}

const DESKTOP_PLAN_FAILURE = /граф|конус|operator_start|turn_event|constraint_solution|map_trajectory|привязк|v41|v43|topology|ambiguous_or_off_heading/i;
const DESKTOP_INTERNAL_FAILURE = /GPU Worker|R³/i;

function isDesktopGpuBusyError(raw: string | undefined): boolean {
  return /GPU занят|gpu busy|already (running|executing)|занят:\s*на/i.test(raw || "");
}

function isDesktopPlanBindFailure(data: VideoAnalysisResult["data"] | undefined): boolean {
  if (!data) return false;
  if (getDesktopTrajectory(data, "probe").length) return false;
  const constraint = (
    data.floorplan_constraint
    || data.processing_stats?.floorplan_constraint
  ) as Record<string, unknown> | undefined;
  const reason = String(
    constraint?.reason
    || (constraint?.operator_start_edge as { reason?: string } | undefined)?.reason
    || "",
  );
  return Boolean(reason) || Boolean(data.method);
}

function formatDesktopProcessingMessage(raw: string | undefined): string {
  const text = (raw || "").trim();
  if (!text) return "Обрабатывается";
  if (isDesktopGpuBusyError(text)) {
    return "TrackAI занят — ждём очередь...";
  }
  if (/Kerama|план|маршрут|Сопоставление/i.test(text)) {
    return text.replace(/GPU|сервер/gi, "").trim() || "Сопоставление с планом...";
  }
  if (/R³|реконструк|кадр|gpu_processing|черновик/i.test(text)) {
    return "Восстановление траектории из видео...";
  }
  if (/upload|загруз|registered|копир/i.test(text)) {
    return "Копирование видео...";
  }
  if (/504|502|503|перегруз|не отвечает|timeout/i.test(text)) {
    return "Продолжаем обработку...";
  }
  if (DESKTOP_PLAN_FAILURE.test(text) || DESKTOP_INTERNAL_FAILURE.test(text)) {
    return "Сопоставление с планом...";
  }
  return text
    .replace(/сервер/gi, "TrackAI")
    .replace(/GPU[- ]?сервер/gi, "процессор")
    .replace(/на GPU/gi, "локально")
    .replace(/GPU Worker[^.]*\.?/gi, "")
    .trim() || "Обрабатывается";
}

function formatDesktopSoftPlanHint(): string {
  return "Уточните старт и направление на плане и загрузите видео снова.";
}

function isDesktopUserFacingFailure(message: string, data?: VideoAnalysisResult["data"]): boolean {
  if (isDesktopGpuBusyError(message)) return false;
  if (isDesktopPlanBindFailure(data)) return true;
  return DESKTOP_PLAN_FAILURE.test(message);
}

function formatDesktopUserFacingError(raw: string | undefined, data?: VideoAnalysisResult["data"]): string {
  if (isDesktopUserFacingFailure(raw || "", data)) {
    return formatDesktopSoftPlanHint();
  }
  let text = (raw || "").trim();
  if (!text) return formatDesktopSoftPlanHint();
  if (isDesktopGpuBusyError(text)) {
    return "TrackAI занят — ждём очередь...";
  }
  if (/failed to fetch|networkerror|504|502|503|перегруз/i.test(text)) {
    return "TrackAI временно занят — подождите и повторите загрузку.";
  }
  const gpuMatch = text.match(/GPU Worker R³ error \(HTTP \d+\):\s*(.+)/s);
  if (gpuMatch) text = gpuMatch[1].trim();
  text = text
    .replace(/^Ошибка R³:\s*/i, "")
    .replace(/сервер/gi, "TrackAI")
    .replace(/GPU[- ]?сервер/gi, "процессор")
    .trim();
  if (DESKTOP_PLAN_FAILURE.test(text)) {
    return formatDesktopSoftPlanHint();
  }
  return text || formatDesktopSoftPlanHint();
}

/** @deprecated use formatDesktopProcessingMessage / formatDesktopUserFacingError */
function formatServerProcessingError(raw: string | undefined): string {
  return formatDesktopUserFacingError(raw);
}

function isRetryableGpuVramError(message: string): boolean {
  return /VRAM|LingBot|CUDA out of memory/i.test(message);
}

function isRetryableGpuBusyError(message: string): boolean {
  return isDesktopGpuBusyError(message);
}

const GPU_BUSY_RETRY_WAIT_MS = 30_000;
const GPU_BUSY_MAX_RETRIES = 40;

function mergeUploadProgress(
  current: CameraImportProgress | null,
  next: CameraImportProgress,
): CameraImportProgress {
  const nextIndex = Number(next.index) || 0;
  const nextPercent = Number(next.percent) || 0;
  if (current) {
    const currentIndex = Number(current.index) || 0;
    // Устаревшее IPC-событие от предыдущего файла/повтора загрузки.
    if (nextIndex < currentIndex) {
      return current;
    }
    // Не перескакиваем на следующий файл, пока текущий не почти залит.
    if (nextIndex > currentIndex && current.percent < 95) {
      return current;
    }
    if (
      nextIndex === currentIndex
      && current.fileName === next.fileName
      && current.total === next.total
    ) {
      const percent = nextPercent <= 2 ? nextPercent : Math.max(current.percent, nextPercent);
      if (Math.abs(percent - current.percent) < 0.25 && current.phase === next.phase) {
        return current;
      }
      return { ...next, percent };
    }
  }
  return next;
}

export default function WindowsCameraDesktop() {
  const cameraImport = useMemo(() => getCameraImportAPI(), []);
  const [state, setState] = useState<DesktopState>("ready");
  const [message, setMessage] = useState("Подключите экшен-камеру и нажмите «Загрузить»");
  const [progress, setProgress] = useState<CameraImportProgress | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [logsDownloading, setLogsDownloading] = useState(false);
  const [history, setHistory] = useState<VideoListItem[]>([]);
  const [trajectories, setTrajectories] = useState<TrajectoryData[]>([]);
  const [stats, setStats] = useState<Record<string, unknown>>({});
  const [processingStatus, setProcessingStatus] = useState<ProcessingResolution | null>(null);
  const [processingProgress, setProcessingProgress] = useState<VideoProcessingProgress[]>([]);
  const [referencePoint, setReferencePoint] = useState<PlanPoint | null>(() => {
    try {
      const raw = localStorage.getItem("desktopReferencePoint");
      return raw ? JSON.parse(raw) as PlanPoint : null;
    } catch { return null; }
  });
  const [directionPoint, setDirectionPoint] = useState<PlanPoint | null>(() => {
    try {
      const raw = localStorage.getItem("desktopDirectionPoint");
      return raw ? JSON.parse(raw) as PlanPoint : null;
    } catch { return null; }
  });
  const [planPickMode, setPlanPickMode] = useState<"start" | "direction">("start");
  const [planZoom, setPlanZoom] = useState(1);
  const [planPan, setPlanPan] = useState({ x: 0, y: 0 });
  const planViewportRef = useRef<HTMLDivElement | null>(null);
  const planDragRef = useRef<{
    active: boolean;
    moved: boolean;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);
  const processingModeRef = useRef<ProcessingMode>("online");
  const analysisQueueRef = useRef<Promise<void>>(Promise.resolve());
  const enqueuedVideoIdsRef = useRef<Set<string>>(new Set());
  /** IDs registered during camera upload — for server progress polling before analysis starts. */
  const uploadedVideoIdsRef = useRef<Set<string>>(new Set());
  const hydratedVideoIdsRef = useRef<Set<string>>(new Set());
  const importStoppedRef = useRef(false);
  const analysisStoppedRef = useRef(false);
  const analysisQueueIdRef = useRef("");
  const analysisSequenceRef = useRef(0);
  const retriedVramVideoIdsRef = useRef<Set<string>>(new Set());
  const retriedPlanBindVideoIdsRef = useRef<Set<string>>(new Set());
  const sessionBatchIndexByVideoIdRef = useRef<Map<string, number>>(new Map());
  const sessionBatchCounterRef = useRef(0);
  const processingProgressRef = useRef(processingProgress);
  processingProgressRef.current = processingProgress;

  useEffect(() => {
    try {
      if (referencePoint) localStorage.setItem("desktopReferencePoint", JSON.stringify(referencePoint));
      else localStorage.removeItem("desktopReferencePoint");
    } catch { /* noop */ }
  }, [referencePoint]);

  useEffect(() => {
    try {
      if (directionPoint) localStorage.setItem("desktopDirectionPoint", JSON.stringify(directionPoint));
      else localStorage.removeItem("desktopDirectionPoint");
    } catch { /* noop */ }
  }, [directionPoint]);

  const upsertProcessingProgress = useCallback((next: VideoProcessingProgress) => {
    setProcessingProgress((current) => {
      const existingIndex = current.findIndex((item) => item.videoId === next.videoId);
      if (existingIndex === -1) return [...current, next];
      return current.map((item, index) => index === existingIndex ? { ...item, ...next } : item);
    });
  }, []);

  const resolveBatchSlot = useCallback((videoId: string): { index: number; total: number } => {
    if (!sessionBatchIndexByVideoIdRef.current.has(videoId)) {
      sessionBatchCounterRef.current += 1;
      sessionBatchIndexByVideoIdRef.current.set(videoId, sessionBatchCounterRef.current);
    }
    const total = Math.max(
      sessionBatchCounterRef.current,
      uploadedVideoIdsRef.current.size,
      enqueuedVideoIdsRef.current.size,
      processingProgressRef.current.length,
    );
    return {
      index: sessionBatchIndexByVideoIdRef.current.get(videoId) || 1,
      total: Math.max(total, 1),
    };
  }, []);

  const hydrateCompletedVideo = useCallback(async (videoId: string) => {
    if (hydratedVideoIdsRef.current.has(videoId)) return;
    hydratedVideoIdsRef.current.add(videoId);
    try {
      const analysis = await getDesktopVideoAnalysis(videoId);
      const nextTrajectory = getDesktopTrajectory(analysis.data, videoId);
      if (!nextTrajectory.length) {
        hydratedVideoIdsRef.current.delete(videoId);
        setMessage(describeGraphBindFailure(analysis.data) || "Готовый анализ найден, но траектория по графу плана отсутствует.");
        return;
      }
      setTrajectories((current) => (
        current.some((item) => item.videoId === videoId)
          ? current
          : [...current, ...nextTrajectory]
      ));
      setStats((analysis.data?.processing_stats || {}) as Record<string, unknown>);
      setState("done");
      setMessage("Готово. Траектория показана на плане Kerama Marazzi.");
    } catch {
      hydratedVideoIdsRef.current.delete(videoId);
    }
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const openVideo = params.get("openVideo") || localStorage.getItem("desktopOpenVideoId");
    if (!openVideo) return;
    try {
      localStorage.removeItem("desktopOpenVideoId");
    } catch { /* noop */ }
    hydratedVideoIdsRef.current.delete(openVideo);
    void hydrateCompletedVideo(openVideo);
  }, [hydrateCompletedVideo]);

  const resolveProcessingMode = useCallback(async () => {
    const bridge = getDesktopBridge();
    const next = await bridge?.processing?.resolveMode?.();
    const mode: ProcessingMode = next?.mode === "local_gpu"
      ? "local_gpu"
      : next?.mode === "local_cpu"
        ? "local_cpu"
        : "online";
    processingModeRef.current = mode;
    if (next) setProcessingStatus(next);
    return mode;
  }, []);

  const runtimeNotice = useMemo(() => {
    if (!processingStatus || processingStatus.mode === "local_gpu") return null;
    const reason = processingStatus.localGpuReason || processingStatus.localGpuDetails?.reason;
    if (reason === "nvidia_driver_missing") {
      return {
        title: "Локальная GPU-обработка недоступна: не установлен драйвер NVIDIA.",
        detail: `Установите NVIDIA Driver версии ${processingStatus.localGpuDetails?.minimumDriver || "560.76"} или новее.`,
        driverUrl: processingStatus.localGpuDetails?.driverUrl,
      };
    }
    if (reason === "gpu_incompatible") {
      const gpu = processingStatus.localGpuDetails?.gpus?.[0];
      const gpuText = gpu?.name
        ? `${gpu.name}, VRAM ${gpu.memoryMb || 0} MB, driver ${gpu.driver || "unknown"}`
        : "Совместимая NVIDIA GPU не найдена";
      return {
        title: "Локальная GPU-обработка недоступна: GPU или драйвер не подходят.",
        detail: `${gpuText}. Нужна NVIDIA GPU с VRAM от 12 GB и driver ${processingStatus.localGpuDetails?.minimumDriver || "560.76"} или новее.`,
        driverUrl: processingStatus.localGpuDetails?.driverUrl,
      };
    }
    return null;
  }, [processingStatus]);

  const refreshHistory = useCallback(async () => {
    try {
      if (processingModeRef.current !== "online") {
        const items = processingModeRef.current === "local_gpu"
          ? await getDesktopBridge()?.localGpu?.history?.()
          : await getDesktopBridge()?.localCpu?.history?.();
        setHistory((items || []) as VideoListItem[]);
      } else {
        const response = await getDesktopUploadedVideosList();
        const videos = response.videos || [];
        setHistory(videos);
        videos
          .filter((video) => ["completed", "done", "success"].includes(String(video.status || "").toLowerCase()))
          .slice(0, 6)
          .forEach((video) => {
            void hydrateCompletedVideo(video.video_id);
          });
      }
    } catch {
      setHistory([]);
    }
  }, [hydrateCompletedVideo]);

  useEffect(() => {
    void resolveProcessingMode().then(() => refreshHistory());
  }, [refreshHistory, resolveProcessingMode]);

  useEffect(() => {
    const unsubscribe = getDesktopBridge()?.localCpu?.onProgress?.((payload) => {
      const progressPayload = payload as { video_id?: string; percent?: number; message?: string };
      if (!progressPayload.video_id) return;
      const percent = Number(progressPayload.percent);
      setProcessingProgress((current) => current.map((item) => (
        item.videoId === progressPayload.video_id
          ? {
              ...item,
              percent: Number.isFinite(percent) ? Math.max(item.percent, Math.min(100, percent)) : item.percent,
              status: percent >= 100 ? "done" : "processing",
              message: progressPayload.message || item.message,
            }
          : item
      )));
    });
    return () => unsubscribe?.();
  }, []);

  const busy = ["looking", "copying", "processing"].includes(state);
  const statusMessage = (state === "looking" || state === "copying") && progress
    ? `Загрузка ${progress.index} из ${progress.total}: ${progress.fileName}`
    : message;

  const resetPlanView = useCallback(() => {
    setPlanZoom(1);
    setPlanPan({ x: 0, y: 0 });
  }, []);

  const zoomPlanAt = useCallback((nextZoom: number, anchorClientX?: number, anchorClientY?: number) => {
    const viewport = planViewportRef.current;
    const clamped = Math.max(PLAN_ZOOM_MIN, Math.min(PLAN_ZOOM_MAX, nextZoom));
    setPlanZoom((prevZoom) => {
      if (!viewport || Math.abs(clamped - prevZoom) < 0.001) return prevZoom;
      const rect = viewport.getBoundingClientRect();
      const anchorX = anchorClientX == null ? rect.width / 2 : anchorClientX - rect.left;
      const anchorY = anchorClientY == null ? rect.height / 2 : anchorClientY - rect.top;
      setPlanPan((prevPan) => {
        const worldX = (anchorX - prevPan.x) / prevZoom;
        const worldY = (anchorY - prevPan.y) / prevZoom;
        return {
          x: anchorX - worldX * clamped,
          y: anchorY - worldY * clamped,
        };
      });
      return clamped;
    });
  }, []);

  const handlePlanWheel = useCallback((event: WheelEvent<HTMLDivElement>) => {
    // Native passive:false listener ниже; React onWheel оставляем как no-op fallback.
    event.preventDefault();
  }, []);

  useEffect(() => {
    const viewport = planViewportRef.current;
    if (!viewport || trajectories.length > 0) return;
    const onWheel = (event: globalThis.WheelEvent) => {
      event.preventDefault();
      const factor = event.deltaY > 0 ? 0.9 : 1.1;
      setPlanZoom((prevZoom) => {
        const clamped = Math.max(PLAN_ZOOM_MIN, Math.min(PLAN_ZOOM_MAX, prevZoom * factor));
        if (Math.abs(clamped - prevZoom) < 0.001) return prevZoom;
        const rect = viewport.getBoundingClientRect();
        const anchorX = event.clientX - rect.left;
        const anchorY = event.clientY - rect.top;
        setPlanPan((prevPan) => {
          const worldX = (anchorX - prevPan.x) / prevZoom;
          const worldY = (anchorY - prevPan.y) / prevZoom;
          return {
            x: anchorX - worldX * clamped,
            y: anchorY - worldY * clamped,
          };
        });
        return clamped;
      });
    };
    viewport.addEventListener("wheel", onWheel, { passive: false });
    return () => viewport.removeEventListener("wheel", onWheel);
  }, [trajectories.length]);

  const handlePlanPointerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (busy || event.button !== 0) return;
    planDragRef.current = {
      active: true,
      moved: false,
      startX: event.clientX,
      startY: event.clientY,
      originX: planPan.x,
      originY: planPan.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [busy, planPan.x, planPan.y]);

  const handlePlanPointerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
    const drag = planDragRef.current;
    if (!drag?.active) return;
    const dx = event.clientX - drag.startX;
    const dy = event.clientY - drag.startY;
    if (!drag.moved && Math.hypot(dx, dy) >= PLAN_DRAG_THRESHOLD_PX) {
      drag.moved = true;
    }
    if (drag.moved) {
      setPlanPan({ x: drag.originX + dx, y: drag.originY + dy });
    }
  }, []);

  const handlePlanPointerUp = useCallback((event: PointerEvent<HTMLDivElement>) => {
    const drag = planDragRef.current;
    if (!drag?.active) return;
    const moved = drag.moved;
    planDragRef.current = null;
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch { /* noop */ }
    if (moved || busy) return;

    const planLayer = event.currentTarget.querySelector("[data-plan-layer]") as HTMLElement | null;
    const target = planLayer || event.currentTarget;
    const rect = target.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    if (
      event.clientX < rect.left
      || event.clientX > rect.right
      || event.clientY < rect.top
      || event.clientY > rect.bottom
    ) {
      return;
    }
    const x = Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100));
    const y = Math.max(0, Math.min(100, ((event.clientY - rect.top) / rect.height) * 100));
    const point = { x, y };
    if (!referencePoint || planPickMode === "start") {
      setReferencePoint(point);
      setDirectionPoint(null);
      setPlanPickMode("direction");
      setMessage("Укажите направление первого движения на плане.");
      return;
    }
    setDirectionPoint(point);
    setPlanPickMode("direction");
    setMessage("Старт и направление заданы. Теперь можно загрузить видео с камеры.");
  }, [busy, planPickMode, referencePoint]);

  const handlePlanClick = useCallback((event: MouseEvent<HTMLDivElement>) => {
    // Клик обрабатывается в pointerup; блокируем native click после drag.
    event.preventDefault();
  }, []);

  const resetPlanSelection = useCallback(() => {
    if (busy) return;
    setReferencePoint(null);
    setDirectionPoint(null);
    setTrajectories([]);
    setPlanPickMode("start");
    resetPlanView();
    setMessage("Укажите стартовую точку на плане Kerama Marazzi.");
  }, [busy, resetPlanView]);

  const beginPickStart = useCallback(() => {
    if (busy) return;
    setPlanPickMode("start");
    // Убираем отрисованные траектории, иначе прячется кликабельный план с кнопками.
    setTrajectories([]);
    setMessage("Укажите стартовую точку на плане Kerama Marazzi.");
  }, [busy]);

  const beginPickDirection = useCallback(() => {
    if (busy || !referencePoint) return;
    setPlanPickMode("direction");
    setTrajectories([]);
    setMessage("Укажите направление первого движения на плане.");
  }, [busy, referencePoint]);

  const processImportedVideos = useCallback(async (videos: CameraImportedVideo[]) => {
    if (!videos.length) return;
    if (!analysisQueueIdRef.current) {
      analysisQueueIdRef.current = `desktop-camera-${Date.now()}`;
    }
    setState("processing");
    setProgress(null);
    setMessage(`Запускаем анализ: 0 из ${Math.max(sessionBatchCounterRef.current, videos.length)}`);
    setProcessingProgress((current) => {
      const incomingIds = new Set(videos.map((video) => video.video_id));
      return [
        ...current.filter((item) => !incomingIds.has(item.videoId)),
        ...videos.map((video) => {
          const batch = resolveBatchSlot(video.video_id);
          return {
            videoId: video.video_id,
            fileName: video.original_filename || video.filename,
            index: batch.index,
            total: batch.total,
            percent: 0,
            status: "queued" as const,
            message: "В очереди",
          };
        }),
      ];
    });

    try {
      let successCount = 0;
      let failedCount = 0;
      const trajectoryVideoIds = new Set<string>();
      for (let index = 0; index < videos.length; index += 1) {
        if (analysisStoppedRef.current) break;
        const video = videos[index];
        const fileName = video.original_filename || video.filename;
        const batch = resolveBatchSlot(video.video_id);
        setMessage(`Анализируем ${batch.index} из ${batch.total}: ${fileName}`);
        upsertProcessingProgress({
          videoId: video.video_id,
          fileName,
          index: batch.index,
          total: batch.total,
          percent: 2,
          status: "processing",
          message: "Запущено",
        });
        try {
        let result: VideoAnalysisResult["data"] | undefined;
        let gpuBusyRetries = 0;
        const videoMode: ProcessingMode = video.localPath ? "local_cpu" : processingModeRef.current;
        if (videoMode !== "online") {
          const local = videoMode === "local_gpu"
            ? await getDesktopBridge()?.localGpu?.process(video)
            : await getDesktopBridge()?.localCpu?.process(video);
          const nextTrajectory = getDesktopTrajectory((local as VideoAnalysisResult | undefined)?.data, video.video_id);
          if (!nextTrajectory.length) throw new Error("Не удалось построить траекторию для этого видео");
          trajectoryVideoIds.add(video.video_id);
          setTrajectories((current) => [...current, ...nextTrajectory]);
          setStats(((local as VideoAnalysisResult).data?.processing_stats || {}) as Record<string, unknown>);
          upsertProcessingProgress({
            videoId: video.video_id,
            fileName,
            index: batch.index,
            total: batch.total,
            percent: 100,
            status: "done",
            message: "Готово",
          });
          successCount += 1;
          continue;
        }
        await waitForVideoReadyForAnalysis(
          video.video_id,
          fileName,
          () => analysisStoppedRef.current,
          (message) => {
            setMessage(`Подготовка ${batch.index} из ${batch.total}: ${message}`);
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: batch.index,
              total: batch.total,
              percent: 3,
              status: "processing",
              message,
            });
          },
        );
        // Явный R³ через очередь смены — без auto_analysis и без параллели на GPU.
        const sequenceIndex = analysisSequenceRef.current;
        analysisSequenceRef.current += 1;
        const mapContext = sequenceIndex === 0
          ? {
              floorplan_id: "kerama_marazzi_2025",
              reference_point: referencePoint,
              direction_point: directionPoint,
            }
          : {
              floorplan_id: "kerama_marazzi_2025",
              queue_inherit_anchor: true,
            };
        const queueOptions = {
          queue_id: analysisQueueIdRef.current,
          sequence_index: sequenceIndex,
        };
        let started: VideoAnalysisResult = {
          success: true,
          status: "queued",
          video_id: video.video_id,
        };
        // Start status polling immediately — do not wait for analyze POST.
        // Installed builds used to freeze at 2%/Запущено when the API was wedged.
        const analyzePromise = apiClient.analyzeVideoById(
            video.video_id,
            12.306,
            true,
            video.original_filename || video.filename,
            undefined,
            mapContext,
            CAMERA_OWNER,
            "r3",
            undefined,
            true,
            queueOptions,
          ).then((value) => {
            started = value;
            return value;
          }).catch((analyzeStartError) => {
            const text = analyzeStartError instanceof Error
              ? analyzeStartError.message
              : "Не удалось стартовать анализ";
            const waitMessage = isRetryableGpuBusyError(text)
              ? "TrackAI занят — ждём очередь..."
              : "Продолжаем обработку...";
            setMessage(`${fileName}: ${formatDesktopProcessingMessage(text)}`);
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: batch.index,
              total: batch.total,
              percent: 5,
              status: "processing",
              message: waitMessage,
            });
            return started;
          });
        void analyzePromise;
        const expectedRunId = started.analysis_run_id;
        result = started.data;
        const shouldPoll = true;

        if (shouldPoll) {
          upsertProcessingProgress({
            videoId: video.video_id,
            fileName,
            index: batch.index,
            total: batch.total,
            percent: 5,
            status: "queued",
            message: "Подготовка",
          });
          for (let attempt = 0; attempt < 1800; attempt += 1) {
            if (analysisStoppedRef.current) break;
            // Give analyze a moment to schedule the worker on first ticks.
            if (attempt === 0) {
              await Promise.race([
                analyzePromise,
                new Promise((resolve) => window.setTimeout(resolve, 3000)),
              ]);
            }
            const status = await getDesktopProcessingStatusResilient(
              video.video_id,
              (message) => setMessage(message || `Обрабатываем ${batch.index} из ${batch.total}`),
            );
            setMessage(formatDesktopProcessingMessage(status.message) || `Обрабатываем ${batch.index} из ${batch.total}`);
            const percent = Number((status as { progress?: unknown }).progress);
            // Прогресс загрузки не трогаем здесь — иначе дёргается бар.
            if (["registered", "uploading"].includes(String(status.status || "").toLowerCase())) {
              setProcessingProgress((current) => current.filter((item) => item.videoId !== video.video_id));
              await new Promise((resolve) => window.setTimeout(resolve, 2000));
              continue;
            }
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: batch.index,
              total: batch.total,
              percent: Number.isFinite(percent) ? Math.max(5, Math.min(99, percent)) : Math.min(95, 8 + attempt),
              status: "processing",
              message: formatDesktopProcessingMessage(status.message) || "Обрабатывается",
            });
            if (status.status === "error" || status.status === "failed") {
              const statusMessage = String(status.message || "");
              if (isRetryableGpuBusyError(statusMessage) && gpuBusyRetries < GPU_BUSY_MAX_RETRIES) {
                gpuBusyRetries += 1;
                const waitSec = Math.round(GPU_BUSY_RETRY_WAIT_MS / 1000);
                upsertProcessingProgress({
                  videoId: video.video_id,
                  fileName,
                  index: batch.index,
                  total: batch.total,
                  percent: Math.max(5, Number.isFinite(percent) ? percent : 5),
                  status: "processing",
                  message: `TrackAI занят — ждём очередь (${waitSec} сек)...`,
                });
                setMessage(`TrackAI занят для ${fileName}. Ждём ${waitSec} сек (${gpuBusyRetries}/${GPU_BUSY_MAX_RETRIES})...`);
                await new Promise((resolve) => window.setTimeout(resolve, GPU_BUSY_RETRY_WAIT_MS));
                if (analysisStoppedRef.current) break;
                try {
                  await apiClient.analyzeVideoById(
                    video.video_id,
                    12.306,
                    true,
                    video.original_filename || video.filename,
                    undefined,
                    mapContext,
                    CAMERA_OWNER,
                    "r3",
                    undefined,
                    true,
                    queueOptions,
                  );
                } catch {
                  /* keep polling */
                }
                continue;
              }
              if (isDesktopUserFacingFailure(statusMessage, status.result)) {
                result = status.result || result;
                break;
              }
              throw new Error(formatDesktopUserFacingError(status.message, status.result));
            }
            // Accept completed even if run_id changed after API restart/re-trigger.
            const sameRun = !expectedRunId
              || !status.analysis_run_id
              || status.analysis_run_id === expectedRunId
              || attempt >= 15;
            if (sameRun && ["completed", "done", "success"].includes(status.status)) {
              result = status.result || (await getDesktopVideoAnalysis(video.video_id)).data;
              break;
            }
            await new Promise((resolve) => window.setTimeout(resolve, 2000));
          }
          if (analysisStoppedRef.current) {
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: batch.index,
              total: batch.total,
              percent: 100,
              status: "error",
              message: "Остановлено",
            });
            break;
          }
        }
        if (!result) {
          result = await recoverTrajectoryFromAnalysis(video.video_id);
        }
        let nextTrajectory = getDesktopTrajectory(result, video.video_id);
        if (
          !nextTrajectory.length
          && isDesktopPlanBindFailure(result)
          && !retriedPlanBindVideoIdsRef.current.has(video.video_id)
        ) {
          retriedPlanBindVideoIdsRef.current.add(video.video_id);
          upsertProcessingProgress({
            videoId: video.video_id,
            fileName,
            index: batch.index,
            total: batch.total,
            percent: 90,
            status: "processing",
            message: "Уточняем маршрут на плане...",
          });
          try {
            await apiClient.analyzeVideoById(
              video.video_id,
              12.306,
              true,
              video.original_filename || video.filename,
              undefined,
              mapContext,
              CAMERA_OWNER,
              "r3",
              undefined,
              true,
              queueOptions,
            );
            for (let retryAttempt = 0; retryAttempt < 60; retryAttempt += 1) {
              if (analysisStoppedRef.current) break;
              const retryStatus = await getDesktopProcessingStatusResilient(video.video_id);
              if (["completed", "done", "success"].includes(String(retryStatus.status || ""))) {
                result = retryStatus.result || (await getDesktopVideoAnalysis(video.video_id)).data;
                break;
              }
              if (["error", "failed"].includes(String(retryStatus.status || ""))) break;
              await new Promise((resolve) => window.setTimeout(resolve, 2000));
            }
            if (!result) {
              result = await recoverTrajectoryFromAnalysis(video.video_id);
            }
            nextTrajectory = getDesktopTrajectory(result, video.video_id);
          } catch {
            /* fall through to soft hint */
          }
        }
        if (!nextTrajectory.length) {
          if (isDesktopPlanBindFailure(result)) {
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: batch.index,
              total: batch.total,
              percent: 100,
              status: "hint",
              message: formatDesktopSoftPlanHint(),
            });
            setMessage(formatDesktopSoftPlanHint());
            continue;
          }
          throw new Error(formatDesktopUserFacingError(undefined, result));
        }
        trajectoryVideoIds.add(video.video_id);
        setTrajectories((current) => [...current, ...nextTrajectory]);
        setStats((result?.processing_stats || {}) as Record<string, unknown>);
        upsertProcessingProgress({
          videoId: video.video_id,
          fileName,
          index: batch.index,
          total: batch.total,
          percent: 100,
          status: "done",
          message: "Готово",
        });
        successCount += 1;
        } catch (videoError) {
          if (analysisStoppedRef.current) {
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: batch.index,
              total: batch.total,
              percent: 100,
              status: "error",
              message: "Остановлено",
            });
            break;
          }
          const text = formatDesktopUserFacingError(
            videoError instanceof Error ? videoError.message : undefined,
            result,
          );
          if (isDesktopUserFacingFailure(
            videoError instanceof Error ? videoError.message : "",
            result,
          )) {
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: batch.index,
              total: batch.total,
              percent: 100,
              status: "hint",
              message: formatDesktopSoftPlanHint(),
            });
            setMessage(formatDesktopSoftPlanHint());
            continue;
          }
          if (isRetryableGpuBusyError(
            videoError instanceof Error ? videoError.message : text,
          ) && gpuBusyRetries < GPU_BUSY_MAX_RETRIES) {
            gpuBusyRetries += 1;
            const waitSec = Math.round(GPU_BUSY_RETRY_WAIT_MS / 1000);
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: batch.index,
              total: batch.total,
              percent: 5,
              status: "processing",
              message: `TrackAI занят — ждём очередь (${waitSec} сек)...`,
            });
            setMessage(`TrackAI занят для ${fileName}. Ждём ${waitSec} сек (${gpuBusyRetries}/${GPU_BUSY_MAX_RETRIES})...`);
            await new Promise((resolve) => window.setTimeout(resolve, GPU_BUSY_RETRY_WAIT_MS));
            if (analysisStoppedRef.current) break;
            index -= 1;
            continue;
          }
          if (isRetryableGpuVramError(text) && !retriedVramVideoIdsRef.current.has(video.video_id)) {
            retriedVramVideoIdsRef.current.add(video.video_id);
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: batch.index,
              total: batch.total,
              percent: 5,
              status: "processing",
              message: "TrackAI занят — повтор через 90 сек...",
            });
            setMessage(`TrackAI занят для ${fileName}. Повторяем через 90 сек...`);
            await new Promise((resolve) => window.setTimeout(resolve, 90_000));
            if (analysisStoppedRef.current) break;
            index -= 1;
            continue;
          }
          const recovered = await recoverTrajectoryFromAnalysis(video.video_id);
          if (recovered) {
            const recoveredTrajectory = getDesktopTrajectory(recovered, video.video_id);
            if (recoveredTrajectory.length) {
              trajectoryVideoIds.add(video.video_id);
              setTrajectories((current) => [...current, ...recoveredTrajectory]);
              setStats((recovered.processing_stats || {}) as Record<string, unknown>);
              upsertProcessingProgress({
                videoId: video.video_id,
                fileName,
                index: batch.index,
                total: batch.total,
                percent: 100,
                status: "done",
                message: "Готово",
              });
              successCount += 1;
              continue;
            }
          }
          failedCount += 1;
          upsertProcessingProgress({
            videoId: video.video_id,
            fileName,
            index: batch.index,
            total: batch.total,
            percent: 100,
            status: "error",
            message: text,
          });
          toast.error(`${fileName}: ${text}`);
        }
      }
      for (const video of videos) {
        if (trajectoryVideoIds.has(video.video_id)) continue;
        const fileName = video.original_filename || video.filename;
        const recovered = await recoverTrajectoryFromAnalysis(video.video_id);
        if (!recovered) continue;
        const recoveredTrajectory = getDesktopTrajectory(recovered, video.video_id);
        if (!recoveredTrajectory.length) continue;
        trajectoryVideoIds.add(video.video_id);
        setTrajectories((current) => [...current, ...recoveredTrajectory]);
        setStats((recovered.processing_stats || {}) as Record<string, unknown>);
        const recoveredBatch = resolveBatchSlot(video.video_id);
        upsertProcessingProgress({
          videoId: video.video_id,
          fileName,
          index: recoveredBatch.index,
          total: recoveredBatch.total,
          percent: 100,
          status: "done",
          message: "Готово",
        });
        successCount += 1;
        failedCount = Math.max(0, failedCount - 1);
      }
      if (analysisStoppedRef.current) {
        setProcessingProgress((current) => current.map((item) => (
          item.status === "processing" || item.status === "queued"
            ? { ...item, status: "error", message: "Остановлено" }
            : item
        )));
        setState("ready");
        setMessage("Полный стоп: выгрузка и обработка прерваны.");
        return;
      }
      const stillImporting = Boolean(
        (await (cameraImport?.getStatus().catch(() => null) ?? Promise.resolve(null)))?.importing,
      );
      if (stillImporting) {
        setState("processing");
        setMessage(
          successCount > 0
            ? `Траектория добавлена на план (${successCount}). Загружаем следующий ролик с камеры...`
            : "Ролик без траектории. Загружаем следующий с камеры...",
        );
      } else {
        setState(successCount > 0 ? "done" : "ready");
        setMessage(
          successCount > 0
            ? (failedCount
              ? `Готово: ${successCount} видео на плане. Для остальных уточните старт и направление.`
              : "Готово. Траектория показана на плане Kerama Marazzi.")
            : (failedCount
              ? formatDesktopSoftPlanHint()
              : "Готово.")
        );
      }
      await refreshHistory();
    } catch (error) {
      setProcessingProgress((current) => current.map((item) => (
        item.status === "processing" || item.status === "queued"
          ? {
              ...item,
              status: "error",
              message: analysisStoppedRef.current
                ? "Остановлено"
                : formatDesktopUserFacingError(error instanceof Error ? error.message : undefined),
            }
          : item
      )));
      setState("ready");
      setMessage(formatDesktopUserFacingError(error instanceof Error ? error.message : undefined));
      await refreshHistory();
    }
  }, [cameraImport, directionPoint, referencePoint, refreshHistory, resolveBatchSlot, upsertProcessingProgress]);

  const enqueueImportedVideos = useCallback((videos: CameraImportedVideo[]) => {
    if (importStoppedRef.current || analysisStoppedRef.current) return Promise.resolve();
    const freshVideos = videos.filter((video) => {
      if (enqueuedVideoIdsRef.current.has(video.video_id)) return false;
      enqueuedVideoIdsRef.current.add(video.video_id);
      return true;
    });
    if (!freshVideos.length) return Promise.resolve();
    freshVideos.forEach((video) => {
      resolveBatchSlot(video.video_id);
    });
    const sessionTotal = Math.max(
      sessionBatchCounterRef.current,
      uploadedVideoIdsRef.current.size,
      enqueuedVideoIdsRef.current.size,
    );
    setProcessingProgress((current) => current.map((item) => ({
      ...item,
      total: Math.max(item.total, sessionTotal),
    })));
    const run = analysisQueueRef.current
      .catch(() => undefined)
      .then(() => processImportedVideos(freshVideos));
    analysisQueueRef.current = run.then(() => undefined).catch(() => undefined);
    return run;
  }, [processImportedVideos, resolveBatchSlot]);

  useEffect(() => {
    if (!cameraImport) return;
    const notifyProcessed = async (videoId: string) => {
      try {
        await cameraImport.notifyProcessed?.(videoId);
      } catch {
        // Gate timeout in main will unblock the next upload if notify fails.
      }
    };
    const unsubscribeProgress = cameraImport.onProgress((next) => {
      if (importStoppedRef.current) return;
      setState("copying");
      setProgress((current) => mergeUploadProgress(current, next));
    });
    const unsubscribeFileImported = cameraImport.onFileImported((video) => {
      if (importStoppedRef.current) {
        void notifyProcessed(video.video_id);
        return;
      }
      // Progressive: upload → analyze → draw → inherit anchor → next upload.
      uploadedVideoIdsRef.current.add(video.video_id);
      setProgress(null);
      setState("processing");
      setMessage(`Анализ после загрузки: ${video.original_filename || video.filename}`);
      void enqueueImportedVideos([video as CameraImportedVideo])
        .catch(() => undefined)
        .finally(() => {
          void notifyProcessed(video.video_id);
        });
    });
    const unsubscribeComplete = cameraImport.onComplete((videos) => {
      setProgress(null);
      if (importStoppedRef.current) return;
      for (const video of videos || []) {
        uploadedVideoIdsRef.current.add(video.video_id);
      }
      // Analysis already started per-file; only pick up any missed ids.
      void enqueueImportedVideos((videos || []) as CameraImportedVideo[]);
      setState((current) => (current === "looking" || current === "copying" ? "processing" : current));
    });
    const unsubscribeError = cameraImport.onError((error) => {
      const text = error.message || "Не удалось загрузить видео с камеры";
      setProgress(null);
      if (importStoppedRef.current || /остановлен пользователем|aborted|AbortError|Upload cancelled/i.test(text)) {
        importStoppedRef.current = true;
        setState((current) => (current === "looking" || current === "copying" ? "ready" : current));
        setMessage("Выгрузка с камеры остановлена. Уже скопированные видео останутся в обработке.");
        return;
      }
      // Ошибка одного файла: не держим бар «100%» и не блокируем кнопку «Загрузить».
      setState((current) => {
        if (current === "looking" || current === "copying") {
          return processingProgressRef.current.some((item) => item.status === "processing" || item.status === "queued")
            ? "processing"
            : "error";
        }
        return current;
      });
      setMessage(text);
      toast.error(text);
    });
    return () => {
      unsubscribeProgress();
      unsubscribeFileImported();
      unsubscribeComplete();
      unsubscribeError();
    };
  }, [cameraImport, enqueueImportedVideos]);

  // Если прогресс загрузки не двигается долго — сбрасываем «залипший» бар (после fetch failed / hung socket).
  useEffect(() => {
    if (!progress || !["looking", "copying"].includes(state)) return;
    const stuckFile = progress.fileName;
    const stuckIndex = progress.index;
    const stuckPercent = progress.percent;
    const timer = window.setTimeout(() => {
      setProgress((current) => {
        if (!current) return current;
        if (current.fileName !== stuckFile || current.index !== stuckIndex) return current;
        if (Math.abs(current.percent - stuckPercent) > 0.5) return current;
        return null;
      });
      setState((current) => (current === "looking" || current === "copying" ? "error" : current));
      setMessage(`Загрузка зависла на ${stuckFile} (${Math.round(stuckPercent)}%). Нажмите «Остановить выгрузку» и повторите «Загрузить».`);
      toast.error(`Загрузка зависла: ${stuckFile}`);
    }, 5 * 60 * 1000);
    return () => window.clearTimeout(timer);
  }, [progress, state]);

  useEffect(() => {
    if (!cameraImport) return;
    let cancelled = false;
    const syncServerProgress = async () => {
      if (processingModeRef.current !== "online") return;
      if (analysisStoppedRef.current) return;
      const sessionIds = new Set([
        ...uploadedVideoIdsRef.current,
        ...enqueuedVideoIdsRef.current,
      ]);
      if (sessionIds.size === 0) return;
      try {
        const response = await getDesktopUploadedVideosList();
        if (cancelled) return;
        const activeVideos = (response.videos || []).filter((video) => {
          if (!sessionIds.has(video.video_id)) return false;
          const status = String(video.status || "").toLowerCase();
          const progressValue = Number(video.progress || 0);
          const isDesktop = video.client_source === "desktop" || /VID\d+\.(avi|mp4|mov|mkv)$/i.test(video.original_filename || video.filename);
          if (!isDesktop) return false;
          return (
            ["registered", "uploading", "uploaded", "queued", "processing", "running", "gpu_processing", "completed", "done", "success", "error", "failed"].includes(status)
            || (progressValue > 0 && progressValue < 100)
          );
        }).slice(0, 8);
        activeVideos.forEach((video) => {
          const progressValue = Number(video.progress || 0);
          const status = String(video.status || "").toLowerCase();
          if (["registered", "uploading", "uploaded"].includes(status)) {
            return;
          }
          // Only mirror server state for videos already handed to R3 queue.
          if (!enqueuedVideoIdsRef.current.has(video.video_id)) {
            return;
          }
          const batch = resolveBatchSlot(video.video_id);
          const serverMessage = String(video.message || "");
          const currentItem = processingProgressRef.current.find((item) => item.videoId === video.video_id);
          if (
            currentItem
            && (currentItem.status === "processing" || currentItem.status === "queued")
            && /ждём очередь/i.test(currentItem.message || "")
          ) {
            return;
          }
          if (currentItem?.status === "hint") {
            return;
          }
          const isGpuBusy = isRetryableGpuBusyError(serverMessage);
          let rowStatus: VideoProcessingProgress["status"] = "processing";
          if (isGpuBusy) {
            rowStatus = "processing";
          } else if (["error", "failed"].includes(status)) {
            rowStatus = "error";
          } else if (["completed", "done", "success"].includes(status)) {
            rowStatus = "done";
          }
          upsertProcessingProgress({
            videoId: video.video_id,
            fileName: video.original_filename || video.filename,
            index: batch.index,
            total: batch.total,
            percent: Number.isFinite(progressValue) ? Math.max(0, Math.min(99, progressValue)) : 0,
            status: rowStatus,
            message: formatDesktopProcessingMessage(serverMessage) || "Обрабатывается",
          });
          if (["completed", "done", "success"].includes(status) && !cancelled
            && enqueuedVideoIdsRef.current.has(video.video_id)) {
            void hydrateCompletedVideo(video.video_id);
          }
        });
      } catch {
        // The import flow still reports hard errors; this poller is only a UI recovery path.
      }
    };
    void syncServerProgress();
    const timer = window.setInterval(() => {
      void syncServerProgress();
    }, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [cameraImport, hydrateCompletedVideo, resolveBatchSlot, upsertProcessingProgress]);

  const handleUpload = async () => {
    if (!cameraImport) {
      setState("error");
      setMessage("Откройте это окно в приложении TrackAI для Windows.");
      return;
    }
    if (!referencePoint || !directionPoint) {
      setState("ready");
      setMessage(!referencePoint
        ? "Перед загрузкой укажите стартовую точку на плане."
        : "Перед загрузкой укажите направление движения на плане.");
      setPlanPickMode(referencePoint ? "direction" : "start");
      return;
    }
    setState("looking");
    setMessage("Проверяем подключение и ищем экшен-камеру...");
    importStoppedRef.current = false;
    analysisStoppedRef.current = false;
    retriedVramVideoIdsRef.current = new Set();
    retriedPlanBindVideoIdsRef.current = new Set();
    sessionBatchIndexByVideoIdRef.current = new Map();
    sessionBatchCounterRef.current = 0;
    enqueuedVideoIdsRef.current = new Set();
    uploadedVideoIdsRef.current = new Set();
    setProcessingProgress([]);
    setProgress(null);
    analysisQueueIdRef.current = `desktop-camera-${Date.now()}`;
    analysisSequenceRef.current = 0;
    try {
      await resolveProcessingMode();
      await cameraImport.setSettings({ enabled: true, ownerName: CAMERA_OWNER });
      await cameraImport.resetImportedState();
      const status = await cameraImport.scanNow({
        forceImport: true,
        ignoreImported: true,
        analysisContext: {
          floorplan_id: "kerama_marazzi_2025",
          employee_name: CAMERA_OWNER,
          analysis_method: "r3",
          scale_factor: 12.306,
          auto_analysis: false,
        },
      });
      if (!status.volumes?.length) {
        setState("needs_camera");
        setMessage("Камера не найдена. Подключите её по USB, разблокируйте накопитель и нажмите «Загрузить» ещё раз.");
      } else if (!status.pendingFiles?.length && !status.importing) {
        setState("ready");
        setMessage("На подключённой камере нет видео.");
      }
    } catch (error) {
      setProgress(null);
      setState("error");
      setMessage(error instanceof Error ? error.message : "Не удалось прочитать камеру");
    }
  };

  const openHistoryItem = async (video: VideoListItem) => {
    try {
      const result = processingModeRef.current !== "online"
        ? processingModeRef.current === "local_gpu"
          ? await getDesktopBridge()?.localGpu?.analysis(video.video_id) as { data?: VideoAnalysisResult["data"] }
          : await getDesktopBridge()?.localCpu?.analysis(video.video_id) as { data?: VideoAnalysisResult["data"] }
        : await getDesktopVideoAnalysis(video.video_id);
      if (!result) throw new Error("Результат анализа не найден");
      const next = getDesktopTrajectory(result.data, video.video_id);
      if (!next.length) {
        throw new Error(formatDesktopSoftPlanHint());
      }
      setTrajectories(next);
      setStats((result?.data?.processing_stats || {}) as Record<string, unknown>);
      setState("done");
      setMessage(`Открыта траектория: ${video.filename}`);
      setHistoryOpen(false);
    } catch (error) {
      setState("error");
      setMessage(formatDesktopUserFacingError(error instanceof Error ? error.message : undefined));
    }
  };

  const downloadLogs = async () => {
    const download = getDesktopBridge()?.logs?.download;
    if (!download) {
      toast.error("Экспорт логов доступен только в приложении TrackAI для Windows.");
      return;
    }
    setLogsDownloading(true);
    try {
      const result = await download();
      if (result.ok) {
        toast.success(`Логи сохранены: ${result.fileName || "TrackAI-logs.zip"}`);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Не удалось сохранить логи");
    } finally {
      setLogsDownloading(false);
    }
  };

  const cancelCameraImport = async () => {
    importStoppedRef.current = true;
    setProgress(null);
    try {
      const result = await cameraImport?.cancel?.();
      // Не держим UI в busy: выгрузка прервана, уже залитые ролики могут доанализироваться.
      setState((current) => (current === "looking" || current === "copying" ? "ready" : current));
      setMessage(
        result?.cancelled
          ? "Выгрузка с камеры остановлена. Уже скопированные видео останутся в обработке."
          : "Выгрузка остановлена."
      );
      toast.success("Выгрузка остановлена");
    } catch (error) {
      setState("ready");
      setMessage(error instanceof Error ? error.message : "Не удалось остановить выгрузку с камеры");
      toast.error(error instanceof Error ? error.message : "Не удалось остановить выгрузку");
    }
  };

  const fullStopCameraSession = async () => {
    importStoppedRef.current = true;
    analysisStoppedRef.current = true;
    setProgress(null);
    try {
      await cameraImport?.cancel?.();
    } catch { /* noop */ }
    try {
      await cameraImport?.setSettings?.({ enabled: false, ownerName: CAMERA_OWNER });
    } catch { /* noop */ }
    analysisQueueRef.current = Promise.resolve();
    setProcessingProgress((current) => current.map((item) => (
      item.status === "processing" || item.status === "queued"
        ? { ...item, status: "error", message: "Остановлено" }
        : item
    )));
    setState("ready");
    setMessage("Полный стоп: выгрузка и обработка прерваны.");
    toast.success("Полный стоп");
  };

  const resetCameraImportMemory = async () => {
    if (busy) return;
    try {
      const result = await cameraImport?.resetImportedState?.();
      setProgress(null);
      setState("ready");
      setMessage(`Память камеры очищена. При следующей загрузке TrackAI снова начнет с первого видео${typeof result?.removed === "number" ? ` (${result.removed} записей сброшено)` : ""}.`);
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Не удалось сбросить память камеры");
    }
  };

  const buttonText = busy ? "Выполняется" : "Загрузить";
  const canUpload = Boolean(referencePoint && directionPoint);
  // Кнопка видна пока идёт поиск/копирование или есть бар загрузки (даже если уже стартовал анализ).
  const canCancelImport = ["looking", "copying"].includes(state) || Boolean(progress);
  const canFullStop = canCancelImport
    || state === "processing"
    || processingProgress.some((item) => item.status === "queued" || item.status === "processing");
  const processingStatusLabel: Record<VideoProcessingProgress["status"], string> = {
    queued: "В очереди",
    processing: "Обработка",
    done: "Готово",
    hint: "Уточните",
    error: "Стоп",
  };

  return (
    <main className="min-h-[100dvh] bg-slate-50 text-slate-950">
      <header className="flex h-16 items-center justify-between border-b border-slate-200 bg-slate-50 px-6">
        <div className="flex items-center gap-3 font-semibold tracking-tight"><MapPinned className="h-5 w-5 text-teal-700" />TrackAI</div>
        <div className="flex items-center gap-2">
          <Button variant="outline" className="gap-2 border-slate-300 bg-white text-slate-700 hover:bg-slate-200 hover:text-slate-950" disabled={logsDownloading} onClick={() => void downloadLogs()}>
            {logsDownloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            СКАЧАТЬ ЛОГИ
          </Button>
          <div className="relative">
            <Button variant="ghost" className="gap-2 text-slate-700 hover:bg-slate-200 hover:text-slate-950" onClick={() => { setHistoryOpen((open) => !open); void refreshHistory(); }}>
              <History className="h-4 w-4" />История<ChevronDown className="h-4 w-4" />
            </Button>
            {historyOpen && <div className="absolute right-0 top-11 z-10 w-96 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 shadow-xl shadow-slate-300/40">
              {history.length ? history.slice(0, 12).map((video) => <button key={video.video_id} onClick={() => void openHistoryItem(video)} className="block w-full border-b border-slate-200 px-4 py-3 text-left text-sm last:border-0 hover:bg-slate-100"><span className="block truncate text-slate-900">{video.filename}</span><span className="text-xs text-slate-500">{video.has_analysis ? "Траектория готова" : "В обработке"}</span></button>) : <p className="px-4 py-5 text-sm text-slate-500">История пока пуста</p>}
            </div>}
          </div>
        </div>
      </header>

      <section className="mx-auto grid max-w-6xl gap-8 px-6 py-10 lg:grid-cols-[360px_1fr]">
        <div className="flex min-h-[430px] flex-col justify-center">
          <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-100 text-teal-800"><Camera className="h-6 w-6" /></div>
          <h1 className="text-4xl font-semibold tracking-tight">Видео с камеры</h1>
          <p className="mt-3 max-w-sm leading-6 text-slate-600">Сначала укажите старт и направление на плане. Затем подключите экшен-камеру, TrackAI скопирует видео и покажет маршрут.</p>
          <Button size="lg" className="mt-8 h-14 w-full gap-3 bg-teal-700 text-base font-semibold text-slate-50 hover:bg-teal-800 active:translate-y-px" disabled={busy || !canUpload} onClick={() => void handleUpload()}>
            {busy ? <Loader2 className="h-5 w-5 animate-spin" /> : <Upload className="h-5 w-5" />}{buttonText}
          </Button>
          {canCancelImport && <Button type="button" variant="outline" className="mt-3 h-11 w-full border-rose-300 bg-white text-rose-700 hover:bg-rose-50 hover:text-rose-800" onClick={() => void cancelCameraImport()}>
            Остановить выгрузку
          </Button>}
          {canFullStop && <Button type="button" variant="outline" className="mt-2 h-11 w-full border-rose-500 bg-rose-600 text-white hover:bg-rose-700 hover:text-white" onClick={() => void fullStopCameraSession()}>
            Полный стоп
          </Button>}
          {!busy && <Button type="button" variant="ghost" className="mt-2 h-10 w-full text-xs text-slate-600 hover:bg-slate-100 hover:text-slate-950" onClick={() => void resetCameraImportMemory()}>
            Начать с первого видео
          </Button>}
          {!canUpload && <p className="mt-3 text-xs font-medium text-teal-800">Перед загрузкой задайте стартовую точку и направление движения на плане.</p>}
          {runtimeNotice && <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-950">
            <p className="font-semibold">{runtimeNotice.title}</p>
            <p className="mt-1 leading-5">{runtimeNotice.detail}</p>
            {runtimeNotice.driverUrl && <button type="button" className="mt-2 text-left font-semibold text-amber-900 underline underline-offset-2" onClick={() => getDesktopBridge()?.openExternal?.(runtimeNotice.driverUrl!)}>
              Открыть страницу NVIDIA Driver
            </button>}
          </div>}
          <p className={`mt-4 text-sm ${state === "error" ? "text-rose-700" : state === "needs_camera" ? "text-amber-700" : "text-slate-600"}`}>{statusMessage}</p>
          {progress && <div className="mt-3">
            <div className="mb-1 flex items-center justify-between gap-3 text-xs font-medium text-slate-600">
              <span className="shrink-0 font-semibold text-slate-700">Загрузка</span>
              <span className="min-w-0 flex-1 truncate text-slate-500">{progress.fileName}</span>
              <span className="shrink-0">{Math.round(progress.percent)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-slate-200">
              <div className="h-full rounded-full bg-teal-600 transition-all" style={{ width: `${Math.max(0, Math.min(100, progress.percent))}%` }} />
            </div>
            <p className="mt-1 text-xs text-teal-700">Файл {progress.index} из {progress.total}</p>
          </div>}
          {processingProgress.length > 0 && <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wide text-slate-500">
              <span>Обработка видео</span>
              <span>{processingProgress.filter((item) => item.status === "done").length}/{Math.max(...processingProgress.map((item) => item.total), processingProgress.length)}</span>
            </div>
            {processingProgress.slice(-6).map((item) => (
              <div key={item.videoId} className="rounded-lg border border-slate-200 bg-white p-2">
                <div className="flex items-center justify-between gap-3 text-xs">
                  <span className="min-w-0 truncate font-medium text-slate-800">{item.fileName}</span>
                  <span className={`shrink-0 font-semibold ${item.status === "error" ? "text-rose-700" : item.status === "done" ? "text-teal-700" : item.status === "hint" ? "text-amber-700" : "text-slate-600"}`}>
                    {processingStatusLabel[item.status]} {Math.round(item.percent)}%
                  </span>
                </div>
                <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-slate-200">
                  <div
                    className={`h-full rounded-full transition-all ${item.status === "error" ? "bg-rose-600" : item.status === "done" ? "bg-teal-600" : item.status === "hint" ? "bg-amber-500" : "bg-blue-600"}`}
                    style={{ width: `${Math.max(0, Math.min(100, item.percent))}%` }}
                  />
                </div>
                {item.message && (
                  <p
                    className={`mt-1 text-[11px] text-slate-500 ${item.status === "error" ? "whitespace-pre-wrap break-words leading-snug text-rose-700" : item.status === "hint" ? "whitespace-pre-wrap break-words leading-snug text-amber-700" : "truncate"}`}
                    title={item.status === "error" ? formatDesktopUserFacingError(item.message) : undefined}
                  >
                    {item.index} из {item.total}: {item.status === "error"
                      ? formatDesktopUserFacingError(item.message)
                      : item.status === "hint"
                        ? item.message
                        : formatDesktopProcessingMessage(item.message)}
                  </p>
                )}
              </div>
            ))}
          </div>}
        </div>
        <div className="min-h-[520px] overflow-hidden rounded-2xl border border-slate-200 bg-slate-100 p-2 shadow-sm shadow-slate-300/40">
          <div className="flex h-full min-h-[500px] flex-col bg-white">
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
              <div>
                <p className="text-sm font-semibold text-slate-950">План Kerama Marazzi</p>
                <p className="text-xs text-slate-500">
                  {!referencePoint
                    ? "1. Приблизьте план и укажите стартовую точку"
                    : !directionPoint
                      ? "2. Укажите направление первого движения"
                      : trajectories.length
                        ? "Можно уточнить старт/направление кликом или кнопками слева"
                        : "Старт и направление заданы — можно загружать"}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button type="button" size="sm" variant={planPickMode === "start" ? "default" : "outline"} disabled={busy} onClick={beginPickStart}>Старт</Button>
                <Button type="button" size="sm" variant={planPickMode === "direction" ? "default" : "outline"} disabled={busy || !referencePoint} onClick={beginPickDirection}>Направление</Button>
                <Button type="button" size="sm" variant="ghost" disabled={busy || (!referencePoint && !directionPoint && !trajectories.length)} onClick={resetPlanSelection}>Сбросить</Button>
              </div>
            </div>
            <div className="relative min-h-0 flex-1 overflow-hidden bg-slate-100">
              {trajectories.length > 0 ? (
                <div className="h-full min-h-[460px] p-3">
                  <TrajectoryMap
                    trajectories={trajectories}
                    stats={stats}
                    floorPlan={FLOORPLAN_URL}
                    compactMode
                    referencePoint={referencePoint}
                    directionPoint={directionPoint}
                  />
                </div>
              ) : (
                <div
                  ref={planViewportRef}
                  className="relative h-full min-h-[460px] touch-none overflow-hidden"
                  onWheel={handlePlanWheel}
                  onPointerDown={handlePlanPointerDown}
                  onPointerMove={handlePlanPointerMove}
                  onPointerUp={handlePlanPointerUp}
                  onPointerCancel={handlePlanPointerUp}
                  onClick={handlePlanClick}
                >
                  <div
                    data-plan-layer
                    className="absolute left-1/2 top-1/2 w-[min(100%,980px)] origin-center cursor-crosshair will-change-transform"
                    style={{
                      transform: `translate(calc(-50% + ${planPan.x}px), calc(-50% + ${planPan.y}px)) scale(${planZoom})`,
                    }}
                  >
                    <div className="relative overflow-hidden rounded-lg border border-slate-300 bg-white shadow-sm">
                      <img src={FLOORPLAN_URL} alt="План Kerama Marazzi" className="block w-full select-none" draggable={false} />
                      {referencePoint && (
                        <div
                          className="pointer-events-none absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-amber-400 shadow"
                          style={{ left: `${referencePoint.x}%`, top: `${referencePoint.y}%` }}
                        >
                          <span className="absolute left-6 top-1/2 -translate-y-1/2 rounded bg-slate-950 px-1.5 py-0.5 text-[10px] font-bold text-white">START</span>
                        </div>
                      )}
                      {referencePoint && directionPoint && (
                        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                          <defs>
                            <marker id="desktop-direction-arrow" markerWidth="4" markerHeight="4" refX="3.6" refY="2" orient="auto">
                              <path d="M0,0 L4,2 L0,4 Z" fill="#16a34a" />
                            </marker>
                          </defs>
                          <line
                            x1={referencePoint.x}
                            y1={referencePoint.y}
                            x2={directionPoint.x}
                            y2={directionPoint.y}
                            stroke="#16a34a"
                            strokeWidth={0.8 / planZoom}
                            strokeLinecap="round"
                            markerEnd="url(#desktop-direction-arrow)"
                          />
                        </svg>
                      )}
                      {directionPoint && (
                        <div
                          className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white bg-green-600 shadow"
                          style={{ left: `${directionPoint.x}%`, top: `${directionPoint.y}%` }}
                        />
                      )}
                    </div>
                  </div>
                  <div
                    className="absolute bottom-4 left-1/2 z-20 flex -translate-x-1/2 items-center gap-1 rounded-xl border border-slate-200 bg-white/95 p-1.5 shadow-lg backdrop-blur"
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={(event) => event.stopPropagation()}
                  >
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      disabled={planZoom <= PLAN_ZOOM_MIN}
                      onClick={() => zoomPlanAt(planZoom / 1.25)}
                      title="Уменьшить"
                    >
                      <ZoomOut className="h-4 w-4" />
                    </Button>
                    <span className="w-12 select-none text-center font-mono text-xs text-slate-700">{Math.round(planZoom * 100)}%</span>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      disabled={planZoom >= PLAN_ZOOM_MAX}
                      onClick={() => zoomPlanAt(planZoom * 1.25)}
                      title="Увеличить"
                    >
                      <ZoomIn className="h-4 w-4" />
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      disabled={planZoom === 1 && planPan.x === 0 && planPan.y === 0}
                      onClick={resetPlanView}
                      title="Сбросить вид"
                    >
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                  </div>
                  <p className="pointer-events-none absolute left-3 top-3 rounded bg-slate-950/75 px-2 py-1 text-[11px] text-white">
                    Колёсико — масштаб · перетаскивание — сдвиг · клик — точка
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
