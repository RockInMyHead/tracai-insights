import { type MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import TrajectoryMap, { type TrajectoryData, type TurnPoint } from "@/components/TrajectoryMap";
import { apiClient, type VideoAnalysisResult, type VideoListItem } from "@/lib/api";
import { getCameraImportAPI, type CameraImportedVideo, type CameraImportProgress } from "@/lib/cameraImport";
import { Camera, ChevronDown, Download, History, Loader2, MapPinned, Upload } from "lucide-react";
import { toast } from "sonner";

const FLOORPLAN_URL = `${import.meta.env.BASE_URL}floorplans/kerama-marazzi-2025.png`;
const CAMERA_OWNER = "Экшен-камера";

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
  status: "queued" | "processing" | "done" | "error";
  message?: string;
};

function getDesktopBridge() {
  return (window as unknown as { trackai?: Window["trackai"] }).trackai;
}

function getDesktopTrajectory(data: VideoAnalysisResult["data"], videoId: string): TrajectoryData[] {
  if (!data) return [];
  const graphFirst = data.graph_first_trajectory?.length
    ? data.graph_first_trajectory
    : undefined;
  const points = (
    data.map_trajectory?.length
      ? data.map_trajectory
      : graphFirst?.length
        ? graphFirst
        : data.plan_trajectory?.length
          ? data.plan_trajectory
          : data.trajectory
  ) || [];
  if (points.length < 1) return [];
  const uncertaintyMarker = data.graph_first_uncertainty?.marker;
  const competingNextEdges = (
    data.graph_first_uncertainty?.competing_next_edges || []
  ).map((edge) => edge.points.map((point) => ({
    x: Number(point[0]) || 0,
    y: Number(point[1]) || 0,
    z: Number(point[2]) || 0,
  })));
  return [{
    trajectory: points.map((point) => ({ x: Number(point[0]) || 0, y: Number(point[1]) || 0, z: Number(point[2]) || 0 })),
    turnPoints: (data.map_turn_points || data.turn_points || []) as TurnPoint[],
    ownerName: CAMERA_OWNER,
    color: graphFirst && !data.map_trajectory?.length ? "#f59e0b" : "#0f766e",
    videoId,
    method: data.method,
    mapAligned: Boolean(data.map_trajectory?.length || graphFirst?.length),
    uncertain: Boolean(graphFirst?.length && !data.map_trajectory?.length),
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
  const [referencePoint, setReferencePoint] = useState<PlanPoint | null>(null);
  const [directionPoint, setDirectionPoint] = useState<PlanPoint | null>(null);
  const [planPickMode, setPlanPickMode] = useState<"start" | "direction">("start");
  const processingModeRef = useRef<ProcessingMode>("online");
  const analysisQueueRef = useRef<Promise<void>>(Promise.resolve());
  const enqueuedVideoIdsRef = useRef<Set<string>>(new Set());

  const upsertProcessingProgress = useCallback((next: VideoProcessingProgress) => {
    setProcessingProgress((current) => {
      const existingIndex = current.findIndex((item) => item.videoId === next.videoId);
      if (existingIndex === -1) return [...current, next];
      return current.map((item, index) => index === existingIndex ? { ...item, ...next } : item);
    });
  }, []);

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
        const response = await apiClient.getUploadedVideosList();
        setHistory(response.videos || []);
      }
    } catch {
      setHistory([]);
    }
  }, []);

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

  const handlePlanClick = useCallback((event: MouseEvent<HTMLDivElement>) => {
    if (busy) return;
    const rect = event.currentTarget.getBoundingClientRect();
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

  const resetPlanSelection = useCallback(() => {
    if (busy) return;
    setReferencePoint(null);
    setDirectionPoint(null);
    setPlanPickMode("start");
    setMessage("Укажите стартовую точку на плане Kerama Marazzi.");
  }, [busy]);

  const processImportedVideos = useCallback(async (videos: CameraImportedVideo[]) => {
    if (!videos.length) return;
    setState("processing");
    setProgress(null);
    setMessage(`Запускаем анализ: 0 из ${videos.length}`);
    setProcessingProgress((current) => {
      const incomingIds = new Set(videos.map((video) => video.video_id));
      return [
        ...current.filter((item) => !incomingIds.has(item.videoId)),
        ...videos.map((video, index) => ({
          videoId: video.video_id,
          fileName: video.original_filename || video.filename,
          index: index + 1,
          total: videos.length,
          percent: 0,
          status: "queued" as const,
          message: "В очереди",
        })),
      ];
    });

    try {
      for (let index = 0; index < videos.length; index += 1) {
        const video = videos[index];
        const fileName = video.original_filename || video.filename;
        setMessage(`Анализируем ${index + 1} из ${videos.length}: ${fileName}`);
        upsertProcessingProgress({
          videoId: video.video_id,
          fileName,
          index: index + 1,
          total: videos.length,
          percent: 2,
          status: "processing",
          message: "Запущено",
        });
        const videoMode: ProcessingMode = video.localPath ? "local_cpu" : processingModeRef.current;
        if (videoMode !== "online") {
          const local = videoMode === "local_gpu"
            ? await getDesktopBridge()?.localGpu?.process(video)
            : await getDesktopBridge()?.localCpu?.process(video);
          const nextTrajectory = getDesktopTrajectory((local as VideoAnalysisResult | undefined)?.data, video.video_id);
          if (!nextTrajectory.length) throw new Error("Не удалось построить траекторию для этого видео");
          setTrajectories((current) => [...current, ...nextTrajectory]);
          setStats(((local as VideoAnalysisResult).data?.processing_stats || {}) as Record<string, unknown>);
          upsertProcessingProgress({
            videoId: video.video_id,
            fileName,
            index: index + 1,
            total: videos.length,
            percent: 100,
            status: "done",
            message: "Готово",
          });
          continue;
        }
        const started = video.auto_analysis_started
          ? {
              status: "queued",
              analysis_run_id: video.analysis_run_id,
              data: undefined,
            } as VideoAnalysisResult
          : await apiClient.analyzeVideoById(
              video.video_id,
              12.306,
              true,
              video.original_filename || video.filename,
              undefined,
              {
                floorplan_id: "kerama_marazzi_2025",
                reference_point: referencePoint,
                direction_point: directionPoint,
              },
              CAMERA_OWNER,
              "r3",
              undefined,
              true,
            );
        const expectedRunId = started.analysis_run_id;
        let result = started.data;

        if (started.status === "queued") {
          upsertProcessingProgress({
            videoId: video.video_id,
            fileName,
            index: index + 1,
            total: videos.length,
            percent: 5,
            status: "queued",
            message: "Ждет сервер",
          });
          for (let attempt = 0; attempt < 1800; attempt += 1) {
            const status = await apiClient.getProcessingStatus(video.video_id);
            setMessage(status.message || `Обрабатываем ${index + 1} из ${videos.length}`);
            const percent = Number((status as { progress?: unknown }).progress);
            if (["registered", "uploading"].includes(String(status.status || "").toLowerCase())) {
              setProgress({
                index: index + 1,
                total: videos.length,
                fileName,
                filePath: "",
                percent: Number.isFinite(percent) ? Math.max(1, Math.min(99, percent)) : 1,
                phase: "uploading",
              });
              setProcessingProgress((current) => current.filter((item) => item.videoId !== video.video_id));
              await new Promise((resolve) => window.setTimeout(resolve, 2000));
              continue;
            }
            setProgress(null);
            upsertProcessingProgress({
              videoId: video.video_id,
              fileName,
              index: index + 1,
              total: videos.length,
              percent: Number.isFinite(percent) ? Math.max(5, Math.min(99, percent)) : Math.min(95, 8 + attempt),
              status: "processing",
              message: status.message || "Обрабатывается",
            });
            if (status.status === "error" || status.status === "failed") {
              throw new Error(status.message || "Сервер не смог обработать видео");
            }
            const sameRun = !expectedRunId || !status.analysis_run_id || status.analysis_run_id === expectedRunId;
            if (sameRun && ["completed", "done", "success"].includes(status.status)) {
              result = status.result || (await apiClient.getVideoAnalysis(video.video_id)).data;
              break;
            }
            await new Promise((resolve) => window.setTimeout(resolve, 2000));
          }
        }
        const nextTrajectory = getDesktopTrajectory(result, video.video_id);
        if (!nextTrajectory.length) throw new Error("Сервер не вернул траекторию для отображения на плане");
        setTrajectories((current) => [...current, ...nextTrajectory]);
        setStats((result?.processing_stats || {}) as Record<string, unknown>);
        upsertProcessingProgress({
          videoId: video.video_id,
          fileName,
          index: index + 1,
          total: videos.length,
          percent: 100,
          status: "done",
          message: "Готово",
        });
      }
      setState("done");
      setMessage("Готово. Траектория показана на плане Kerama Marazzi.");
      await refreshHistory();
    } catch (error) {
      setProcessingProgress((current) => current.map((item) => (
        item.status === "processing" || item.status === "queued"
          ? { ...item, status: "error", message: error instanceof Error ? error.message : "Ошибка обработки" }
          : item
      )));
      setState("error");
      setMessage(error instanceof Error ? error.message : "Не удалось выполнить анализ");
      await refreshHistory();
    }
  }, [directionPoint, referencePoint, refreshHistory, upsertProcessingProgress]);

  const enqueueImportedVideos = useCallback((videos: CameraImportedVideo[]) => {
    const freshVideos = videos.filter((video) => {
      if (enqueuedVideoIdsRef.current.has(video.video_id)) return false;
      enqueuedVideoIdsRef.current.add(video.video_id);
      return true;
    });
    if (!freshVideos.length) return;
    analysisQueueRef.current = analysisQueueRef.current
      .catch(() => undefined)
      .then(() => processImportedVideos(freshVideos));
  }, [processImportedVideos]);

  useEffect(() => {
    if (!cameraImport) return;
    const unsubscribeProgress = cameraImport.onProgress((next) => {
      setState("copying");
      setProgress(next);
      setMessage(`Загрузка ${next.index} из ${next.total}: ${next.fileName}`);
    });
    const unsubscribeFileImported = cameraImport.onFileImported((video) => {
      enqueueImportedVideos([video as CameraImportedVideo]);
    });
    const unsubscribeComplete = cameraImport.onComplete((videos) => {
      setProgress(null);
      enqueueImportedVideos((videos || []) as CameraImportedVideo[]);
    });
    const unsubscribeError = cameraImport.onError((error) => {
      const text = error.message || "Не удалось загрузить видео с камеры";
      if (processingProgress.some((item) => item.status === "processing" || item.status === "queued")) {
        toast.error(text);
        return;
      }
      setState("error");
      setMessage(text);
    });
    return () => {
      unsubscribeProgress();
      unsubscribeFileImported();
      unsubscribeComplete();
      unsubscribeError();
    };
  }, [cameraImport, enqueueImportedVideos, processingProgress]);

  useEffect(() => {
    if (!cameraImport) return;
    let cancelled = false;
    const syncServerProgress = async () => {
      if (processingModeRef.current !== "online") return;
      try {
        const response = await apiClient.getUploadedVideosList();
        if (cancelled) return;
        const activeVideos = (response.videos || []).filter((video) => {
          const status = String(video.status || "").toLowerCase();
          const progressValue = Number(video.progress || 0);
          const isDesktop = video.client_source === "desktop" || /VID\d+\.(avi|mp4|mov|mkv)$/i.test(video.original_filename || video.filename);
          if (!isDesktop) return false;
          return (
            ["registered", "uploading", "uploaded", "queued", "processing", "running", "gpu_processing", "error", "failed"].includes(status)
            || (progressValue > 0 && progressValue < 100)
          );
        }).slice(0, 8);
        activeVideos.forEach((video, index) => {
          const progressValue = Number(video.progress || 0);
          const status = String(video.status || "").toLowerCase();
          if (["registered", "uploading"].includes(status)) {
            setProgress({
              index: index + 1,
              total: activeVideos.length,
              fileName: video.original_filename || video.filename,
              filePath: "",
              percent: Number.isFinite(progressValue) ? Math.max(1, Math.min(99, progressValue)) : 1,
              phase: "uploading",
            });
            setProcessingProgress((current) => current.filter((item) => item.videoId !== video.video_id));
            return;
          }
          setProgress((current) => current?.fileName === (video.original_filename || video.filename) ? null : current);
          upsertProcessingProgress({
            videoId: video.video_id,
            fileName: video.original_filename || video.filename,
            index: index + 1,
            total: activeVideos.length,
            percent: Number.isFinite(progressValue) ? Math.max(0, Math.min(99, progressValue)) : 0,
            status: ["error", "failed"].includes(status) ? "error" : status === "completed" ? "done" : "processing",
            message: video.message || (status ? `Сервер: ${status}` : "Сервер обрабатывает"),
          });
        });
      } catch {
        // The import flow still reports hard errors; this poller is only a UI recovery path.
      }
    };
    void syncServerProgress();
    const timer = window.setInterval(() => {
      void syncServerProgress();
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [cameraImport, upsertProcessingProgress]);

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
    try {
      await resolveProcessingMode();
      await cameraImport.setSettings({ enabled: true, ownerName: CAMERA_OWNER });
      await cameraImport.resetImportedState();
      const status = await cameraImport.scanNow({
        forceImport: true,
        ignoreImported: true,
        analysisContext: {
          floorplan_id: "kerama_marazzi_2025",
          reference_point: referencePoint,
          direction_point: directionPoint,
          employee_name: CAMERA_OWNER,
          analysis_method: "r3",
          scale_factor: 12.306,
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
        : await apiClient.getVideoAnalysis(video.video_id);
      if (!result) throw new Error("Результат анализа не найден");
      const next = getDesktopTrajectory(result.data, video.video_id);
      if (!next.length) throw new Error("Для этого видео ещё нет готовой траектории");
      setTrajectories(next);
      setStats((result?.data?.processing_stats || {}) as Record<string, unknown>);
      setState("done");
      setMessage(`Открыта траектория: ${video.filename}`);
      setHistoryOpen(false);
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Не удалось открыть результат");
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
    try {
      const result = await cameraImport?.cancel?.();
      setProgress(null);
      if (result?.cancelled) {
        setMessage("Выгрузка с камеры остановлена. Уже скопированные видео останутся в обработке.");
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Не удалось остановить выгрузку с камеры");
    }
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
  const canCancelImport = state === "copying" || (progress && state !== "processing");
  const processingStatusLabel: Record<VideoProcessingProgress["status"], string> = {
    queued: "В очереди",
    processing: "Обработка",
    done: "Готово",
    error: "Ошибка",
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
          <p className={`mt-4 text-sm ${state === "error" ? "text-rose-700" : state === "needs_camera" ? "text-amber-700" : "text-slate-600"}`}>{message}</p>
          {progress && <div className="mt-3">
            <div className="mb-1 flex items-center justify-between gap-3 text-xs font-medium text-slate-600">
              <span className="shrink-0 font-semibold text-slate-700">Загрузка</span>
              <span className="min-w-0 flex-1 truncate text-slate-500">{progress.fileName}</span>
              <span className="shrink-0">{Math.round(progress.percent)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-slate-200">
              <div className="h-full rounded-full bg-teal-600 transition-all" style={{ width: `${Math.max(0, Math.min(100, progress.percent))}%` }} />
            </div>
            <p className="mt-1 text-xs text-teal-700">На сервер: {progress.index} из {progress.total}</p>
          </div>}
          {processingProgress.length > 0 && <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wide text-slate-500">
              <span>Обработка видео</span>
              <span>{processingProgress.filter((item) => item.status === "done").length}/{processingProgress.length}</span>
            </div>
            {processingProgress.slice(-6).map((item) => (
              <div key={item.videoId} className="rounded-lg border border-slate-200 bg-white p-2">
                <div className="flex items-center justify-between gap-3 text-xs">
                  <span className="min-w-0 truncate font-medium text-slate-800">{item.fileName}</span>
                  <span className={`shrink-0 font-semibold ${item.status === "error" ? "text-rose-700" : item.status === "done" ? "text-teal-700" : "text-slate-600"}`}>
                    {processingStatusLabel[item.status]} {Math.round(item.percent)}%
                  </span>
                </div>
                <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-slate-200">
                  <div
                    className={`h-full rounded-full transition-all ${item.status === "error" ? "bg-rose-600" : item.status === "done" ? "bg-teal-600" : "bg-blue-600"}`}
                    style={{ width: `${Math.max(0, Math.min(100, item.percent))}%` }}
                  />
                </div>
                {item.message && <p className="mt-1 truncate text-[11px] text-slate-500">{item.index} из {item.total}: {item.message}</p>}
              </div>
            ))}
          </div>}
        </div>
        <div className="min-h-[520px] overflow-hidden rounded-2xl border border-slate-200 bg-slate-100 p-2 shadow-sm shadow-slate-300/40">
          {trajectories.length ? <TrajectoryMap trajectories={trajectories} stats={stats} floorPlan={FLOORPLAN_URL} compactMode /> : (
            <div className="flex h-full min-h-[500px] flex-col bg-white">
              <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
                <div>
                  <p className="text-sm font-semibold text-slate-950">План Kerama Marazzi</p>
                  <p className="text-xs text-slate-500">{referencePoint ? "2. Укажите направление первого движения" : "1. Укажите стартовую точку"}</p>
                </div>
                <div className="flex items-center gap-2">
                  <Button type="button" size="sm" variant={planPickMode === "start" ? "default" : "outline"} disabled={busy} onClick={() => setPlanPickMode("start")}>Старт</Button>
                  <Button type="button" size="sm" variant={planPickMode === "direction" ? "default" : "outline"} disabled={busy || !referencePoint} onClick={() => setPlanPickMode("direction")}>Направление</Button>
                  <Button type="button" size="sm" variant="ghost" disabled={busy || (!referencePoint && !directionPoint)} onClick={resetPlanSelection}>Сбросить</Button>
                </div>
              </div>
              <div className="relative flex-1 overflow-auto bg-slate-100 p-3">
                <div className="relative mx-auto w-full max-w-[980px] cursor-crosshair overflow-hidden rounded-lg border border-slate-300 bg-white" onClick={handlePlanClick}>
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
                        strokeWidth="0.8"
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
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
