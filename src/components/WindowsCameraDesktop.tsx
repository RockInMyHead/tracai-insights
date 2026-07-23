import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import TrajectoryMap, { type TrajectoryData, type TurnPoint } from "@/components/TrajectoryMap";
import { apiClient, type VideoAnalysisResult, type VideoListItem } from "@/lib/api";
import { getCameraImportAPI, type CameraImportedVideo, type CameraImportProgress } from "@/lib/cameraImport";
import { Camera, ChevronDown, History, Loader2, MapPinned, Upload } from "lucide-react";

const FLOORPLAN_URL = "/floorplans/kerama-marazzi-2025.png";
const CAMERA_OWNER = "Экшен-камера";

type DesktopState = "ready" | "looking" | "copying" | "processing" | "done" | "needs_camera" | "error";
type ProcessingMode = "online" | "local";

function getDesktopBridge() {
  return (window as unknown as { trackai?: Window["trackai"] }).trackai;
}

function getDesktopTrajectory(data: VideoAnalysisResult["data"], videoId: string): TrajectoryData[] {
  if (!data) return [];
  const points = (data.map_trajectory?.length ? data.map_trajectory : data.plan_trajectory?.length ? data.plan_trajectory : data.trajectory) || [];
  if (points.length < 2) return [];
  return [{
    trajectory: points.map((point) => ({ x: Number(point[0]) || 0, y: Number(point[1]) || 0, z: Number(point[2]) || 0 })),
    turnPoints: (data.map_turn_points || data.turn_points || []) as TurnPoint[],
    ownerName: CAMERA_OWNER,
    color: "#0f766e",
    videoId,
    method: data.method,
    mapAligned: Boolean(data.map_trajectory?.length),
  }];
}

export default function WindowsCameraDesktop() {
  const cameraImport = useMemo(() => getCameraImportAPI(), []);
  const [state, setState] = useState<DesktopState>("ready");
  const [message, setMessage] = useState("Подключите экшен-камеру и нажмите «Загрузить»");
  const [progress, setProgress] = useState<CameraImportProgress | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState<VideoListItem[]>([]);
  const [trajectories, setTrajectories] = useState<TrajectoryData[]>([]);
  const [stats, setStats] = useState<Record<string, unknown>>({});
  const processingModeRef = useRef<ProcessingMode>("online");

  const resolveProcessingMode = useCallback(async () => {
    const bridge = getDesktopBridge();
    const next = await bridge?.processing?.resolveMode?.();
    const mode = next?.mode === "local" ? "local" : "online";
    processingModeRef.current = mode;
    return mode;
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      if (processingModeRef.current === "local") {
        const items = await getDesktopBridge()?.localCpu?.history?.();
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

  const processImportedVideos = useCallback(async (videos: CameraImportedVideo[]) => {
    if (!videos.length) return;
    setState("processing");
    setProgress(null);
    setMessage(`Запускаем анализ: 0 из ${videos.length}`);

    try {
      for (let index = 0; index < videos.length; index += 1) {
        const video = videos[index];
        setMessage(`Анализируем ${index + 1} из ${videos.length}: ${video.original_filename || video.filename}`);
        if (processingModeRef.current === "local") {
          const local = await getDesktopBridge()?.localCpu?.process(video);
          const nextTrajectory = getDesktopTrajectory((local as VideoAnalysisResult | undefined)?.data, video.video_id);
          if (!nextTrajectory.length) throw new Error("Не удалось построить траекторию для этого видео");
          setTrajectories((current) => [...current, ...nextTrajectory]);
          setStats(((local as VideoAnalysisResult).data?.processing_stats || {}) as Record<string, unknown>);
          continue;
        }
        const started = await apiClient.analyzeVideoById(
          video.video_id,
          12.306,
          true,
          video.original_filename || video.filename,
          undefined,
          { floorplan_id: "kerama_marazzi_2025" },
          CAMERA_OWNER,
          "r3",
          undefined,
          true,
        );
        const expectedRunId = started.analysis_run_id;
        let result = started.data;

        if (started.status === "queued") {
          for (let attempt = 0; attempt < 1800; attempt += 1) {
            const status = await apiClient.getProcessingStatus(video.video_id);
            setMessage(status.message || `Обрабатываем ${index + 1} из ${videos.length}`);
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
      }
      setState("done");
      setMessage("Готово. Траектория показана на плане Kerama Marazzi.");
      await refreshHistory();
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Не удалось выполнить анализ");
      await refreshHistory();
    }
  }, [refreshHistory]);

  useEffect(() => {
    if (!cameraImport) return;
    const unsubscribeProgress = cameraImport.onProgress((next) => {
      setState("copying");
      setProgress(next);
      setMessage(`Копируем видео ${next.index + 1} из ${next.total}: ${next.fileName}`);
    });
    const unsubscribeComplete = cameraImport.onComplete((videos) => {
      void processImportedVideos(videos as CameraImportedVideo[]);
    });
    const unsubscribeError = cameraImport.onError((error) => {
      setState("error");
      setMessage(error.message || "Не удалось загрузить видео с камеры");
    });
    return () => {
      unsubscribeProgress();
      unsubscribeComplete();
      unsubscribeError();
    };
  }, [cameraImport, processImportedVideos]);

  const handleUpload = async () => {
    if (!cameraImport) {
      setState("error");
      setMessage("Откройте это окно в приложении TrackAI для Windows.");
      return;
    }
    setState("looking");
    setMessage("Проверяем подключение и ищем экшен-камеру...");
    try {
      await resolveProcessingMode();
      await cameraImport.setSettings({ enabled: true, ownerName: CAMERA_OWNER });
      const status = await cameraImport.scanNow({ forceImport: true });
      if (!status.volumes?.length) {
        setState("needs_camera");
        setMessage("Камера не найдена. Подключите её по USB, разблокируйте накопитель и нажмите «Загрузить» ещё раз.");
      } else if (!status.pendingFiles?.length && !status.importing) {
        setState("ready");
        setMessage("На подключённой камере нет новых видео.");
      }
    } catch (error) {
      setState("error");
      setMessage(error instanceof Error ? error.message : "Не удалось прочитать камеру");
    }
  };

  const openHistoryItem = async (video: VideoListItem) => {
    try {
      const result = processingModeRef.current === "local"
        ? await getDesktopBridge()?.localCpu?.analysis(video.video_id) as { data?: VideoAnalysisResult["data"] }
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

  const busy = ["looking", "copying", "processing"].includes(state);
  const buttonText = busy ? "Выполняется" : "Загрузить";

  return (
    <main className="min-h-[100dvh] bg-slate-950 text-slate-50">
      <header className="flex h-16 items-center justify-between border-b border-slate-800 px-6">
        <div className="flex items-center gap-3 font-semibold tracking-tight"><MapPinned className="h-5 w-5 text-teal-400" />TrackAI</div>
        <div className="relative">
          <Button variant="ghost" className="gap-2 text-slate-200 hover:bg-slate-800 hover:text-white" onClick={() => { setHistoryOpen((open) => !open); void refreshHistory(); }}>
            <History className="h-4 w-4" />История<ChevronDown className="h-4 w-4" />
          </Button>
          {historyOpen && <div className="absolute right-0 top-11 z-10 w-96 overflow-hidden rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
            {history.length ? history.slice(0, 12).map((video) => <button key={video.video_id} onClick={() => void openHistoryItem(video)} className="block w-full border-b border-slate-800 px-4 py-3 text-left text-sm last:border-0 hover:bg-slate-800"><span className="block truncate text-slate-100">{video.filename}</span><span className="text-xs text-slate-400">{video.has_analysis ? "Траектория готова" : "В обработке"}</span></button>) : <p className="px-4 py-5 text-sm text-slate-400">История пока пуста</p>}
          </div>}
        </div>
      </header>

      <section className="mx-auto grid max-w-6xl gap-8 px-6 py-10 lg:grid-cols-[360px_1fr]">
        <div className="flex min-h-[430px] flex-col justify-center">
          <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-400 text-slate-950"><Camera className="h-6 w-6" /></div>
          <h1 className="text-4xl font-semibold tracking-tight">Видео с камеры</h1>
          <p className="mt-3 max-w-sm leading-6 text-slate-400">Подключите экшен-камеру. TrackAI скопирует новые видео, обработает их и покажет маршрут на плане.</p>
          <Button size="lg" className="mt-8 h-14 w-full gap-3 bg-teal-400 text-base font-semibold text-slate-950 hover:bg-teal-300 active:translate-y-px" disabled={busy} onClick={() => void handleUpload()}>
            {busy ? <Loader2 className="h-5 w-5 animate-spin" /> : <Upload className="h-5 w-5" />}{buttonText}
          </Button>
          <p className={`mt-4 text-sm ${state === "error" ? "text-rose-300" : state === "needs_camera" ? "text-amber-300" : "text-slate-400"}`}>{message}</p>
          {progress && <p className="mt-2 text-xs text-teal-300">{Math.round(progress.percent)}% скопировано</p>}
        </div>
        <div className="min-h-[520px] overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 p-2">
          {trajectories.length ? <TrajectoryMap trajectories={trajectories} stats={stats} floorPlan={FLOORPLAN_URL} compactMode /> : <div className="flex h-full min-h-[500px] items-center justify-center bg-slate-950/40 text-center text-sm text-slate-500">После обработки здесь появится траектория<br />на плане Kerama Marazzi</div>}
        </div>
      </section>
    </main>
  );
}
