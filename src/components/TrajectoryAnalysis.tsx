import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Video, MapPin, Activity, Clock, Navigation, Loader2, User, X, Plus, FolderOpen, Eye } from "lucide-react";
import { toast } from "sonner";
import { apiClient, R3TrajectorySource, VideoAnalysisResult, VideoListItem } from "@/lib/api";
import { finiteNum } from "@/lib/numbers";
import RealTimeR3Visualization from "./RealTimeR3Visualization";
import ProcessingDashboard from "./ProcessingDashboard";

/** Временно скрыть в UI: масштаб, detect/turn/ml roi, подсказки и переключатель стабилизации (значения по умолчанию в коде сохраняются). */
const SHOW_ADVANCED_ANALYSIS_SETTINGS = false;

const FLOORPLAN_DIAGNOSTIC_LABELS: Record<string, string> = {
  route_length_changed_significantly: "маршрут перестроен по проходам",
  large_map_correction_applied: "применена большая коррекция по плану",
  walking_speed_prior_inconsistent: "скорость не согласуется с метрической моделью",
  start_projected_to_walkable_area: "старт перенесён в проходимую зону",
  no_collision_free_route: "между точками нет связного прохода",
  different_walkable_components: "точки попали в разные связные зоны маски",
  no_walkable_segment_endpoint: "конец сегмента не попал в проходимую область",
  local_search_exhausted: "локальный поиск не нашёл безопасный обход",
  detour_spike_rejected: "найденный обход отклонён как аномальная петля",
  corridor_graph_recovery_failed: "граф коридоров не подтвердил безопасный обход",
  corridor_graph_topology_recovery_applied: "аномальный A* обход заменён маршрутом по графу коридоров",
  authoritative_safe_map_fallback: "использован безопасный графовый маршрут по фиксированному плану",
  independent_monocular_rescue: "использован независимый монокулярный LingBot",
  independent_residual_collision: "независимый LingBot пересекает запрещённую область",
  insufficient_independent_net_progress: "траектория LingBot слишком сжата или возвращается к старту",
  compressed_or_looping_independent_observation: "наблюдение LingBot не даёт достаточного перемещения по плану",
  ambiguous_independent_map_alignment: "для LingBot найдено несколько неоднозначных привязок к плану",
  independent_scale_or_topology_ambiguous: "масштаб или ветвь прохода для LingBot неоднозначны",
  implausible_metric_scale: "масштаб не согласуется со временем и скоростью движения",
  implausible_corrected_metric_scale: "после обхода препятствий нарушилась метрическая модель",
  metric_prior_inconsistent: "не найден физически правдоподобный масштаб",
  metric_prior_unavailable: "нет надёжной временной шкалы для независимого масштаба",
  start_too_far_from_walkable_area: "старт слишком далеко от проходимой области",
  final_publication_invariant_failed: "итоговая линия не прошла проверку безопасности",
  unsafe_or_disconnected_final_polyline: "итоговый маршрут небезопасен или несвязен",
  constraint_solution_not_found: "не найден допустимый маршрут по маске",
  topology_destroying_map_correction: "коррекция плана ломает форму маршрута",
  map_correction_exceeds_observation_budget: "коррекция плана превышает бюджет наблюдения",
};

const LINGBOT_FUSION_REASON_LABELS: Record<string, string> = {
  trajectory_disagreement_too_large: "траектории слишком расходятся",
  turn_chirality_conflict: "разный знак поворотов",
  trajectory_too_short: "слишком короткая траектория",
  similarity_fit_failed: "не удалось совместить траектории",
  insufficient_net_progress: "траектория слишком сжата или возвращается к старту",
};

const MAP_OBSERVATION_SOURCE_LABELS: Record<string, string> = {
  r3_lingbot_fusion: "на карте: R³ + LingBot fusion",
  lingbot_independent: "на карте: независимый LingBot",
  r3: "на карте: R³",
};

const floorplanDiagnosticLabel = (value: unknown): string => {
  const key = String(value || "constraint_solution_not_found");
  return FLOORPLAN_DIAGNOSTIC_LABELS[key] || key;
};

const lingbotFusionReasonLabel = (value: unknown): string => {
  const key = String(value || "");
  return LINGBOT_FUSION_REASON_LABELS[key] || key;
};

const mapObservationSourceLabel = (value: string): string => {
  return MAP_OBSERVATION_SOURCE_LABELS[value] || "";
};

type AnalysisData = NonNullable<VideoAnalysisResult["data"]>;

const convertTrajectory = (traj: unknown): { x: number; y: number; z?: number }[] => {
  if (!traj || !Array.isArray(traj) || traj.length === 0) return [];
  if (Array.isArray(traj[0])) {
    return traj.map((point: unknown) => {
      const arr = point as number[];
      return {
        x: finiteNum(arr[0]),
        y: finiteNum(arr[1]),
        z: finiteNum(arr[2]),
      };
    });
  }
  if (typeof traj[0] === "object" && traj[0] !== null) {
    return traj.map((p: unknown) => {
      const o = p as Record<string, unknown> & { 0?: unknown; 1?: unknown; 2?: unknown };
      return {
        x: finiteNum(o.x ?? o[0]),
        y: finiteNum(o.y ?? o[1]),
        z: finiteNum(o.z ?? o[2]),
      };
    });
  }
  return [];
};

// Интерфейс для видео с владельцем (локальный файл или с сервера)
interface VideoWithOwner {
  id: string;
  file?: File;
  video_id?: string; // ID на сервере (если уже загружено)
  serverFilename?: string; // имя файла на сервере
  ownerName: string;
  analysisResult?: AnalysisData;
  allowManualUpdates?: boolean;
  isAnalyzing?: boolean;
  uploadProgress?: number;
  color: string;
  uploadedAt?: number;
}

interface Employee {
  name: string;
  color: string;
}

interface TrajectoryData {
  trajectory: number[][] | { x: number; y: number; z?: number }[];
  turnPoints: Record<string, unknown>[];
  ownerName: string;
  color: string;
  videoId?: string;
  method?: string;
  coordinateConvention?: string;
  mapAligned?: boolean;
  manualPlanSpace?: boolean;
  r3CameraPoints?: number[][];  // все позиции камер R³
  mapScaleFactor?: number;
  r3AutoFitToPlan?: boolean;
}

type R3TrajectoryResponse = Awaited<ReturnType<typeof apiClient.getR3Trajectory>>;

const mergeR3TrajectoryResponse = (
  analysisData: AnalysisData,
  response: R3TrajectoryResponse,
  rawTrajectory3d?: number[][],
  extraStats: Record<string, unknown> = {},
): AnalysisData => {
  const planTrajectory = Array.isArray(response.plan_trajectory ?? response.trajectory)
    ? (response.plan_trajectory ?? response.trajectory).filter(
        (point) => Array.isArray(point) && point.length >= 2,
      )
    : [];
  if (!response.success || planTrajectory.length < 2) return analysisData;
  const mapTrajectory = Array.isArray(response.map_trajectory)
    ? response.map_trajectory.filter(
        (point) => Array.isArray(point) && point.length >= 2,
      )
    : [];
  const displayTrajectory = mapTrajectory.length >= 2 ? mapTrajectory : planTrajectory;

  const selectedSource = response.trajectory_source ?? "raw";
  const trajectoryQuality = response.trajectory_quality ?? {};
  const cleanedDistance = trajectoryQuality.cleaned_distance;
  const currentStats =
    (analysisData.processing_stats as Record<string, unknown> | undefined) || {};
  const responseStats = response.processing_stats ?? {};
  const responseEstimatedDistance = finiteNum(responseStats.estimated_distance, Number.NaN);
  const mergedRawTrajectory = Array.isArray(rawTrajectory3d) && rawTrajectory3d.length >= 2
    ? rawTrajectory3d
    : analysisData.raw_trajectory_3d;

  return {
    ...analysisData,
    method: selectedSource === "scale_aware_candidate"
      ? "r3_reconstruction_scale_aware"
      : selectedSource === "robust_candidate"
        ? "r3_reconstruction_robust_candidate"
        : "r3_reconstruction",
    trajectory: displayTrajectory,
    plan_trajectory: planTrajectory,
    map_trajectory: mapTrajectory.length >= 2 ? mapTrajectory : undefined,
    raw_trajectory_3d: mergedRawTrajectory,
    r3_camera_points: mergedRawTrajectory,
    r3_source_frame_indices: response.source_frame_indices ?? [],
    r3_source_timestamps_seconds: response.source_timestamps_seconds ?? [],
    r3_pose_graph: response.pose_graph,
    r3_pose_graph_candidate: response.pose_graph_candidate,
    r3_scale_aware_candidate: response.scale_aware_candidate,
    turn_points: Array.isArray(response.turn_points)
      ? response.turn_points
      : analysisData.turn_points,
    map_turn_points: Array.isArray(response.map_turn_points)
      ? response.map_turn_points
      : undefined,
    floorplan_constraint: response.floorplan_constraint,
    lingbot_fusion_candidate:
      response.lingbot_fusion_candidate ?? analysisData.lingbot_fusion_candidate,
    lingbot_shadow: response.lingbot_shadow ?? analysisData.lingbot_shadow,
    trajectory_points: displayTrajectory.length,
    processing_stats: {
      ...currentStats,
      ...extraStats,
      ...responseStats,
      estimated_distance:
        Number.isFinite(responseEstimatedDistance)
          ? responseEstimatedDistance
          : typeof cleanedDistance === "number" && Number.isFinite(cleanedDistance)
          ? cleanedDistance
          : currentStats.estimated_distance,
      turns_detected: Array.isArray(response.turn_points)
        ? response.turn_points.length
        : currentStats.turns_detected,
      r3_trajectory_quality: trajectoryQuality,
      r3_trajectory_source: selectedSource,
      r3_trajectory_source_requested: response.trajectory_source_requested ?? selectedSource,
      r3_trajectory_source_fallback_reason:
        response.trajectory_source_fallback_reason ?? null,
      r3_trajectory_source_selection: response.trajectory_source_selection ?? {},
      floorplan_constraint: response.floorplan_constraint ?? {},
      map_matching_applied: mapTrajectory.length >= 2,
      map_trajectory_points: mapTrajectory.length,
    },
  } as AnalysisData;
};

const trajectoryDataFromVideo = (video: VideoWithOwner): TrajectoryData => {
  const result = video.analysisResult as AnalysisData;
  const stats = result.processing_stats as Record<string, unknown> | undefined;
  const manualOverride = Boolean(stats?.manual_override);
  const method = String(result.method || "");
  const isR3 = method.startsWith("r3");
  const isLingBot = method === "lingbot_map";
  const hasMapTrajectory = Boolean(result.map_trajectory);
  const isAlreadyInPlanSpace = (hasMapTrajectory && !isLingBot) || manualOverride;
  const floorplanConstraint = (
    result.floorplan_constraint || stats?.floorplan_constraint || {}
  ) as Record<string, unknown>;
  const mapRejected = isR3 && floorplanConstraint.accepted === false;
  const r3Quality = stats?.r3_trajectory_quality as Record<string, unknown> | undefined;
  const r3Projection = r3Quality?.projection as Record<string, unknown> | undefined;
  return {
    trajectory: convertTrajectory(
      mapRejected ? [] : (result.map_trajectory || result.plan_trajectory || result.trajectory),
    ),
    turnPoints: result.map_turn_points || result.turn_points || [],
    ownerName: video.ownerName,
    color: video.color,
    videoId: video.video_id,
    method: result.method,
    coordinateConvention: String(r3Projection?.plan_coordinate_convention || "") || undefined,
    mapAligned: isAlreadyInPlanSpace,
    manualPlanSpace: manualOverride,
    r3CameraPoints: isR3 && Array.isArray(result.plan_trajectory || result.trajectory)
      ? (result.plan_trajectory || result.trajectory)
      : undefined,
    mapScaleFactor: (isR3 || isLingBot) ? 1 : finiteNum(stats?.scale_factor, 1),
    r3AutoFitToPlan: (isR3 || isLingBot) && !isAlreadyInPlanSpace && !mapRejected,
  };
};

interface TrajectoryAnalysisProps {
  onTrajectoryAnalyzed?: (
    trajectory: number[][] | { x: number; y: number; z?: number }[],
    turnPoints: Record<string, unknown>[],
    stats: Record<string, unknown>,
    trajectories?: TrajectoryData[]
  ) => void;
  floorPlan?: string | null;
  drawnPlan?: unknown[] | null;
  referencePoint?: { x: number; y: number } | null;
  directionPoint?: { x: number; y: number } | null;
}

const TrajectoryAnalysis = ({ onTrajectoryAnalyzed, floorPlan: externalFloorPlan, drawnPlan, referencePoint: externalReferencePoint, directionPoint: externalDirectionPoint }: TrajectoryAnalysisProps) => {
  const [videos, setVideos] = useState<VideoWithOwner[]>([]);
  const [currentOwnerName, setCurrentOwnerName] = useState('');
  const [scaleFactor, setScaleFactor] = useState<number>(12.306);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [liveProgress, setLiveProgress] = useState(0);
  const [liveStatus, setLiveStatus] = useState('queued');
  const [liveStage, setLiveStage] = useState('queued');
  const [liveElapsedSeconds, setLiveElapsedSeconds] = useState<number | null>(null);
  const [liveEtaSeconds, setLiveEtaSeconds] = useState<number | null>(null);
  const [liveOwnerName, setLiveOwnerName] = useState('');
  const [analysisStartedAtMs, setAnalysisStartedAtMs] = useState<number | null>(null);
  const [batchIndex, setBatchIndex] = useState(0);
  const [batchTotal, setBatchTotal] = useState(1);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [serverVideos, setServerVideos] = useState<VideoListItem[]>([]);
  const [selectedServerVideos, setSelectedServerVideos] = useState<VideoListItem[]>([]);
  const [showServerPicker, setShowServerPicker] = useState(false);
  const [loadingServerVideos, setLoadingServerVideos] = useState(false);
  const [stabilizationEnabled, setStabilizationEnabled] = useState<boolean>(true);
  const [detectInterval, setDetectInterval] = useState<number>(5);
  const [turnVoteThreshold, setTurnVoteThreshold] = useState<number>(3);
  const [useMlRoi, setUseMlRoi] = useState<boolean>(true);
  const [analysisMethod] = useState<'slam' | 'r3' | 'lingbot'>('r3');
  const r3TrajectorySource: R3TrajectorySource = 'scale_aware_candidate';
  const [liveViewVideoId, setLiveViewVideoId] = useState<string | null>(null);
  const [showLiveView, setShowLiveView] = useState(false);
  const [floorPlan, setFloorPlan] = useState<string | null>(null);
  const [floorPlanFile, setFloorPlanFile] = useState<File | null>(null);
  const [referencePoint, setReferencePoint] = useState(null);
  const [existingOwners, setExistingOwners] = useState<string[]>([]);
  const manualTrajectoryVersionsRef = useRef<Record<string, string>>({});
  const manualSuppressedVideoIdsRef = useRef<Set<string>>(new Set());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const floorPlanInputRef = useRef<HTMLInputElement>(null);

  // Цвета для разных пользователей
  const userColors = [
    '#3b82f6', // blue
    '#ef4444', // red
    '#10b981', // emerald
    '#f59e0b', // amber
    '#8b5cf6', // violet
    '#06b6d4', // cyan
    '#84cc16', // lime
    '#f97316', // orange
    '#ec4899', // pink
    '#6b7280', // gray
  ];

  // Для обратной совместимости с отображением результатов
  const firstAnalyzedVideo = videos.find(v => v.analysisResult);
  const analysisResult = firstAnalyzedVideo ? {
    data: firstAnalyzedVideo.analysisResult
  } : null;
  const activeAnalysisStats = (
    analysisResult?.data?.processing_stats || {}
  ) as Record<string, unknown>;
  const activeFloorplanConstraint = (
    activeAnalysisStats.floorplan_constraint || {}
  ) as Record<string, unknown>;
  const activeFloorplanWarnings = Array.isArray(activeFloorplanConstraint.quality_warnings)
    ? activeFloorplanConstraint.quality_warnings.map(floorplanDiagnosticLabel)
    : [];
  const activeFloorplanRejections = Array.isArray(activeFloorplanConstraint.rejection_reasons)
    ? activeFloorplanConstraint.rejection_reasons.map(floorplanDiagnosticLabel)
    : [];
  const activeLingbotFusion = (
    activeAnalysisStats.lingbot_fusion || {}
  ) as Record<string, unknown>;
  const activeMapObservationSource = String(
    activeAnalysisStats.map_observation_source || "",
  );

  // Загрузка плана из localStorage при инициализации
  useEffect(() => {
    const savedFloorPlan = localStorage.getItem('floorPlan');
    if (savedFloorPlan) {
      if (savedFloorPlan.length > 1_500_000) {
        localStorage.removeItem('floorPlan');
      } else {
        setFloorPlan(savedFloorPlan);
      }
    }
  }, []);

  // Загрузка настройки стабилизации и имен сотрудников
  useEffect(() => {
    const savedStabilization = localStorage.getItem('stabilizationEnabled');
    if (savedStabilization !== null) {
      setStabilizationEnabled(JSON.parse(savedStabilization));
    }

    const savedOwners = localStorage.getItem('trackai_owners');
    if (savedOwners) {
      try {
        setExistingOwners(JSON.parse(savedOwners));
      } catch (e) {
        console.error("Failed to load owners", e);
      }
    }
  }, []);

  useEffect(() => {
    if (!onTrajectoryAnalyzed) return;
    const trackedVideos = videos.filter((v) => v.video_id && v.allowManualUpdates);
    if (trackedVideos.length === 0) return;

    let cancelled = false;

    const fetchManualUpdates = async () => {
      for (const video of trackedVideos) {
        if (!video.video_id) continue;
        if (manualSuppressedVideoIdsRef.current.has(video.video_id)) continue;
        try {
          const manual = await apiClient.getManualTrajectory(video.video_id);
          if (
            cancelled ||
            manualSuppressedVideoIdsRef.current.has(video.video_id) ||
            !manual.exists ||
            !Array.isArray(manual.trajectory)
          ) continue;

          const version = manual.updated_at || `${manual.trajectory.length}`;
          if (
            manualSuppressedVideoIdsRef.current.has(video.video_id) ||
            manualTrajectoryVersionsRef.current[video.video_id] === version
          ) continue;
          manualTrajectoryVersionsRef.current[video.video_id] = version;

          const converted = convertTrajectory(manual.trajectory);
          if (converted.length < 2) continue;

          const manualStats = {
            manual_override: true,
            scale_factor: 1,
            trajectory_points: converted.length,
          };
          const manualResult = {
            method: "manual_admin",
            trajectory: manual.trajectory,
            turn_points: manual.turn_points || [],
            processing_stats: manualStats,
          } as AnalysisData;

          setVideos((prev) =>
            prev.map((v) =>
              v.video_id === video.video_id
                ? { ...v, analysisResult: manualResult, isAnalyzing: false }
                : v
            )
          );
          onTrajectoryAnalyzed(
            converted,
            manual.turn_points || [],
            manualStats,
            [
              {
                trajectory: converted,
                turnPoints: manual.turn_points || [],
                ownerName: video.ownerName,
                color: video.color,
                mapAligned: true,
                manualPlanSpace: true,
              },
            ]
          );
          toast.success(`Администратор отправил ручную траекторию для ${video.ownerName}`);
        } catch {
          // Ручная траектория может еще не существовать.
        }
      }
    };

    fetchManualUpdates();
    const interval = window.setInterval(fetchManualUpdates, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [videos, onTrajectoryAnalyzed]);

  const handleStabilizationToggle = () => {
    const newValue = !stabilizationEnabled;
    setStabilizationEnabled(newValue);
    localStorage.setItem('stabilizationEnabled', JSON.stringify(newValue));
  };

  const handleFloorPlanUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        toast.error("Пожалуйста, загрузите изображение плана");
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result as string;
        setFloorPlan(result);
        setFloorPlanFile(file);
        try {
          localStorage.setItem('floorPlan', result);
        } catch (storageError) {
          console.warn("Could not save floor plan to localStorage (quota exceeded)", storageError);
          // We continue anyway, it's just not saved between refreshes
        }
        toast.success(`План "${file.name}" загружен`);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFloorPlanRemove = () => {
    setFloorPlan(null);
    setFloorPlanFile(null);
    localStorage.removeItem('floorPlan');
    toast.info("План помещения удален");
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const newFiles: File[] = [];
      const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.3gp'];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const isVideoMime = file.type.startsWith("video/");
        const hasVideoExt = videoExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

        if (isVideoMime || hasVideoExt) {
          // Проверяем, нет ли уже такого файла в списке (по имени и размеру)
          const isDuplicate = selectedFiles.some(f => f.name === file.name && f.size === file.size);
          if (!isDuplicate) {
            newFiles.push(file);
          }
        } else {
          toast.error(`Файл ${file.name} не похож на видео. Если это видео, попробуйте переименовать или сменить формат.`);
        }
      }

      if (newFiles.length > 0) {
        setSelectedFiles(prev => [...prev, ...newFiles]);
        toast.info(`Добавлено ${newFiles.length} файл(ов) к выбору`);
      }
    }

    // Очищаем значение инпута, чтобы можно было выбрать те же файлы снова
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeSelectedFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const clearSelectedFiles = () => {
    setSelectedFiles([]);
  };

  const fetchServerVideos = async () => {
    setLoadingServerVideos(true);
    try {
      const res = await apiClient.getUploadedVideosList();
      setServerVideos(res.videos || []);
      setSelectedServerVideos([]);
      setShowServerPicker(true);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Не удалось загрузить список видео");
    } finally {
      setLoadingServerVideos(false);
    }
  };

  const toggleServerVideoSelection = (v: VideoListItem) => {
    setSelectedServerVideos(prev =>
      prev.some(x => x.video_id === v.video_id)
        ? prev.filter(x => x.video_id !== v.video_id)
        : [...prev, v]
    );
  };

  const addServerVideosToList = async () => {
    if (selectedServerVideos.length === 0) {
      toast.error("Выберите одно или несколько видео");
      return;
    }
    if (!currentOwnerName.trim()) {
      toast.error("Введите имя сотрудника");
      return;
    }
    const ownerName = currentOwnerName.trim();
    if (!existingOwners.includes(ownerName)) {
      const newOwners = [...existingOwners, ownerName];
      setExistingOwners(newOwners);
      localStorage.setItem('trackai_owners', JSON.stringify(newOwners));
    }
    const timestamp = Date.now();
    const existingOwner = videos.find(v => v.ownerName === ownerName);
    const ownerColor = existingOwner ? existingOwner.color : userColors[Array.from(new Set(videos.map(v => v.ownerName))).length % userColors.length];
    const selectedServerVideoIds = new Set(selectedServerVideos.map((v) => v.video_id));
    selectedServerVideoIds.forEach((videoId) => {
      manualSuppressedVideoIdsRef.current.add(videoId);
      delete manualTrajectoryVersionsRef.current[videoId];
    });

    try {
      await Promise.all(
        selectedServerVideos.map((video) =>
          apiClient.registerExistingVideoTask(video.video_id, ownerName)
        )
      );
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Не удалось добавить видео в админку");
      return;
    }

    const newVideos: VideoWithOwner[] = selectedServerVideos.map((v, i) => ({
      id: `server-${v.video_id}-${timestamp}-${i}`,
      video_id: v.video_id,
      serverFilename: v.filename,
      ownerName,
      color: ownerColor,
      allowManualUpdates: false,
      isAnalyzing: false,
      uploadedAt: timestamp
    }));
    setVideos(prev => [...prev.filter((v) => !v.video_id || !selectedServerVideoIds.has(v.video_id)), ...newVideos]);
    onTrajectoryAnalyzed?.([], [], { cleared: true }, []);
    setShowServerPicker(false);
    setSelectedServerVideos([]);
    toast.success(`Добавлено ${newVideos.length} видео с сервера для ${ownerName}`);
  };

  const addVideoToList = () => {
    if (selectedFiles.length === 0) {
      toast.error("Сначала выберите один или несколько видео файлов");
      return;
    }

    if (!currentOwnerName.trim()) {
      toast.error("Введите имя или выберите сотрудника");
      return;
    }

    const ownerName = currentOwnerName.trim();

    // Сохраняем имя в список существующих
    if (!existingOwners.includes(ownerName)) {
      const newOwners = [...existingOwners, ownerName];
      setExistingOwners(newOwners);
      localStorage.setItem('trackai_owners', JSON.stringify(newOwners));
    }

    const timestamp = Date.now();

    // Ищем существующий цвет для этого сотрудника
    const existingOwner = videos.find(v => v.ownerName === ownerName);
    const ownerColor = existingOwner ? existingOwner.color : userColors[Array.from(new Set(videos.map(v => v.ownerName))).length % userColors.length];

    const newVideos: VideoWithOwner[] = selectedFiles.map((file, index) => ({
      id: `${timestamp}-${index}`,
      file: file,
      ownerName: ownerName,
      color: ownerColor,
      allowManualUpdates: false,
      isAnalyzing: false,
      uploadedAt: timestamp
    }));

    setVideos(prev => [...prev, ...newVideos]);
    onTrajectoryAnalyzed?.([], [], { cleared: true }, []);
    setCurrentOwnerName('');
    setSelectedFiles([]);

    // Очищаем input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    toast.success(`Добавлено ${newVideos.length} видео для сотрудника ${ownerName}`);
  };

  const removeVideo = (videoId: string) => {
    setVideos(prev => prev.filter(v => v.id !== videoId));
    toast.info("Видео удалено из списка");
  };

  const analyzeAllVideos = async () => {
    if (videos.length === 0) {
      console.log('❌ Ошибка: Нет видео для анализа');
      toast.error("Добавьте хотя бы одно видео для анализа");
      return;
    }
    if (!(externalReferencePoint || referencePoint) || !externalDirectionPoint) {
      toast.error("Сначала укажите на фиксированном плане точку старта и направление движения");
      return;
    }

    console.log(`🚀 Начало пакетного анализа ${videos.length} видео`);
    console.log(`📏 Коэффициент масштаба: ${scaleFactor}`);
    console.log(`🎥 Стабилизация: ${stabilizationEnabled ? 'включена' : 'отключена'}`);

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setLiveProgress(0);
    setLiveStatus("queued");
    setLiveStage("queued");
    setLiveElapsedSeconds(0);
    setLiveEtaSeconds(null);
    setLiveOwnerName("");
    setAnalysisStartedAtMs(Date.now());
    setCurrentStep("Подготовка production-пайплайна...");

    const applyLiveStatus = (
      status: Awaited<ReturnType<typeof apiClient.getProcessingStatus>>,
      ownerName: string,
      completedBefore: number,
      totalCount: number,
    ) => {
      const serverProgress = Math.max(0, Math.min(100, Number(status.progress) || 0));
      const unit = 100 / Math.max(totalCount, 1);
      const blended = Math.min(99, completedBefore * unit + (serverProgress / 100) * unit);
      setLiveProgress(serverProgress);
      setLiveStatus(status.status || "processing");
      setLiveStage(status.stage || "processing");
      setLiveOwnerName(ownerName);
      if (typeof status.elapsed_seconds === "number") {
        setLiveElapsedSeconds(status.elapsed_seconds);
      }
      if (typeof status.eta_seconds === "number") {
        setLiveEtaSeconds(status.eta_seconds);
      } else if (serverProgress >= 100) {
        setLiveEtaSeconds(0);
      }
      if (status.message) {
        setCurrentStep(`[${ownerName}] ${status.message}`);
      }
      setAnalysisProgress(Math.round(blended));
    };

    try {
      const startTime = Date.now();

      // Use the current `videos` state for analysis
      const videosToAnalyze = [...videos];
      const batchId = (
        typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ) ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const batchSize = videosToAnalyze.length;
      setBatchTotal(batchSize);

      // Загрузка и анализ по очереди (одно видео за раз)
      const finalizedVideos: VideoWithOwner[] = [];
      let completedBefore = 0;

      for (let videoIndex = 0; videoIndex < videosToAnalyze.length; videoIndex++) {
        const video = videosToAnalyze[videoIndex];
        setBatchIndex(videoIndex);
        setLiveOwnerName(video.ownerName);
        if (video.analysisResult) {
          completedBefore += 1;
          setAnalysisProgress(Math.round((completedBefore / batchSize) * 100));
          setLiveProgress(100);
          finalizedVideos.push(video);
          continue;
        }

        setVideos((prev) =>
          prev.map((v) => (v.id === video.id ? { ...v, isAnalyzing: true } : v))
        );
        const displayName = video.file?.name || video.serverFilename || "video";
        setCurrentStep(`Анализ видео ${video.ownerName} (${displayName})...`);

        let pollInterval: ReturnType<typeof setInterval> | null = null;

        try {
          let processingId: string | null = null;
          let uploadedVideoId: string;

          if (video.video_id) {
            uploadedVideoId = video.video_id;
            processingId = uploadedVideoId;
            // Set live view video ID for R³
            if (analysisMethod === 'r3' && !liveViewVideoId) {
              setLiveViewVideoId(uploadedVideoId);
            }
            apiClient
              .updateTaskContext(uploadedVideoId, {
                floorplan_id: 'kerama_marazzi_2025',
                floor_plan_data: externalFloorPlan || floorPlan,
                drawn_plan: drawnPlan || null,
                reference_point: externalReferencePoint || referencePoint,
                direction_point: externalDirectionPoint || null,
                employee_name: video.ownerName,
              })
              .catch(() => {});
          } else if (video.file) {
            setCurrentStep(`Загрузка ${video.file.name} на сервер...`);
            setLiveStage("upload");
            setLiveStatus("uploading");
            const uploadResult = await apiClient.uploadVideo(
              video.file,
              (progress) => {
                setVideos((prev) =>
                  prev.map((v) =>
                    v.id === video.id ? { ...v, uploadProgress: progress } : v
                  )
                );
                const uploadPct = Math.max(0, Math.min(100, progress));
                setLiveProgress(Math.max(1, Math.round(uploadPct * 0.08)));
                setAnalysisProgress(
                  Math.round(
                    completedBefore * (100 / batchSize)
                    + (uploadPct * 0.08 / 100) * (100 / batchSize)
                  )
                );
                setCurrentStep(`Загрузка ${video.file.name}: ${progress.toFixed(0)}%`);
              },
              video.ownerName,
              batchId,
              batchSize
            );
            uploadedVideoId = uploadResult.video_id;
            if (analysisMethod === 'r3' && !liveViewVideoId) {
              setLiveViewVideoId(uploadedVideoId);
            }
            processingId = uploadedVideoId;
            apiClient
              .updateTaskContext(uploadedVideoId, {
                floorplan_id: 'kerama_marazzi_2025',
                floor_plan_data: externalFloorPlan || floorPlan,
                drawn_plan: drawnPlan || null,
                reference_point: externalReferencePoint || referencePoint,
                direction_point: externalDirectionPoint || null,
                batch_id: batchId,
                batch_size: batchSize,
                employee_name: video.ownerName,
              })
              .catch(() => {});
          } else {
            throw new Error("Нет файла или video_id");
          }

          pollInterval = setInterval(async () => {
            if (!processingId) return;
            try {
              const status = await apiClient.getProcessingStatus(processingId);
              if (status) {
                applyLiveStatus(status, video.ownerName, completedBefore, batchSize);
              }
            } catch {
              /* ignore */
            }
          }, 1000);

          let result = await apiClient.analyzeVideoById(
            uploadedVideoId,
            scaleFactor,
            stabilizationEnabled,
            displayName,
            {
              detect_interval: detectInterval,
              turn_vote_threshold: turnVoteThreshold,
              use_ml_roi: useMlRoi,
            },
            {
              floorplan_id: 'kerama_marazzi_2025',
              floor_plan_data: externalFloorPlan || floorPlan,
              drawn_plan: drawnPlan || null,
              reference_point: externalReferencePoint || referencePoint,
              direction_point: externalDirectionPoint || null,
            },
            video.ownerName,
            analysisMethod,
            analysisMethod === 'r3' ? { frame_stride: 3, max_frames: 2000, ckpt: 'r3_long.safetensors', size: 392, mode: 'strided' } : undefined,
            true
          );

          if (result.status === "queued") {
            const maxAttempts = 1800;
            let attempts = 0;

            while (attempts < maxAttempts) {
              const status = await apiClient.getProcessingStatus(uploadedVideoId);

              if (status.status === "completed" && status.result) {
                result = { success: true, data: status.result, message: "Success" };
                applyLiveStatus(
                  { ...status, progress: 100, stage: "done", eta_seconds: 0 },
                  video.ownerName,
                  completedBefore,
                  batchSize,
                );
                break;
              } else if (status.status === "error") {
                throw new Error(status.message || "Ошибка при обработке на сервере");
              }

              applyLiveStatus(status, video.ownerName, completedBefore, batchSize);

              attempts++;
              await new Promise((resolve) => setTimeout(resolve, 2000));
            }

            if (attempts >= maxAttempts) {
              throw new Error(
                "Таймаут ожидания анализа (60 минут). Для больших AVI конвертация может занять 30+ мин."
              );
            }
          }

          if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
          }

          let analysisData = result.data;
          if (analysisMethod === "r3" && uploadedVideoId && analysisData) {
            try {
              setCurrentStep(`[${video.ownerName}] Перенос R³ траектории на план...`);
              setLiveStage("map");
              setLiveProgress((prev) => Math.max(prev, 96));
              // These two endpoints can be very slow on big AVI.
              // If they take too long, keep the UI unblocked and skip the
              // refined merge — the saved production result is still valid.
              const withTimeout = async <T,>(p: Promise<T>, ms: number): Promise<T> =>
                new Promise<T>((resolve, reject) => {
                  const t = window.setTimeout(() => reject(new Error("r3_refine_timeout")), ms);
                  p.then((v) => {
                    window.clearTimeout(t);
                    resolve(v);
                  }).catch((e) => {
                    window.clearTimeout(t);
                    reject(e);
                  });
                });

              const fetchRefined = Promise.all([
                apiClient.getR3PointCloudFiltered(uploadedVideoId, {
                  maxPoints: 100000,
                  minConf: 1.4,
                  samplingStrategy: "per_frame_uniform",
                  includeTrajectory: true,
                  includeCameras: false,
                }),
                apiClient.getR3Trajectory(uploadedVideoId, r3TrajectorySource),
              ]);

              const [filtered, selectedTrajectory] = await withTimeout(fetchRefined, 25000);
              const rawTrajectory3d = Array.isArray(filtered.raw_trajectory_3d)
                ? filtered.raw_trajectory_3d.filter((p) => Array.isArray(p) && p.length >= 3)
                : [];
              if (filtered.success && selectedTrajectory.success) {
                const filteredStats = filtered.stats || {};
                analysisData = mergeR3TrajectoryResponse(
                  analysisData,
                  selectedTrajectory,
                  rawTrajectory3d,
                  {
                    r3_filtered_trajectory: true,
                    r3_source_points: filteredStats.source_points,
                    r3_filtered_points: filteredStats.filtered_points,
                    r3_returned_points: filteredStats.returned_points,
                  },
                );
              }
            } catch (err) {
              console.warn("Failed to fetch filtered R3 trajectory for plan:", err);
            }
          }

          try {
            const existingManual = await apiClient.getManualTrajectory(uploadedVideoId);
            if (existingManual.exists && Array.isArray(existingManual.trajectory)) {
              manualTrajectoryVersionsRef.current[uploadedVideoId] =
                existingManual.updated_at || `${existingManual.trajectory.length}`;
            }
          } catch {
            /* manual trajectory may not exist */
          }
          manualSuppressedVideoIdsRef.current.delete(uploadedVideoId);

          const analyzedVideo: VideoWithOwner = {
            ...video,
            video_id: uploadedVideoId,
            analysisResult: analysisData,
            allowManualUpdates: true,
            isAnalyzing: false,
          };

          setVideos((prev) => prev.map((v) => (v.id === video.id ? analyzedVideo : v)));
          completedBefore += 1;
          setLiveProgress(100);
          setLiveStage("done");
          setLiveEtaSeconds(0);
          setAnalysisProgress(Math.round((completedBefore / batchSize) * 100));
          finalizedVideos.push(analyzedVideo);
        } catch (err: unknown) {
          if (pollInterval) {
            clearInterval(pollInterval);
          }
          console.error(`Error analyzing video ${video.id}:`, err);
          toast.error(
            `Ошибка при анализе ${displayName}: ${err instanceof Error ? err.message : "Неизвестная ошибка"}`
          );

          const errorVideo: VideoWithOwner = { ...video, isAnalyzing: false };
          setVideos((prev) => prev.map((v) => (v.id === video.id ? errorVideo : v)));
          completedBefore += 1;
          setLiveStatus("error");
          setLiveStage("error");
          setAnalysisProgress(Math.round((completedBefore / batchSize) * 100));
          finalizedVideos.push(errorVideo);
        }
      }

      setVideos(finalizedVideos);
      setAnalysisProgress(100);
      setLiveProgress(100);
      setLiveStage("done");
      setLiveEtaSeconds(0);
      setCurrentStep('Анализ всех видео завершен');

      const totalTime = (Date.now() - startTime) / 1000;
      console.log(`\n🎉 Пакетный анализ завершен!`);
      console.log(`📊 Статистика:`);
      console.log(`   • Обработано видео: ${videosToAnalyze.length}`);
      console.log(`   • Общее время: ${totalTime.toFixed(1)} сек`);
      console.log(`   • Среднее время на видео: ${(totalTime / finalizedVideos.length).toFixed(1)} сек`);

      const finalAnalyzedVideos = finalizedVideos.filter(v => v.analysisResult);

      if (finalAnalyzedVideos.length > 0 && onTrajectoryAnalyzed) {
        const trajectoriesData = finalAnalyzedVideos.map(trajectoryDataFromVideo);

        const totalPoints = trajectoriesData.reduce((sum, t) => sum + t.trajectory.length, 0);
        if (totalPoints === 0) {
          toast.warning("Траектория пуста: точек пути нет. Попробуйте другое видео или отключите стабилизацию.");
        }

        onTrajectoryAnalyzed(
          trajectoriesData[0].trajectory,
          trajectoriesData[0].turnPoints,
          (finalAnalyzedVideos[0].analysisResult.processing_stats || {}) as Record<string, unknown>,
          trajectoriesData
        );
        toast.success(`Анализ ${finalAnalyzedVideos.length} видео завершен успешно!`);
      } else {
        onTrajectoryAnalyzed?.([], [], { cleared: true, error: true }, []);
        toast.error("Анализ не вернул траекторию. Проверьте ошибку R³ в статусе/логах.");
      }

    } catch (error) {
      console.error("Analysis error:", error);
      toast.error("Ошибка при анализе видео");
    } finally {
      setIsAnalyzing(false);
      setTimeout(() => {
        setAnalysisProgress(0);
        setCurrentStep('');
        setLiveProgress(0);
        setLiveElapsedSeconds(null);
        setLiveEtaSeconds(null);
        setAnalysisStartedAtMs(null);
      }, 2000);
    }
  };

  const clearAnalysis = () => {
    setVideos([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Video className="h-5 w-5" />
          Анализ траектории движения
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Video Management */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label className="text-base font-medium">Загрузка видео для сотрудников</Label>
            <Badge variant="secondary">{videos.length} видео</Badge>
          </div>

          {/* Add new video */}
          <div className="border-2 border-dashed border-border rounded-lg p-5 bg-secondary/10 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="owner-name" className="text-sm font-semibold">Сотрудник</Label>
                <div className="flex flex-col gap-2">
                  <Input
                    id="owner-name"
                    type="text"
                    placeholder="Введите имя или выберите из списка..."
                    value={currentOwnerName}
                    onChange={(e) => setCurrentOwnerName(e.target.value)}
                    className="mt-1"
                    onKeyPress={(e) => e.key === 'Enter' && addVideoToList()}
                  />
                  {existingOwners.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {existingOwners.slice(0, 5).map(name => (
                        <Badge
                          key={name}
                          variant="outline"
                          className="cursor-pointer hover:bg-primary/20 transition-colors"
                          onClick={() => setCurrentOwnerName(name)}
                        >
                          {name}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="video-file" className="text-sm font-semibold">Видео файлы (можно несколько)</Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    ref={fileInputRef}
                    id="video-file"
                    type="file"
                    accept="video/*"
                    multiple
                    onChange={handleFileSelect}
                    className="flex-1 bg-background"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="shrink-0 gap-1.5"
                    onClick={fetchServerVideos}
                    disabled={loadingServerVideos}
                  >
                    {loadingServerVideos ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FolderOpen className="h-3.5 w-3.5" />}
                    Из загруженных
                  </Button>
                </div>
                {selectedFiles.length > 0 && (
                  <div className="mt-2 p-3 bg-primary/5 rounded-lg border border-primary/20 space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-xs font-bold text-primary">Готовы к добавлению ({selectedFiles.length}):</p>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-[10px] text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={clearSelectedFiles}
                      >
                        Очистить всё
                      </Button>
                    </div>
                    <ul className="text-[10px] space-y-1.5 max-h-40 overflow-y-auto pr-1">
                      {selectedFiles.map((f, i) => (
                        <li key={`${f.name}-${i}`} className="flex items-center justify-between group bg-background/50 p-1.5 rounded">
                          <div className="flex items-center gap-2 overflow-hidden mr-2">
                            <Video className="h-3 w-3 text-primary/50 flex-shrink-0" />
                            <span className="truncate">{f.name}</span>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            <span className="text-muted-foreground">{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                            <button
                              onClick={() => removeSelectedFile(i)}
                              className="text-muted-foreground hover:text-destructive transition-colors"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={addVideoToList}
                disabled={!currentOwnerName.trim() || selectedFiles.length === 0}
                className="flex-1 bg-primary/90 hover:bg-primary"
              >
                <Plus className="h-4 w-4 mr-2" />
                Добавить видео для этого сотрудника
              </Button>
            </div>

            <Dialog open={showServerPicker} onOpenChange={setShowServerPicker}>
              <DialogContent className="max-w-md max-h-[80vh] flex flex-col">
                <DialogHeader>
                  <DialogTitle>Выбрать из загруженных видео</DialogTitle>
                  <DialogDescription>
                    Выберите ранее загруженное видео для текущего сотрудника.
                  </DialogDescription>
                </DialogHeader>
                <p className="text-sm text-muted-foreground">
                  Сотрудник: <strong>{currentOwnerName || '— введите выше'}</strong>
                </p>
                {serverVideos.length === 0 ? (
                  <p className="text-sm text-muted-foreground py-4">Сервер не вернул видео. Загрузите видео через форму выше.</p>
                ) : (
                  <ul className="space-y-2 overflow-y-auto max-h-60 flex-1 pr-2">
                    {serverVideos.map((v, idx) => {
                      const isSelected = selectedServerVideos.some(x => x.video_id === v.video_id);
                      return (
                        <li
                          key={`${v.video_id}-${v.filename || idx}`}
                          onClick={() => toggleServerVideoSelection(v)}
                          className={`flex items-center gap-3 p-2 rounded-lg border cursor-pointer transition-colors ${
                            isSelected ? 'bg-primary/10 border-primary/30' : 'hover:bg-secondary/50'
                          }`}
                        >
                          <div className={`w-4 h-4 rounded border flex items-center justify-center ${isSelected ? 'bg-primary' : ''}`}>
                            {isSelected && <span className="text-white text-xs">✓</span>}
                          </div>
                          <div className="flex-1 min-w-0">
                            <span className="text-sm font-medium truncate block">{v.filename}</span>
                            <span className="text-muted-foreground text-xs">{(v.file_size / 1024 / 1024).toFixed(1)} MB</span>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
                <DialogFooter>
                  <Button variant="outline" onClick={() => setShowServerPicker(false)}>Отмена</Button>
                  <Button
                    onClick={addServerVideosToList}
                    disabled={!currentOwnerName.trim() || selectedServerVideos.length === 0}
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Добавить ({selectedServerVideos.length})
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>

          {/* Video list grouped by owner */}
          {videos.length > 0 && (
            <div className="space-y-4 mt-6">
              <Label className="text-sm font-bold">Очередь обработки по сотрудникам:</Label>
              {Array.from(new Set(videos.map(v => v.ownerName))).map(ownerName => {
                const ownerVideos = videos.filter(v => v.ownerName === ownerName);
                const ownerColor = ownerVideos[0]?.color;

                return (
                  <div key={ownerName} className="p-4 rounded-xl border border-border/50 bg-secondary/20 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded-full border border-white" style={{ backgroundColor: ownerColor }} />
                        <span className="font-bold text-lg">{ownerName}</span>
                        <Badge variant="secondary">{ownerVideos.length} видео</Badge>
                      </div>
                    </div>

                    <div className="space-y-2">
                      {ownerVideos.map((video) => (
                        <div
                          key={video.id}
                          className="flex items-center justify-between p-2 pl-3 bg-background/50 rounded-lg border border-border/30 group"
                        >
                          <div className="flex items-center gap-3 overflow-hidden flex-1">
                            <Video className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                            <div className="flex flex-col overflow-hidden min-w-0">
                              <span className="text-sm font-medium truncate">{video.file?.name || video.serverFilename || 'video'}</span>
                              <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                                <span>{video.file ? `${(video.file.size / 1024 / 1024).toFixed(1)} MB` : 'на сервере'}</span>
                                {video.uploadedAt && (
                                  <>
                                    <span>•</span>
                                    <span>{new Date(video.uploadedAt).toLocaleString('ru-RU', {
                                      day: '2-digit',
                                      month: '2-digit',
                                      year: 'numeric',
                                      hour: '2-digit',
                                      minute: '2-digit'
                                    })}</span>
                                  </>
                                )}
                              </div>
                            </div>
                            {video.isAnalyzing && (
                              <Badge variant="secondary" className="gap-1 animate-pulse flex-shrink-0">
                                <Loader2 className="h-3 w-3 animate-spin" />
                                {video.uploadProgress !== undefined && video.uploadProgress < 100
                                  ? `Загрузка ${video.uploadProgress.toFixed(0)}%`
                                  : 'Обработка...'}
                              </Badge>
                            )}
                            {video.analysisResult && (
                              <Badge variant="default" className="gap-1 bg-green-500/20 text-green-500 border-green-500/20 flex-shrink-0">
                                ✓ Готово
                              </Badge>
                            )}
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => removeVideo(video.id)}
                            disabled={isAnalyzing}
                            className="h-8 w-8 text-destructive opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {SHOW_ADVANCED_ANALYSIS_SETTINGS && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="scale-factor">Коэффициент масштаба</Label>
            <span className="text-sm font-medium text-primary bg-primary/10 px-2 py-1 rounded">
              {scaleFactor.toFixed(1)}
            </span>
          </div>

          <div className="space-y-3">
            <Slider
              id="scale-factor"
              min={1}
              max={50}
              step={0.1}
              value={[scaleFactor]}
              onValueChange={(value) => setScaleFactor(value[0])}
              className="w-full"
            />

            <div className="flex justify-between text-xs text-muted-foreground">
              <span>1.0</span>
              <span>25.0</span>
              <span>50.0</span>
            </div>
          </div>

          <div className="flex gap-2">
            <Input
              type="number"
              step="0.1"
              min="1"
              max="50"
              value={scaleFactor}
              onChange={(e) => setScaleFactor(parseFloat(e.target.value) || 1)}
              className="w-24"
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setScaleFactor(12.306)}
            >
              Сбросить
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label htmlFor="detect-interval" className="text-xs text-muted-foreground">Detect interval (кадры)</Label>
              <Input
                id="detect-interval"
                type="number"
                min="1"
                max="30"
                value={detectInterval}
                onChange={(e) => setDetectInterval(Math.max(1, Math.min(30, Number(e.target.value) || 5)))}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="turn-vote-threshold" className="text-xs text-muted-foreground">Turn vote threshold</Label>
              <Input
                id="turn-vote-threshold"
                type="number"
                min="1"
                max="5"
                value={turnVoteThreshold}
                onChange={(e) => setTurnVoteThreshold(Math.max(1, Math.min(5, Number(e.target.value) || 3)))}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="use-ml-roi" className="text-xs text-muted-foreground">ML ROI (YOLO+Tracker)</Label>
              <div className="flex items-center gap-2">
                <Switch
                  id="use-ml-roi"
                  checked={useMlRoi}
                  onCheckedChange={(v) => setUseMlRoi(!!v)}
                />
                <span className="text-sm">{useMlRoi ? 'Вкл' : 'Выкл'}</span>
              </div>
            </div>
          </div>

          <p className="text-sm text-muted-foreground">
            Коэффициент для перевода пикселей в метры. Влияет на точность расчета расстояний.
          </p>
        </div>
        )}

        {/* Live processing dashboard */}
        {isAnalyzing && (
          <ProcessingDashboard
            ownerName={liveOwnerName || undefined}
            method={analysisMethod}
            progress={Math.max(analysisProgress, liveProgress)}
            message={currentStep}
            status={liveStatus}
            stage={liveStage}
            elapsedSeconds={liveElapsedSeconds}
            etaSeconds={liveEtaSeconds}
            batchIndex={batchIndex}
            batchTotal={batchTotal}
            startedAtMs={analysisStartedAtMs}
          />
        )}

        {SHOW_ADVANCED_ANALYSIS_SETTINGS && (
        <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg border">
          <div className="flex flex-col">
            <Label htmlFor="stabilization-toggle" className="text-sm font-medium">
              Программная стабилизация видео
            </Label>
            <p className="text-xs text-muted-foreground">
              Удаляет тряску камеры для более точного анализа
            </p>
          </div>
          <Switch
            id="stabilization-toggle"
            checked={stabilizationEnabled}
            onCheckedChange={handleStabilizationToggle}
          />
        </div>
        )}

        {/* Single production pipeline */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Production-анализ</Label>
          {analysisMethod === 'r3' && (
            <div className="space-y-3 rounded-lg border border-primary/20 bg-primary/5 p-3">
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                <span>
                  Единый production-конвейер: R³ → robust graph → scale-aware → LingBot shadow/fusion
                  → ограничения фиксированного плана Kerama Marazzi.
                </span>
                <Badge variant="default">Production pipeline</Badge>
                {activeMapObservationSource && mapObservationSourceLabel(activeMapObservationSource) && (
                  <Badge variant={activeMapObservationSource === 'r3' ? 'outline' : 'default'}>
                    {mapObservationSourceLabel(activeMapObservationSource)}
                  </Badge>
                )}
                {activeLingbotFusion.accepted === true && activeMapObservationSource !== 'r3_lingbot_fusion' && (
                  <Badge variant="outline">
                    LingBot согласован с R³ (shadow)
                    {activeFloorplanConstraint.accepted === false ? ', но карта не приняла маршрут' : ''}
                  </Badge>
                )}
                {activeLingbotFusion.accepted === false && activeLingbotFusion.reason && (
                  <Badge variant="outline">
                    Fusion не принят: {lingbotFusionReasonLabel(activeLingbotFusion.reason)}
                  </Badge>
                )}
                {typeof activeFloorplanConstraint.accepted === 'boolean' && (
                  <Badge variant={activeFloorplanConstraint.accepted ? 'default' : 'outline'}>
                    {activeFloorplanConstraint.accepted
                      ? `План применён · ${Math.round(finiteNum(activeFloorplanConstraint.confidence) * 100)}%`
                      : `Нет допустимого маршрута: ${activeFloorplanRejections[0] || floorplanDiagnosticLabel(activeFloorplanConstraint.reason)}`}
                  </Badge>
                )}
                {activeFloorplanConstraint.accepted === true && activeFloorplanWarnings.length > 0 && (
                  <Badge variant="outline">
                    {activeFloorplanWarnings.join(' · ')}
                  </Badge>
                )}
              </div>
            </div>
          )}
          {analysisMethod === 'lingbot' && (
            <p className="text-xs text-muted-foreground">
              MVP LingBot-Map worker: потоковая 3D-реконструкция на RTX 3090 через отдельный FastAPI-сервис
            </p>
          )}
        </div>

        {/* Live view button for R³ monitoring */}
        {analysisMethod === 'r3' && isAnalyzing && liveViewVideoId && !showLiveView && (
          <div className="space-y-2">
            <Button
              onClick={() => setShowLiveView(true)}
              variant="outline"
              className="w-full border-primary/30 hover:bg-primary/10 gap-2"
            >
              <Eye className="h-4 w-4 text-primary" />
              <span>Смотреть реконструкцию в реальном времени</span>
            </Button>
          </div>
        )}

        {/* Real-time R³ visualization */}
        {showLiveView && liveViewVideoId && (
          <RealTimeR3Visualization
            videoId={liveViewVideoId}
            onComplete={() => {
              // Keep showing the visualization when complete
            }}
            onClose={() => {
              setShowLiveView(false);
            }}
          />
        )}

        {/* Analyze buttons */}
        <div className="space-y-3">
          <Button
            onClick={analyzeAllVideos}
            disabled={
              videos.length === 0
              || isAnalyzing
              || !(externalReferencePoint || referencePoint)
              || !externalDirectionPoint
            }
            className="w-full"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Анализ в процессе...
              </>
            ) : (
              <>
                <Activity className="h-4 w-4 mr-2" />
                Запустить production-анализ
                {videos.length > 0 ? ` (${videos.length} видео)` : ''}
              </>
            )}
          </Button>
        </div>

        {SHOW_ADVANCED_ANALYSIS_SETTINGS && (
        <div className="space-y-2 mt-3">
          <p className="text-xs text-muted-foreground text-center">
            🎥 Автоматическая стабилизация + SLAM анализ траектории
          </p>
        </div>
        )}

        {/* Analysis results */}
        {analysisResult && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Результаты анализа</h3>

            {/* Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-secondary rounded-lg">
                <MapPin className="h-5 w-5 mx-auto mb-1 text-primary" />
                <div className="text-2xl font-bold">
                  {analysisResult?.data?.trajectory_points || 0}
                </div>
                <div className="text-xs text-muted-foreground">Точек траектории</div>
              </div>

              <div className="text-center p-3 bg-secondary rounded-lg">
                <Navigation className="h-5 w-5 mx-auto mb-1 text-primary" />
                <div className="text-2xl font-bold">
                  {analysisResult?.data?.processing_stats?.estimated_distance?.toFixed(1) || "0.0"}
                </div>
                <div className="text-xs text-muted-foreground">Расстояние (м)</div>
              </div>

              <div className="text-center p-3 bg-secondary rounded-lg">
                <Activity className="h-5 w-5 mx-auto mb-1 text-primary" />
                <div className="text-2xl font-bold">
                  {analysisResult?.data?.turn_points?.length || 0}
                </div>
                <div className="text-xs text-muted-foreground">Поворотов</div>
              </div>

              <div className="text-center p-3 bg-secondary rounded-lg">
                <Clock className="h-5 w-5 mx-auto mb-1 text-primary" />
                <div className="text-2xl font-bold">
                  {analysisResult?.data?.total_processing_time?.toFixed(1) || "0.0"}
                </div>
                <div className="text-xs text-muted-foreground">Время анализа (с)</div>
              </div>
            </div>

            {/* Video info */}
            <div className="p-4 bg-secondary rounded-lg">
              <h4 className="font-medium mb-2">Информация о видео</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Разрешение:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.video_info?.width || 0}x{analysisResult?.data?.video_info?.height || 0}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">FPS:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.video_info?.fps?.toFixed(1) || "0.0"}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Длительность:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.video_info?.duration ? formatTime(analysisResult.data.video_info.duration) : "0:00"}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Кадров:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.video_info?.frame_count || 0}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Matches/кадр:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.processing_stats?.avg_matches_per_frame?.toFixed?.(1) || "0.0"}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Gating fail rate:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.processing_stats?.gating_failure_rate !== undefined
                      ? `${(analysisResult.data.processing_stats.gating_failure_rate * 100).toFixed(1)}%`
                      : "0.0%"}
                  </span>
                </div>
              </div>
            </div>

            {/* Turn points */}
            {analysisResult?.data?.turn_points && analysisResult.data.turn_points.length > 0 && (
              <div className="p-4 bg-secondary rounded-lg">
                <h4 className="font-medium mb-2">Обнаруженные повороты</h4>
                <div className="space-y-2">
                  {analysisResult.data.turn_points.map((turn: Record<string, unknown>, index: number) => (
                    <div key={index} className="flex items-center justify-between text-sm">
                      <span>Поворот {index + 1}</span>
                      <div className="flex gap-4">
                        <span>{Number(turn.angle_degrees).toFixed(1) || "0.0"}°</span>
                        <span className="capitalize">{turn.turn_type === "left" ? "Влево" : "Вправо"}</span>
                        <span>
                          (
                          {Array.isArray(turn.position)
                            ? `${finiteNum(turn.position[0]).toFixed(1)}, ${finiteNum(turn.position[1]).toFixed(1)}`
                            : "0.0, 0.0"}
                          )
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TrajectoryAnalysis;
