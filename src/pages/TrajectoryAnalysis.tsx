import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Video, Library, Upload, Map, X, ZoomIn, ZoomOut, RotateCcw, Navigation } from "lucide-react";
import Navbar from "@/components/Navbar";
import TrajectoryAnalysis from "@/components/TrajectoryAnalysis";
import TrajectoryMap from "@/components/TrajectoryMap";
import VideoLibrary from "@/components/VideoLibrary";
import { apiClient, VideoListItem, Plan } from "@/lib/api";
import { toast } from "sonner";
import PlanEditor from "@/components/PlanEditor";
import PlanLibrary from "@/components/PlanLibrary";
import { PenTool, Library as LibraryIcon } from "lucide-react";

const TrajectoryAnalysisPage = () => {
  const [trajectory, setTrajectory] = useState([]);
  const [turnPoints, setTurnPoints] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [floorPlan, setFloorPlan] = useState(null);
  const [floorPlanFile, setFloorPlanFile] = useState(null);
  const [drawnPlan, setDrawnPlan] = useState<any[] | null>(null);
  const [activePlanId, setActivePlanId] = useState<number | null>(null);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [planMode, setPlanMode] = useState<'upload' | 'draw' | 'library'>('upload');
  const [referencePoint, setReferencePoint] = useState(null); // {x, y} - точка отсчета на плане
  const [directionPoint, setDirectionPoint] = useState(null); // {x, y} - направление движения от точки отсчета
  const [setDirectionMode, setSetDirectionMode] = useState(false);
  const [planScale, setPlanScale] = useState(1); // Масштаб превью плана
  const floorPlanInputRef = useRef(null);
  const planImgRef = useRef<HTMLImageElement>(null);
  const [planImgLayout, setPlanImgLayout] = useState<{ dispX: number; dispY: number; dispW: number; dispH: number; cw: number; ch: number } | null>(null);

  // Загрузка плана и точки отсчета из localStorage при инициализации
  useEffect(() => {
    const savedFloorPlan = localStorage.getItem('floorPlan');
    if (savedFloorPlan) {
      try {
        const planData = JSON.parse(savedFloorPlan);
        if (planData.data && planData.type) {
          if (planData.type === "application/pdf") {
            toast.info("PDF план загружен. Для повторного отображения загрузите файл заново.");
            localStorage.removeItem('floorPlan');
          } else if ((planData.data?.length || 0) > 1_500_000) {
            // Слишком большой план (старый DWG SVG) — вызывает Out of memory и Not allowed to load
            localStorage.removeItem('floorPlan');
          } else if (planData.type === "image/svg+xml" && (planData.data?.length || 0) > 500_000) {
            // Старый формат DWG→SVG — удаляем, теперь используем PNG
            localStorage.removeItem('floorPlan');
          } else {
            setFloorPlan(planData.data);
            setFloorPlanFile({ name: planData.name, type: planData.type });
          }
        } else {
          const str = typeof planData === 'string' ? planData : planData?.data;
          if (str && str.length > 1_500_000) {
            localStorage.removeItem('floorPlan');
          } else {
            setFloorPlan(typeof planData === 'string' ? planData : planData?.data || savedFloorPlan);
            setFloorPlanFile(planData?.name ? { name: planData.name, type: planData.type || 'image/png' } : null);
          }
        }
      } catch (e) {
        if (savedFloorPlan.length > 1_500_000) {
          localStorage.removeItem('floorPlan');
        } else {
          setFloorPlan(savedFloorPlan);
        }
      }
    }

    // Загрузка точки отсчета
    const savedReferencePoint = localStorage.getItem('referencePoint');
    if (savedReferencePoint) {
      try {
        const pointData = JSON.parse(savedReferencePoint);
        setReferencePoint(pointData);
      } catch (e) {
        console.warn('Failed to load reference point from localStorage:', e);
      }
    }

    // Загрузка направления движения
    const savedDirectionPoint = localStorage.getItem('directionPoint');
    if (savedDirectionPoint) {
      try {
        const pointData = JSON.parse(savedDirectionPoint);
        setDirectionPoint(pointData);
      } catch (e) {
        console.warn('Failed to load direction point from localStorage:', e);
      }
    }
  }, []);

  // Измерение layout изображения плана (object-contain) для корректного отображения маркеров
  const measurePlanImgLayout = () => {
    const img = planImgRef.current;
    if (!img || !floorPlan || img.naturalWidth === 0) return;
    const rect = img.getBoundingClientRect();
    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    const scale = Math.min(rect.width / nw, rect.height / nh);
    const dispW = nw * scale;
    const dispH = nh * scale;
    const dispX = (rect.width - dispW) / 2;
    const dispY = (rect.height - dispH) / 2;
    setPlanImgLayout({ dispX, dispY, dispW, dispH, cw: rect.width, ch: rect.height });
  };

  useEffect(() => {
    if (!floorPlan || floorPlanFile?.type === "application/pdf") return;
    measurePlanImgLayout();
    const img = planImgRef.current;
    if (!img) return;
    const ro = new ResizeObserver(measurePlanImgLayout);
    ro.observe(img);
    return () => ro.disconnect();
  }, [floorPlan, floorPlanFile?.type]);

  const handleFloorPlanUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const isDwg = file.name.toLowerCase().endsWith('.dwg');
    const isPdf = file.type === "application/pdf";
    const isImage = file.type.startsWith("image/");

    if (!isImage && !isPdf && !isDwg) {
      toast.error("Загрузите изображение, PDF или DWG файл плана");
      return;
    }

    // DWG: конвертируем на сервере в PNG с прогрессом
    if (isDwg) {
      const fileSizeMB = (file.size / 1024 / 1024).toFixed(1);
      const sizeMB = file.size / 1024 / 1024;
      if (sizeMB > 50) {
        toast.warning(
          `DWG ${fileSizeMB} MB — загрузка может занять 20–30 мин. Рекомендуем: в AutoCAD File → Export → PNG, затем загрузить PNG.`,
          { duration: 8000 }
        );
      }
      const toastId = 'dwg-convert';
      toast.loading(`Конвертация DWG (${fileSizeMB} MB): загрузка 0%... (до 30 мин для больших файлов)`, { id: toastId });
      apiClient
        .convertDwgToImage(file, (progress, message) => {
          toast.loading(`Конвертация DWG (${fileSizeMB} MB): ${message} ${progress}%`, { id: toastId });
        })
        .then(({ png, filename }) => {
          if (!png || png.length === 0) throw new Error("Сервер вернул пустое изображение");
          const dataUrl = `data:image/png;base64,${png}`;
          setFloorPlan(dataUrl);
          setFloorPlanFile({ name: filename, type: 'image/png' });
          setReferencePoint(null);
          setDirectionPoint(null);
          localStorage.removeItem('referencePoint');
          localStorage.removeItem('directionPoint');
          try {
            localStorage.setItem('floorPlan', JSON.stringify({ data: dataUrl, type: 'image/png', name: filename }));
          } catch (e) {
            if (e?.name === 'QuotaExceededError') console.warn('localStorage full, план не сохранён');
          }
          toast.success(`План "${filename}" загружен`, { id: toastId });
        })
        .catch((err) => {
          toast.error(err?.message || "Ошибка конвертации DWG", { id: toastId });
        });
      e.target.value = '';
      return;
    }

    // PDF: конвертируем на сервере в PNG
    if (isPdf) {
      const toastId = 'pdf-convert';
      toast.loading('Конвертация PDF: загрузка 0%...', { id: toastId });
      apiClient
        .convertPdfToImage(file, (progress) => {
          toast.loading(`Конвертация PDF: загрузка ${progress}%...`, { id: toastId });
        })
        .then(({ png, filename }) => {
          if (!png || png.length === 0) throw new Error("Сервер вернул пустое изображение");
          const dataUrl = `data:image/png;base64,${png}`;
          setFloorPlan(dataUrl);
          setFloorPlanFile({ name: filename, type: 'image/png' });
          setReferencePoint(null);
          setDirectionPoint(null);
          localStorage.removeItem('referencePoint');
          localStorage.removeItem('directionPoint');
          try {
            localStorage.setItem('floorPlan', JSON.stringify({ data: dataUrl, type: 'image/png', name: filename }));
          } catch (err) {
            if (err?.name === 'QuotaExceededError') console.warn('localStorage full');
          }
          toast.success(`План "${filename}" загружен`, { id: toastId });
        })
        .catch((err) => {
          toast.error(err?.message || 'Ошибка конвертации PDF', { id: toastId });
        });
      e.target.value = '';
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      let result = event.target?.result;

      setFloorPlan(result);
      setFloorPlanFile(file);
      setReferencePoint(null);
      setDirectionPoint(null);
      localStorage.removeItem('referencePoint');
      localStorage.removeItem('directionPoint');
      const planData = { data: result, type: file.type, name: file.name };
      try {
        localStorage.setItem('floorPlan', JSON.stringify(planData));
      } catch (e) {
        console.warn('Failed to save floor plan to localStorage:', e);
      }
      toast.success(`План "${file.name}" загружен`);
    };

    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const handleFloorPlanClick = (event) => {
    if (!floorPlan && !drawnPlan) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const clientX = event.clientX - rect.left;
    const clientY = event.clientY - rect.top;

    let x: number, y: number;

    const target = event.currentTarget;
    if (target instanceof HTMLImageElement && target.naturalWidth > 0) {
      // Изображение: object-contain — конвертируем клик в координаты относительно контента изображения
      const imgW = target.naturalWidth;
      const imgH = target.naturalHeight;
      const scale = Math.min(rect.width / imgW, rect.height / imgH);
      const dispW = imgW * scale;
      const dispH = imgH * scale;
      const dispX = (rect.width - dispW) / 2;
      const dispY = (rect.height - dispH) / 2;
      x = Math.max(0, Math.min(100, ((clientX - dispX) / dispW) * 100));
      y = Math.max(0, Math.min(100, ((clientY - dispY) / dispH) * 100));
    } else if (target instanceof SVGSVGElement) {
      // SVG (drawn plan): viewBox 800x600, meet — конвертируем клик в координаты viewBox
      const vbW = 800;
      const vbH = 600;
      const scale = Math.min(rect.width / vbW, rect.height / vbH);
      const dispW = vbW * scale;
      const dispH = vbH * scale;
      const dispX = (rect.width - dispW) / 2;
      const dispY = (rect.height - dispH) / 2;
      x = Math.max(0, Math.min(100, ((clientX - dispX) / dispW) * 100));
      y = Math.max(0, Math.min(100, ((clientY - dispY) / dispH) * 100));
    } else {
      // PDF iframe и прочее: используем координаты контейнера
      x = (clientX / rect.width) * 100;
      y = (clientY / rect.height) * 100;
    }

    if (setDirectionMode && referencePoint) {
      setDirectionPoint({ x, y });
      localStorage.setItem('directionPoint', JSON.stringify({ x, y }));
      setSetDirectionMode(false);
      toast.success(`Направление движения установлено: (${x.toFixed(1)}%, ${y.toFixed(1)}%)`);
    } else {
      setReferencePoint({ x, y });
      setDirectionPoint(null);
      localStorage.setItem('referencePoint', JSON.stringify({ x, y }));
      localStorage.removeItem('directionPoint');
      toast.success(`Точка отсчета установлена: (${x.toFixed(1)}%, ${y.toFixed(1)}%)`);
    }
  };

  const handleReferencePointRemove = () => {
    setReferencePoint(null);
    setDirectionPoint(null);
    setSetDirectionMode(false);
    localStorage.removeItem('referencePoint');
    localStorage.removeItem('directionPoint');
    toast.info("Точка отсчета и направление удалены");
  };

  const handleDirectionPointRemove = () => {
    setDirectionPoint(null);
    setSetDirectionMode(false);
    localStorage.removeItem('directionPoint');
    toast.info("Направление движения удалено");
  };

  const handleDirectionPointSet = (point: { x: number; y: number }) => {
    setDirectionPoint(point);
    setSetDirectionMode(false);
    localStorage.setItem('directionPoint', JSON.stringify(point));
    toast.success(`Направление установлено: (${point.x.toFixed(1)}%, ${point.y.toFixed(1)}%)`);
  };

  const handleFloorPlanRemove = () => {
    // Очищаем blob URL для PDF файлов
    if (floorPlan && floorPlanFile?.type === "application/pdf" && floorPlan.startsWith('blob:')) {
      URL.revokeObjectURL(floorPlan);
    }

    setFloorPlan(null);
    setFloorPlanFile(null);
    setReferencePoint(null);
    setDirectionPoint(null);
    setSetDirectionMode(false);
    localStorage.removeItem('floorPlan');
    localStorage.removeItem('referencePoint');
    localStorage.removeItem('directionPoint');
    toast.info("План помещения и точка отсчета удалены");
  };

  const handlePlanSelect = (plan: Plan) => {
    setDrawnPlan(plan.data);
    setActivePlanId(plan.id || null);
    setFloorPlan(null); // Сбрасываем загруженное изображение
    setFloorPlanFile({ name: plan.name, type: 'drawn' });
    setPlanMode('upload'); // Переключаемся на просмотр

    // Сбрасываем точку отсчета и направление при выборе нового плана
    setReferencePoint(null);
    setDirectionPoint(null);
    localStorage.removeItem('referencePoint');
    localStorage.removeItem('directionPoint');

    toast.success(`План "${plan.name}" выбран`);
  };

  const handlePlanSaved = (plan: Plan) => {
    handlePlanSelect(plan);
    setIsEditorOpen(false);
  };

  const handleTrajectoryAnalyzed = (newTrajectory, newTurnPoints, newStats, trajectories) => {
    console.log('📥 handleTrajectoryAnalyzed called with:', {
      newTrajectory: newTrajectory ? newTrajectory.length : 'null/undefined',
      newTurnPoints: newTurnPoints ? newTurnPoints.length : 'null/undefined',
      newStats: newStats ? 'present' : 'null',
      trajectories: trajectories ? trajectories.length : 'null/undefined'
    });

    if (trajectories && trajectories.length > 0) {
      // Используем множественные траектории
      console.log('🎯 Setting trajectories (multiple):', trajectories.length, 'items');
      setTrajectory(trajectories);
    } else {
      // Обратная совместимость с одиночной траекторией
      console.log('🎯 Setting single trajectory:', newTrajectory?.length || 0, 'points');
      setTrajectory(newTrajectory);
      setTurnPoints(newTurnPoints);
    }
    setStats(newStats);
  };

  const handleVideoSelected = (video) => {
    setSelectedVideo(video);
    setVideoUrl(apiClient.getVideoDownloadUrl(video.video_id));
  };

  const handleAnalysisLoaded = (newTrajectory, newTurnPoints, newStats) => {
    // Конвертируем [[x,y,z], ...] в [{x,y,z}, ...] для TrajectoryMap
    const convertTrajectory = (traj) => {
      if (!traj || !Array.isArray(traj)) return [];
      if (traj.length === 0) return [];
      if (Array.isArray(traj[0])) {
        return traj.map((p) => ({
          x: Number(p[0]) ?? 0,
          y: Number(p[1]) ?? 0,
          z: Number(p[2]) ?? 0
        }));
      }
      if (typeof traj[0] === 'object' && traj[0] && 'x' in traj[0]) return traj;
      return [];
    };
    const converted = convertTrajectory(newTrajectory);
    const trajectoriesData = [{
      trajectory: converted,
      turnPoints: newTurnPoints || [],
      ownerName: 'Библиотека',
      color: '#3b82f6'
    }];
    setTrajectory(trajectoriesData);
    setTurnPoints(newTurnPoints || []);
    setStats(newStats);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <Navbar />
      <div className="container mx-auto px-4 py-8 pt-24">
        <div className="mb-8">
          <h1 className="text-4xl font-bold tracking-tight mb-2">
            Анализ траектории движения
          </h1>
          <p className="text-muted-foreground text-lg">
            Загрузите план помещения и видео для визуализации траектории движения на реальной карте
          </p>
        </div>

        {/* Floor Plan Upload */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Map className="h-5 w-5" />
                План помещения
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant={planMode === 'upload' ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={() => setPlanMode('upload')}
                  className="gap-2"
                >
                  <Upload className="h-4 w-4" /> Загрузить
                </Button>
                <Button
                  variant={planMode === 'draw' ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={() => setIsEditorOpen(true)}
                  className="gap-2"
                >
                  <PenTool className="h-4 w-4" /> Нарисовать
                </Button>
                <Button
                  variant={planMode === 'library' ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={() => setPlanMode('library')}
                  className="gap-2"
                >
                  <LibraryIcon className="h-4 w-4" /> Библиотека
                </Button>
                <Button
                  variant={setDirectionMode ? "default" : "outline"}
                  size="sm"
                  className="gap-2 text-emerald-600 hover:text-emerald-700 border-emerald-500/50"
                  onClick={() => referencePoint && setSetDirectionMode(v => !v)}
                  disabled={!referencePoint}
                  title={referencePoint ? "Кликните на план ниже, чтобы указать направление движения" : "Сначала установите точку отсчёта"}
                >
                  <Navigation className="h-4 w-4" />
                  Указать направление
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {planMode === 'library' && (
              <PlanLibrary onSelect={handlePlanSelect} selectedId={activePlanId || undefined} />
            )}

            {planMode === 'upload' && !floorPlan && (
              <div className="border-2 border-dashed border-border rounded-xl p-8 text-center bg-muted/10">
                <input
                  ref={floorPlanInputRef}
                  type="file"
                  accept="image/*,.pdf,.dwg,application/dwg,application/acad,application/vnd.autodesk.autocad.drawing"
                  className="hidden"
                  onChange={handleFloorPlanUpload}
                />

                <div className="flex flex-col items-center gap-3">
                  <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <Map className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="font-medium mb-1">Загрузите план помещения или выберите из библиотеки</p>
                    <p className="text-sm text-muted-foreground">
                      Изображение, PDF или DWG (AutoCAD)
                    </p>
                    <div className="text-xs bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2 mt-2 text-left max-w-md mx-auto">
                      <strong>DWG &gt; 50 MB</strong> — экспортируйте в PNG в AutoCAD: <br />
                      File → Export → PNG (или Plot → PNG)
                      <br />
                      <strong>Чертёж отображается некорректно?</strong> — экспортируйте план в PNG в AutoCAD для лучшего результата
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    className="gap-2"
                    onClick={() => floorPlanInputRef.current?.click()}
                  >
                    <Upload className="h-4 w-4" />
                    Выбрать файл
                  </Button>
                </div>
              </div>
            )}

            {(floorPlan || drawnPlan) && planMode !== 'library' && (
              <div className="space-y-4">
                {/* Zoom Controls */}
                <div className="flex items-center justify-center gap-2">
                  <div className="flex items-center gap-1 p-1 bg-background/80 backdrop-blur-md rounded-lg border border-border/50 shadow-sm">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 hover:bg-primary/20"
                      onClick={() => setPlanScale(s => Math.min(s + 0.2, 3))}
                      title="Увеличить план"
                    >
                      <ZoomIn className="h-4 w-4" />
                    </Button>
                    <span className="text-xs font-mono font-medium text-foreground px-2 select-none min-w-[3rem] text-center">
                      {(planScale * 100).toFixed(0)}%
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 hover:bg-primary/20"
                      onClick={() => setPlanScale(s => Math.max(s - 0.2, 0.5))}
                      title="Уменьшить план"
                    >
                      <ZoomOut className="h-4 w-4" />
                    </Button>
                    <div className="h-6 w-px bg-border mx-1" />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-foreground"
                      onClick={() => setPlanScale(1)}
                      title="Сбросить масштаб"
                    >
                      <RotateCcw className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                {/* Floor Plan Preview with Zoom */}
                <div className="relative rounded-xl overflow-auto bg-secondary/50 max-w-4xl mx-auto max-h-[500px] border">
                  <div
                    className="relative inline-block min-w-full transition-transform duration-200"
                    style={{
                      transform: `scale(${planScale})`,
                      transformOrigin: 'center top'
                    }}
                  >
                    {drawnPlan ? (
                      <div className="bg-slate-950 p-8 rounded-lg bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:20px_20px] min-w-[800px] relative">
                        <svg viewBox="0 0 800 600" className="w-full h-auto cursor-crosshair" onClick={handleFloorPlanClick}>
                          {drawnPlan.map((s: any) => s.type === 'rect' ?
                            <rect key={s.id} x={Math.min(s.points[0].x, s.points[1].x)} y={Math.min(s.points[0].y, s.points[1].y)} width={Math.abs(s.points[1].x - s.points[0].x)} height={Math.abs(s.points[1].y - s.points[0].y)} fill="rgba(56, 189, 248, 0.2)" stroke="#38bdf8" strokeWidth="2" /> :
                            <line key={s.id} x1={s.points[0].x} y1={s.points[0].y} x2={s.points[1].x} y2={s.points[1].y} stroke="white" strokeWidth="3" strokeLinecap="round" />
                          )}

                          {/* Точка отсчета внутри SVG для точности */}
                          {referencePoint && (
                            <g transform={`translate(${(referencePoint.x / 100) * 800}, ${(referencePoint.y / 100) * 600})`}>
                              <circle r="12" fill="rgba(239, 68, 68, 0.2)" className="animate-pulse" />
                              <circle r="6" fill="rgb(239, 68, 68)" stroke="white" strokeWidth="2" />
                              <text y="-15" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Точка отсчета</text>
                            </g>
                          )}
                          {/* Направление движения (стрелка от точки отсчета) */}
                          {referencePoint && directionPoint && (
                            <g>
                              <defs>
                                <marker id="arrowhead-direction" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
                                  <polygon points="0 0, 6 2, 0 4" fill="#22c55e" />
                                </marker>
                              </defs>
                              <line
                                x1={(referencePoint.x / 100) * 800}
                                y1={(referencePoint.y / 100) * 600}
                                x2={(directionPoint.x / 100) * 800}
                                y2={(directionPoint.y / 100) * 600}
                                stroke="#22c55e"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                markerEnd="url(#arrowhead-direction)"
                              />
                              <circle cx={(directionPoint.x / 100) * 800} cy={(directionPoint.y / 100) * 600} r="3" fill="#22c55e" stroke="white" strokeWidth="1" />
                              <text x={(directionPoint.x / 100) * 800} y={(directionPoint.y / 100) * 600 - 8} textAnchor="middle" fill="#166534" fontSize="9" fontWeight="600">Направление</text>
                            </g>
                          )}
                        </svg>
                      </div>
                    ) : floorPlanFile?.type === "application/pdf" ? (
                      <div className="relative">
                        <iframe
                          src={floorPlan}
                          className="w-full h-96 border-0 cursor-crosshair"
                          title="План помещения (PDF)"
                          onClick={handleFloorPlanClick}
                        />
                      </div>
                    ) : (
                      <div className="relative">
                        <img
                          ref={planImgRef}
                          src={floorPlan}
                          alt="План помещения"
                          className="w-full h-auto object-contain cursor-crosshair"
                          onClick={handleFloorPlanClick}
                          onLoad={measurePlanImgLayout}
                        />
                        {referencePoint && directionPoint && (
                          <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                            <defs>
                              <marker id="arrowhead-direction-img" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
                                <polygon points="0 0, 6 2, 0 4" fill="#22c55e" />
                              </marker>
                            </defs>
                            {planImgLayout ? (
                              <line
                                x1={(planImgLayout.dispX + (referencePoint.x / 100) * planImgLayout.dispW) / planImgLayout.cw * 100}
                                y1={(planImgLayout.dispY + (referencePoint.y / 100) * planImgLayout.dispH) / planImgLayout.ch * 100}
                                x2={(planImgLayout.dispX + (directionPoint.x / 100) * planImgLayout.dispW) / planImgLayout.cw * 100}
                                y2={(planImgLayout.dispY + (directionPoint.y / 100) * planImgLayout.dispH) / planImgLayout.ch * 100}
                                stroke="#22c55e"
                                strokeWidth="0.8"
                                strokeLinecap="round"
                                markerEnd="url(#arrowhead-direction-img)"
                              />
                            ) : (
                              <line
                                x1={referencePoint.x}
                                y1={referencePoint.y}
                                x2={directionPoint.x}
                                y2={directionPoint.y}
                                stroke="#22c55e"
                                strokeWidth="0.8"
                                strokeLinecap="round"
                                markerEnd="url(#arrowhead-direction-img)"
                              />
                            )}
                          </svg>
                        )}
                        {referencePoint && (
                          <div
                            className="absolute w-4 h-4 bg-red-500 border-2 border-white rounded-full shadow-lg transform -translate-x-1/2 -translate-y-1/2 animate-pulse"
                            style={planImgLayout ? {
                              left: `${(planImgLayout.dispX + (referencePoint.x / 100) * planImgLayout.dispW) / planImgLayout.cw * 100}%`,
                              top: `${(planImgLayout.dispY + (referencePoint.y / 100) * planImgLayout.dispH) / planImgLayout.ch * 100}%`,
                            } : {
                              left: `${referencePoint.x}%`,
                              top: `${referencePoint.y}%`,
                            }}
                          >
                            <div className="absolute -top-8 -left-8 bg-red-500 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
                              Точка отсчета
                            </div>
                          </div>
                        )}
                        {referencePoint && directionPoint && (
                          <div
                            className="absolute w-2 h-2 bg-green-500 border border-white rounded-full transform -translate-x-1/2 -translate-y-1/2"
                            style={planImgLayout ? {
                              left: `${(planImgLayout.dispX + (directionPoint.x / 100) * planImgLayout.dispW) / planImgLayout.cw * 100}%`,
                              top: `${(planImgLayout.dispY + (directionPoint.y / 100) * planImgLayout.dispH) / planImgLayout.ch * 100}%`,
                            } : {
                              left: `${directionPoint.x}%`,
                              top: `${directionPoint.y}%`,
                            }}
                          >
                            <div className="absolute -top-5 left-1 bg-green-600 text-white text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap">
                              Направление
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                    <div className="flex flex-wrap items-center gap-4">
                      <span className="text-muted-foreground">
                        План загружен: {floorPlanFile?.name || 'план помещения'}
                      </span>
                      {referencePoint && (
                        <span className="text-green-600 text-xs">
                          Точка отсчета: ({referencePoint.x.toFixed(1)}%, {referencePoint.y.toFixed(1)}%)
                        </span>
                      )}
                      {directionPoint && (
                        <span className="text-emerald-600 text-xs">
                          Направление: ({directionPoint.x.toFixed(1)}%, {directionPoint.y.toFixed(1)}%)
                        </span>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {referencePoint && (
                        <>
                          <Button
                            variant={setDirectionMode ? "default" : "outline"}
                            size="sm"
                            className="gap-1 text-emerald-600 hover:text-emerald-700 border-emerald-500/50 bg-emerald-500/10"
                            onClick={() => setSetDirectionMode(v => !v)}
                            title="Кликните на план выше, чтобы указать направление"
                          >
                            <Navigation className="h-4 w-4" />
                            Указать направление
                          </Button>
                          {directionPoint && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="gap-1 text-emerald-600 hover:text-emerald-700"
                              onClick={handleDirectionPointRemove}
                            >
                              <X className="h-4 w-4" />
                              Сбросить направление
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            className="gap-1 text-orange-600 hover:text-orange-700"
                            onClick={handleReferencePointRemove}
                          >
                            <X className="h-4 w-4" />
                            Сбросить точку
                          </Button>
                        </>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="gap-1 text-destructive hover:text-destructive"
                        onClick={handleFloorPlanRemove}
                      >
                        <X className="h-4 w-4" />
                        Удалить план
                      </Button>
                    </div>
                  </div>

                <div className="text-xs text-muted-foreground text-center space-y-1">
                  <p>Изображения используются как фон для траектории</p>
                  <p>PDF и DWG конвертируются в PNG на сервере</p>
                  <p className="text-blue-600 font-medium">Кликните на план, чтобы установить точку отсчета траектории</p>
                  <p className="text-green-600 font-medium">Траектория будет рисоваться от выбранной точки</p>
                  <p className="text-emerald-600 font-medium">Нажмите «Указать направление» и кликните на план — траектория выровняется по направлению движения</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Tabs defaultValue="upload" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="upload" className="gap-2">
              <Video className="h-4 w-4" />
              Новое видео
            </TabsTrigger>
            <TabsTrigger value="library" className="gap-2">
              <Library className="h-4 w-4" />
              Библиотека
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <TrajectoryAnalysis onTrajectoryAnalyzed={handleTrajectoryAnalyzed} />

                {stats && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Статистика анализа</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center p-4 bg-secondary rounded-lg">
                          <div className="text-2xl font-bold text-primary">
                            {stats.estimated_distance?.toFixed(1)}
                          </div>
                          <div className="text-sm text-muted-foreground">метров пройдено</div>
                        </div>
                        <div className="text-center p-4 bg-secondary rounded-lg">
                          <div className="text-2xl font-bold text-primary">
                            {turnPoints.length}
                          </div>
                          <div className="text-sm text-muted-foreground">поворотов обнаружено</div>
                        </div>
                        <div className="text-center p-4 bg-secondary rounded-lg">
                          <div className="text-2xl font-bold text-primary">
                            {stats.fps?.toFixed(1)}
                          </div>
                          <div className="text-sm text-muted-foreground">FPS обработки</div>
                        </div>
                        <div className="text-center p-4 bg-secondary rounded-lg">
                          <div className="text-2xl font-bold text-primary">
                            {stats.scale_factor}
                          </div>
                          <div className="text-sm text-muted-foreground">масштаб</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>

              <div className="lg:col-span-1">
                <Card className="h-[600px]">
                  <CardContent className="p-0 h-full">
                    <TrajectoryMap
                      trajectories={Array.isArray(trajectory) && trajectory.length > 0 && typeof trajectory[0] === 'object' && 'trajectory' in trajectory[0] ? trajectory : undefined}
                      trajectory={!Array.isArray(trajectory) || (trajectory.length > 0 && typeof trajectory[0] !== 'object') ? trajectory : undefined}
                      turnPoints={!Array.isArray(trajectory) || (trajectory.length > 0 && typeof trajectory[0] !== 'object') ? turnPoints : undefined}
                      stats={stats}
                      floorPlan={floorPlanFile?.type === "application/pdf" ? null : floorPlan}
                      drawnPlan={drawnPlan}
                      referencePoint={referencePoint}
                      directionPoint={directionPoint}
                      setDirectionMode={setDirectionMode}
                      onSetDirectionModeChange={setSetDirectionMode}
                      onDirectionPointSet={handleDirectionPointSet}
                    />
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="library" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-1">
                <VideoLibrary
                  onVideoSelected={handleVideoSelected}
                  onAnalysisLoaded={handleAnalysisLoaded}
                />
              </div>

              <div className="lg:col-span-2 space-y-6">
                {selectedVideo && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Video className="h-5 w-5" />
                        {selectedVideo.filename}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {videoUrl && (
                        <div className="mb-4">
                          <video
                            src={videoUrl}
                            controls
                            className="w-full max-h-96 rounded-lg"
                            preload="metadata"
                          >
                            Ваш браузер не поддерживает видео.
                          </video>
                        </div>
                      )}

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Размер:</span>
                          <div className="font-medium">
                            {(selectedVideo.file_size / 1024 / 1024).toFixed(1)} MB
                          </div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Загружено:</span>
                          <div className="font-medium">
                            {new Date(selectedVideo.uploaded_at).toLocaleDateString('ru-RU')}
                          </div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Масштаб:</span>
                          <div className="font-medium">{selectedVideo.scale_factor}</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Стабилизация:</span>
                          <div className="font-medium">
                            {selectedVideo.stabilized ? "Да" : "Нет"}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                <Card className="h-[400px]">
                  <CardContent className="p-0 h-full">
                    <TrajectoryMap
                      trajectories={Array.isArray(trajectory) && trajectory.length > 0 && typeof trajectory[0] === 'object' && 'trajectory' in trajectory[0] ? trajectory : undefined}
                      trajectory={!Array.isArray(trajectory) || (trajectory.length > 0 && typeof trajectory[0] !== 'object') ? trajectory : undefined}
                      turnPoints={!Array.isArray(trajectory) || (trajectory.length > 0 && typeof trajectory[0] !== 'object') ? turnPoints : undefined}
                      stats={stats}
                      floorPlan={floorPlanFile?.type === "application/pdf" ? null : floorPlan}
                      drawnPlan={drawnPlan}
                      referencePoint={referencePoint}
                      directionPoint={directionPoint}
                      setDirectionMode={setDirectionMode}
                      onSetDirectionModeChange={setSetDirectionMode}
                      onDirectionPointSet={handleDirectionPointSet}
                    />
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {turnPoints.length > 0 && (
          <Card className="mt-8">
            <CardHeader>
              <CardTitle>Детали поворотов</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">#</th>
                      <th className="text-left p-2">Тип</th>
                      <th className="text-left p-2">Угол (°)</th>
                      <th className="text-left p-2">Координаты</th>
                      <th className="text-left p-2">Кадр</th>
                    </tr>
                  </thead>
                  <tbody>
                    {turnPoints.map((turn, index) => (
                      <tr key={index} className="border-b">
                        <td className="p-2 font-medium">{index + 1}</td>
                        <td className="p-2">
                          <span className={`px-2 py-1 rounded text-xs ${turn.turn_type === 'left' ? 'bg-orange-100 text-orange-800' : 'bg-purple-100 text-purple-800'}`}>
                            {turn.turn_type === 'left' ? 'Влево' : 'Вправо'}
                          </span>
                        </td>
                        <td className="p-2">{turn.angle_degrees?.toFixed(1)}°</td>
                        <td className="p-2">
                          ({turn.position?.[0]?.toFixed(1)}, {turn.position?.[1]?.toFixed(1)})
                        </td>
                        <td className="p-2">{turn.frame_index}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {isEditorOpen && (
        <PlanEditor
          onSave={handlePlanSaved}
          onCancel={() => setIsEditorOpen(false)}
        />
      )}
    </div>
  );
};

export default TrajectoryAnalysisPage;