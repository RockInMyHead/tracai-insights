import { useEffect, useRef, useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Thermometer, Route, ZoomIn, ZoomOut, RotateCcw, Maximize, Maximize2, Minimize2, X, Map as MapIcon, Footprints, Ruler, Compass } from "lucide-react";
import { correctPathWithFloorPlan } from "@/lib/pathfinding";
import { finiteNum } from "@/lib/numbers";

export interface TrajectoryPoint {
  x: number;
  y: number;
  z: number;
}

export interface TurnPoint {
  frame_index: number;
  trajectory_index: number;
  angle_degrees: number;
  position: number[];
  turn_type: string;
}

/** Элемент нарисованного плана в PlanEditor */
interface DrawnPlanShape {
  id: string;
  type: string;
  points: { x: number; y: number }[];
}

export interface TrajectoryData {
  trajectory: TrajectoryPoint[];
  turnPoints: TurnPoint[];
  ownerName: string;
  color: string;
  mapAligned?: boolean;
}

interface TrajectoryMapProps {
  trajectory?: TrajectoryPoint[]; // Для обратной совместимости
  turnPoints?: TurnPoint[]; // Для обратной совместимости
  trajectories?: TrajectoryData[]; // Новый формат для множественных траекторий
  stats?: Record<string, unknown>;
  floorPlan?: string | null;
  drawnPlan?: unknown[] | null;
  referencePoint?: { x: number; y: number } | null;
  directionPoint?: { x: number; y: number } | null;
  setDirectionMode?: boolean;
  onSetDirectionModeChange?: (enabled: boolean) => void;
  onDirectionPointSet?: (point: { x: number; y: number }) => void;
}

function normalizeTrajectoryPoints(traj: unknown): TrajectoryPoint[] {
  if (!traj || !Array.isArray(traj) || traj.length === 0) return [];
  const first = traj[0];
  if (Array.isArray(first)) {
    return traj.map((p: unknown) => {
      const arr = p as number[];
      return { x: finiteNum(arr[0]), y: finiteNum(arr[1]), z: finiteNum(arr[2]) };
    });
  }
  if (typeof first === 'object' && first !== null && ('x' in first || 0 in first)) {
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
}

const TrajectoryMap = ({ trajectory, turnPoints, trajectories, stats, floorPlan, drawnPlan, referencePoint, directionPoint, setDirectionMode, onSetDirectionModeChange, onDirectionPointSet }: TrajectoryMapProps) => {
  // Поддержка старого формата + нормализация точек (массив массивов → {x,y,z}[])
  const trajectoryData = useMemo(() => {
    const raw = trajectories || (trajectory ? [{
      trajectory: Array.isArray(trajectory) ? trajectory : [],
      turnPoints: turnPoints || [],
      ownerName: 'Пользователь',
      color: '#3b82f6'
    }] : []);
    return raw.map((item) => ({
      ...item,
      trajectory: normalizeTrajectoryPoints(item.trajectory),
    }));
  }, [trajectories, trajectory, turnPoints]);

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [viewBox, setViewBox] = useState({ width: 800, height: 600 });
  const [fixedViewBox, setFixedViewBox] = useState<{ width: number; height: number } | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<TurnPoint | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);

  // State for independent floor plan zoom/scale
  const [planScale, setPlanScale] = useState(1);
  const [imageSize, setImageSize] = useState({ width: 800, height: 600 });

  // Trajectory calibration states
  const [trajScale, setTrajScale] = useState(1);
  const [trajOffset, setTrajOffset] = useState({ x: 0, y: 0 });
  const [showCalibration, setShowCalibration] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Нативный полноэкранный режим (Fullscreen API) — чертёж на весь экран
  const enterFullscreen = () => {
    const el = containerRef.current;
    if (!el) return;
    const req = el.requestFullscreen ?? (el as HTMLElement & { webkitRequestFullscreen?: () => Promise<void> }).webkitRequestFullscreen;
    if (req && !document.fullscreenElement && !(document as Document & { webkitFullscreenElement?: Element }).webkitFullscreenElement) {
      req.call(el).then(() => setIsFullscreen(true)).catch(() => setIsFullscreen(false));
    } else {
      setIsFullscreen(true);
    }
  };
  const exitFullscreen = () => {
    const ex = document.exitFullscreen ?? (document as Document & { webkitExitFullscreen?: () => Promise<void> }).webkitExitFullscreen;
    if (ex && (document.fullscreenElement || (document as Document & { webkitFullscreenElement?: Element }).webkitFullscreenElement)) {
      ex.call(document).then(() => setIsFullscreen(false)).catch(() => setIsFullscreen(false));
    } else {
      setIsFullscreen(false);
    }
  };
  useEffect(() => {
    const isFs = () => !!(document.fullscreenElement ?? (document as Document & { webkitFullscreenElement?: Element }).webkitFullscreenElement);
    const onFullscreenChange = () => setIsFullscreen(isFs());
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isFs()) exitFullscreen();
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    document.addEventListener("webkitfullscreenchange", onFullscreenChange);
    window.addEventListener("keydown", onKeyDown);
    document.body.style.overflow = isFullscreen ? "hidden" : "";
    return () => {
      document.removeEventListener("fullscreenchange", onFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", onFullscreenChange);
      window.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = "";
    };
  }, [isFullscreen]);

  // Pathfinding: траектория следует плану, не проходит сквозь стены
  const [usePathfinding, setUsePathfinding] = useState(true);
  const [correctedPaths, setCorrectedPaths] = useState<Map<number, { x: number; y: number }[]>>(new Map());

  // Калибровка масштаба по двум точкам (известное расстояние в метрах)
  const [scaleCalibStep, setScaleCalibStep] = useState<'idle' | 'A' | 'B' | 'input'>('idle');
  const [scalePointA, setScalePointA] = useState<{ x: number; y: number } | null>(null);
  const [scalePointB, setScalePointB] = useState<{ x: number; y: number } | null>(null);
  const [scaleDistanceInput, setScaleDistanceInput] = useState("");

  // Компас: угол поворота плана (0–360°), учитывается при отрисовке траектории
  const [compassAngle, setCompassAngle] = useState(() => {
    try {
      const saved = localStorage.getItem('compassAngle');
      return saved ? parseFloat(saved) : 0;
    } catch { return 0; }
  });

  // Разворот траектории на 180° при несовпадении с указанным направлением (SLAM/камера)
  const [directionFlip180, setDirectionFlip180] = useState(() => {
    try {
      return localStorage.getItem('directionFlip180') === 'true';
    } catch { return false; }
  });

  // Ручная подстройка угла траектории относительно стрелки направления (градусы)
  const [directionAngleOffset, setDirectionAngleOffset] = useState(() => {
    try {
      const v = localStorage.getItem('directionAngleOffset');
      return v ? parseFloat(v) : 0;
    } catch { return 0; }
  });

  useEffect(() => {
    try {
      localStorage.setItem('directionFlip180', String(directionFlip180));
    } catch { /* ignore */ }
  }, [directionFlip180]);

  useEffect(() => {
    try {
      localStorage.setItem('directionAngleOffset', String(directionAngleOffset));
    } catch { /* ignore */ }
  }, [directionAngleOffset]);

  // Load image size when floorPlan changes
  useEffect(() => {
    if (floorPlan) {
      const img = new Image();
      img.onload = () => {
        if (img.width > 0 && img.height > 0) {
          setImageSize({ width: img.width, height: img.height });
        }
      };
      img.src = floorPlan;
    }
  }, [floorPlan]);

  // Transform trajectory coordinates relative to reference point (with rotation)
  const getTransformedTrajectories = useMemo(() => {
    if (!trajectoryData || trajectoryData.length === 0) return [];

    return trajectoryData.map(data => {
      const validPoints = data.trajectory.filter(p =>
        p &&
        typeof p.x === 'number' && !isNaN(p.x) &&
        typeof p.y === 'number' && !isNaN(p.y)
      );

      if (validPoints.length === 0) return { ...data, trajectory: [] };
      if (data.mapAligned) {
        return {
          ...data,
          trajectory: validPoints,
          turnPoints: data.turnPoints || []
        };
      }

      // Base coordinates from the first point of THIS trajectory
      const startX = validPoints[0].x;
      const startY = validPoints[0].y;

      let transformed: { x: number; y: number; z?: number }[] = [];
      let refX: number, refY: number;

      if (referencePoint) {
        refX = (referencePoint.x / 100) * viewBox.width;
        refY = (referencePoint.y / 100) * viewBox.height;

        transformed = validPoints.map(point => ({
          ...point,
          x: (point.x - startX) * trajScale + refX + trajOffset.x,
          y: (point.y - startY) * trajScale + refY + trajOffset.y
        }));
      } else {
        refX = viewBox.width / 2;
        refY = viewBox.height / 2;

        transformed = validPoints.map(point => ({
          ...point,
          x: (point.x - startX) * trajScale + refX + trajOffset.x,
          y: (point.y - startY) * trajScale + refY + trajOffset.y
        }));
      }

      // Вычисляем угол поворота: при указанном направлении — только оно; иначе — компас
      let rotationRad: number;
      if (directionPoint && referencePoint) {
        const dirX = (directionPoint.x / 100) * viewBox.width;
        const dirY = (directionPoint.y / 100) * viewBox.height;
        const dist = Math.hypot(dirX - refX, dirY - refY);
        if (dist <= viewBox.width * 0.02) {
          rotationRad = (compassAngle * Math.PI) / 180 + (directionAngleOffset * Math.PI) / 180;
        } else {
          const directionAngle = Math.atan2(dirY - refY, dirX - refX);
          if (validPoints.length >= 2) {
            // Берём угол по более длинному сегменту (первые ~10% точек), чтобы уменьшить влияние шума
            const segLen = Math.max(2, Math.min(20, Math.floor(validPoints.length * 0.1)));
            const p0 = validPoints[0];
            const pN = validPoints[segLen - 1];
            const dx = (pN.x - p0.x) * trajScale;
            const dy = (pN.y - p0.y) * trajScale;
            const segDist = Math.hypot(dx, dy);
            const trajAngle = segDist > 1e-6 ? Math.atan2(dy, dx) : Math.atan2((validPoints[1].y - p0.y) * trajScale, (validPoints[1].x - p0.x) * trajScale);
            rotationRad = directionAngle - trajAngle;
            if (directionFlip180) rotationRad += Math.PI;
            rotationRad += (directionAngleOffset * Math.PI) / 180;
          } else {
            rotationRad = directionAngle + (directionAngleOffset * Math.PI) / 180;
          }
        }
      } else {
        rotationRad = (compassAngle * Math.PI) / 180;
      }

      const cos = Math.cos(rotationRad);
      const sin = Math.sin(rotationRad);
      const applyRotation = (px: number, py: number) => {
        const dx = px - refX;
        const dy = py - refY;
        return {
          x: dx * cos - dy * sin + refX,
          y: dx * sin + dy * cos + refY
        };
      };

      if (Math.abs(rotationRad) > 1e-6) {
        transformed = transformed.map(point => ({
          ...point,
          ...applyRotation(point.x, point.y)
        }));
      }

      // Трансформируем точки поворотов (position) так же, как траекторию
      const transformedTurnPoints = (data.turnPoints || []).map(turn => {
        const pos = turn.position;
        if (!pos || !Array.isArray(pos)) return turn;
        const posRec = pos as number[] | Record<string, unknown>;
        const pxVal =
          typeof pos[0] === "number" ? pos[0] : finiteNum((posRec as Record<string, unknown>).x);
        const pyVal =
          typeof pos[1] === "number" ? pos[1] : finiteNum((posRec as Record<string, unknown>).y);
        if (typeof pxVal !== 'number' || typeof pyVal !== 'number' || isNaN(pxVal) || isNaN(pyVal)) return turn;
        const px = (pxVal - startX) * trajScale + refX + trajOffset.x;
        const py = (pyVal - startY) * trajScale + refY + trajOffset.y;
        const rotated = applyRotation(px, py);
        return { ...turn, position: [rotated.x, rotated.y, pos[2] ?? 0] };
      });

      return {
        ...data,
        trajectory: transformed,
        turnPoints: transformedTurnPoints
      };
    });
  }, [trajectoryData, referencePoint, directionPoint, viewBox.width, viewBox.height, trajScale, trajOffset, compassAngle, directionFlip180, directionAngleOffset]);

  // Get transformed turn points for all trajectories
  const getAllTransformedTurnPoints = useMemo(() => {
    return getTransformedTrajectories.flatMap(data => data.turnPoints || []);
  }, [getTransformedTrajectories]);

  // Pathfinding: корректируем траекторию по плану помещения
  useEffect(() => {
    if (!floorPlan || !usePathfinding || getTransformedTrajectories.length === 0) {
      setCorrectedPaths(new Map());
      return;
    }
    if (floorPlan.includes("application/pdf")) return; // PDF не поддерживается для pathfinding

    let cancelled = false;
    const run = async () => {
      const next = new Map<number, { x: number; y: number }[]>();
      for (let i = 0; i < getTransformedTrajectories.length; i++) {
        if (cancelled) return;
        const data = getTransformedTrajectories[i];
        const points = data.trajectory.map(p => ({ x: p.x, y: p.y }));
        if (points.length < 2) continue;
        if (data.mapAligned) {
          next.set(i, points);
          continue;
        }
        try {
          const corrected = await correctPathWithFloorPlan(
            floorPlan,
            points,
            viewBox.width,
            viewBox.height
          );
          if (!cancelled) next.set(i, corrected);
        } catch {
          if (!cancelled) next.set(i, points);
        }
      }
      if (!cancelled) setCorrectedPaths(next);
    };
    run();
    return () => { cancelled = true; };
  }, [floorPlan, usePathfinding, getTransformedTrajectories, viewBox.width, viewBox.height]);

  // Generate heatmap data based on trajectory density
  const generateHeatmapData = useMemo(() => {
    const allTrajectories = getTransformedTrajectories.flatMap(data => data.trajectory);
    if (allTrajectories.length === 0) return [];

    // Create a grid to calculate density
    const gridSize = 20;
    const grid = new Map();

    // Count points in each grid cell
    allTrajectories.forEach(point => {
      const gridX = Math.floor(point.x / gridSize);
      const gridY = Math.floor(point.y / gridSize);
      const key = `${gridX}-${gridY}`;

      grid.set(key, (grid.get(key) || 0) + 1);
    });

    // Convert grid to heatmap data with intensity
    const heatmapData = [];
    const maxDensity = Math.max(...grid.values());

    for (const [key, density] of grid.entries()) {
      const [gridX, gridY] = key.split('-').map(Number);
      const intensity = density / maxDensity;

      heatmapData.push({
        x: gridX * gridSize,
        y: gridY * gridSize,
        width: gridSize,
        height: gridSize,
        intensity,
      });
    }

    return heatmapData;
  }, [getTransformedTrajectories]);

  // Calculate movement analysis data
  const movementAnalysis = useMemo(() => {
    const allTrajectories = getTransformedTrajectories.flatMap(data => data.trajectory);
    if (allTrajectories.length < 2) return null;

    let totalDistance = 0;
    const totalTime = allTrajectories.length * 0.033; // Assuming 30fps
    let maxSpeed = 0;
    let avgSpeed = 0;

    for (let i = 1; i < allTrajectories.length; i++) {
      const prev = allTrajectories[i - 1];
      const curr = allTrajectories[i];

      const distance = Math.sqrt(
        Math.pow(curr.x - prev.x, 2) + Math.pow(curr.y - prev.y, 2)
      );
      totalDistance += distance;

      const speed = distance / 0.033; // pixels per second
      maxSpeed = Math.max(maxSpeed, speed);
      avgSpeed += speed;
    }

    avgSpeed /= (allTrajectories.length - 1);

    return {
      totalDistance,
      totalTime,
      avgSpeed,
      maxSpeed,
      trajectoryPoints: allTrajectories.length
    };
  }, [getTransformedTrajectories]);


  useEffect(() => {
    if (floorPlan && imageSize.width > 0) {
      setViewBox({ width: imageSize.width, height: imageSize.height });
      return;
    }

    if (drawnPlan) {
      setViewBox({ width: 800, height: 600 });
      return;
    }

    const allTrajectories = trajectoryData.flatMap(data => data.trajectory);
    if (allTrajectories && allTrajectories.length > 0) {
      const validPoints = allTrajectories.filter(p =>
        p && typeof p.x === 'number' && !isNaN(p.x) && typeof p.y === 'number' && !isNaN(p.y)
      );

      if (validPoints.length > 0) {
        const xCoords = validPoints.map(p => p.x);
        const yCoords = validPoints.map(p => p.y);
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);

        const width = Math.max(maxX - minX, 100);
        const height = Math.max(maxY - minY, 100);
        const padding = Math.max(width, height) * 0.2;

        setViewBox({
          width: Math.max(width + 2 * padding, 400),
          height: Math.max(height + 2 * padding, 400)
        });
      } else {
        setViewBox({ width: 800, height: 600 });
      }
    } else {
      setViewBox({ width: 800, height: 600 });
    }
  }, [trajectoryData, imageSize, floorPlan, drawnPlan]);

  // Авто-масштаб траектории при наличии плана: если линия получается < 30px — подобрать масштаб
  useEffect(() => {
    if (!floorPlan || !trajectoryData.length) return;
    const allPoints = trajectoryData.flatMap(d => d.trajectory).filter(
      p => p && typeof p.x === 'number' && !isNaN(p.x) && typeof p.y === 'number' && !isNaN(p.y)
    );
    if (allPoints.length < 2) return;
    const xs = allPoints.map(p => p.x);
    const ys = allPoints.map(p => p.y);
    const extentX = Math.max(Math.max(...xs) - Math.min(...xs), 1e-6);
    const extentY = Math.max(Math.max(...ys) - Math.min(...ys), 1e-6);
    const extent = Math.max(extentX, extentY);
    const targetSize = Math.min(viewBox.width, viewBox.height) * 0.4;
    const suggestedScale = targetSize / extent;
    setTrajScale(prev => {
      const currentSpan = extent * prev;
      if (currentSpan < 30 && suggestedScale > 0.5 && suggestedScale < 500) return suggestedScale;
      return prev;
    });
  }, [floorPlan, trajectoryData, viewBox.width, viewBox.height]);

  const getPathString = (trajectory: { x: number; y: number }[]) => {
    if (trajectory.length === 0) return "";

    return trajectory.reduce((acc, point, i) => {
      const x = point.x;
      const y = point.y;
      return acc + (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    }, "");
  };

  const handleDirectionOverlayClick = (e: React.MouseEvent<SVGRectElement>) => {
    if (!onDirectionPointSet || !referencePoint) return;
    const rect = e.currentTarget;
    const pt = rect.ownerSVGElement!.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const m = rect.getScreenCTM()?.inverse();
    if (!m) return;
    const svgPt = pt.matrixTransform(m);
    const x = (svgPt.x / viewBox.width) * 100;
    const y = (svgPt.y / viewBox.height) * 100;
    onDirectionPointSet({ x, y });
    e.stopPropagation();
  };

  const handleScaleCalibClick = (e: React.MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget;
    const pt = rect.ownerSVGElement!.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const m = rect.getScreenCTM()?.inverse();
    if (!m) return;
    const svgPt = pt.matrixTransform(m);
    const x = svgPt.x;
    const y = svgPt.y;
    if (scaleCalibStep === 'A') {
      setScalePointA({ x, y });
      setScaleCalibStep('B');
    } else if (scaleCalibStep === 'B') {
      setScalePointB({ x, y });
      setScaleCalibStep('input');
    }
    e.stopPropagation();
  };

  const applyScaleCalibration = () => {
    const distM = parseFloat(scaleDistanceInput.replace(',', '.'));
    if (!scalePointA || !scalePointB || !distM || distM <= 0) return;
    const px = Math.sqrt((scalePointB.x - scalePointA.x) ** 2 + (scalePointB.y - scalePointA.y) ** 2);
    const pixelsPerMeter = px / distM;
    setTrajScale(pixelsPerMeter);
    setScaleCalibStep('idle');
    setScalePointA(null);
    setScalePointB(null);
    setScaleDistanceInput("");
  };

  const resetView = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  const zoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 5));
  };

  const zoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.2));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left click
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setScale(prev => {
      const newScale = Math.max(0.2, Math.min(prev + delta, 5));
      return newScale;
    });
  };

  // Показываем заглушку только если нет ни плана, ни данных траектории
  const hasContent = floorPlan || drawnPlan || (trajectoryData.length > 0 && !trajectoryData.every(data => data.trajectory.length === 0));

  if (!hasContent) {
    return (
      <div className="flex items-center justify-center h-full min-h-[400px] bg-secondary/20 rounded-2xl border-2 border-dashed border-border/50 m-4">
        <div className="text-center space-y-4">
          <MapIcon className="h-12 w-12 text-muted-foreground mx-auto opacity-50" />
          <p className="text-muted-foreground font-medium">Загрузите план или видео для начала анализа</p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full rounded-2xl bg-gradient-card border border-border/50 overflow-hidden cursor-move ${isFullscreen ? "fixed inset-0 z-[100] rounded-none border-0 bg-background min-w-full min-h-full" : ""}`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    >
      {/* Кнопка выхода из полноэкранного режима */}
      {isFullscreen && (
        <div className="absolute top-4 right-4 z-50 flex items-center gap-2 bg-background/90 backdrop-blur-md px-3 py-2 rounded-lg border border-border/50 shadow-xl">
          <span className="text-sm font-medium text-muted-foreground">Чертёж на весь экран</span>
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5"
            onClick={(e) => { e.stopPropagation(); exitFullscreen(); }}
            title="Выйти (Escape)"
          >
            <Minimize2 className="h-4 w-4" />
            Выйти
          </Button>
        </div>
      )}

      {/* Floor Plan Controls (Top-Left) */}
      {(floorPlan || drawnPlan) && (
        <div className="absolute top-4 left-4 z-20 flex flex-col gap-2 bg-background/80 backdrop-blur-md p-2 rounded-lg border border-border/50 shadow-xl" onClick={(e) => e.stopPropagation()}>
          <span className="text-[10px] font-bold text-center text-muted-foreground uppercase tracking-wider mb-1">План</span>
          {referencePoint && onSetDirectionModeChange && (
            <Button
              variant={setDirectionMode ? "default" : "ghost"}
              size="sm"
              className="h-8 gap-1.5 text-[10px] text-emerald-600 hover:text-emerald-700"
              onClick={(e) => { e.stopPropagation(); onSetDirectionModeChange(!setDirectionMode); }}
              title="Кликните на карту, чтобы указать направление движения"
            >
              <Compass className="h-3.5 w-3.5" />
              {setDirectionMode ? "Кликните на карту..." : "Указать направление"}
            </Button>
          )}
          {referencePoint && directionPoint && (
            <>
              <Button
                variant={directionFlip180 ? "default" : "ghost"}
                size="sm"
                className="h-8 gap-1.5 text-[10px]"
                onClick={(e) => { e.stopPropagation(); setDirectionFlip180(v => !v); }}
                title="Развернуть траекторию на 180°, если она идёт в противоположную сторону"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Развернуть 180°
              </Button>
              <div className="flex items-center gap-1" title="Подстройка угла траектории относительно стрелки направления">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 w-7 p-0 text-[10px]"
                  onClick={(e) => { e.stopPropagation(); setDirectionAngleOffset(v => v - 15); }}
                >
                  −15°
                </Button>
                <span className="text-[10px] font-mono min-w-[3ch] text-center">{directionAngleOffset}°</span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 w-7 p-0 text-[10px]"
                  onClick={(e) => { e.stopPropagation(); setDirectionAngleOffset(v => v + 15); }}
                >
                  +15°
                </Button>
              </div>
            </>
          )}
          {floorPlan && !floorPlan.includes("application/pdf") && (
            <Button
              variant={usePathfinding ? "default" : "ghost"}
              size="sm"
              className="h-8 gap-1.5 text-[10px]"
              onClick={(e) => { e.stopPropagation(); setUsePathfinding(v => !v); }}
              title={usePathfinding ? "Траектория следует плану (не проходит сквозь стены)" : "Прямые линии между точками"}
            >
              <Footprints className="h-3.5 w-3.5" />
              {usePathfinding ? "По плану" : "Прямо"}
            </Button>
          )}
          <div className="flex flex-col gap-1 items-center">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 hover:bg-primary/20"
              onClick={(e) => { e.stopPropagation(); setPlanScale(s => Math.min(s + 0.1, 5)); }}
              title="Увеличить план"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            <span className="text-xs font-mono font-medium text-foreground py-1 select-none">
              {(planScale * 100).toFixed(0)}%
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 hover:bg-primary/20"
              onClick={(e) => { e.stopPropagation(); setPlanScale(s => Math.max(s - 0.1, 0.1)); }}
              title="Уменьшить план"
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            <div className="h-px w-full bg-border my-1" />
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-muted-foreground hover:text-foreground"
              onClick={(e) => { e.stopPropagation(); setPlanScale(1); }}
              title="Сбросить масштаб плана"
            >
              <RotateCcw className="h-3 w-3" />
            </Button>
          </div>
        </div>
      )}

      {/* View controls */}
      <div className="absolute top-4 right-4 z-20 flex flex-col gap-2">
        <div className="flex flex-col gap-1 p-1 bg-background/80 backdrop-blur-md rounded-lg border border-border/50 shadow-xl">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 hover:bg-primary/20"
            onClick={(e) => { e.stopPropagation(); zoomIn(); }}
            title="Приблизить"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 hover:bg-primary/20"
            onClick={(e) => { e.stopPropagation(); zoomOut(); }}
            title="Отдалить"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <div className="h-px bg-border mx-1 my-0.5" />
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 hover:bg-primary/20"
            onClick={(e) => { e.stopPropagation(); resetView(); }}
            title="Сбросить вид"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>

        <Button
          variant={showHeatmap ? "default" : "outline"}
          size="icon"
          className="h-10 w-10 shadow-lg backdrop-blur-md"
          onClick={(e) => { e.stopPropagation(); setShowHeatmap(!showHeatmap); }}
          title={showHeatmap ? "Показать траекторию" : "Показать тепловую карту"}
        >
          {showHeatmap ? <Route className="h-5 w-5" /> : <Thermometer className="h-5 w-5" />}
        </Button>

        <Button
          variant={showCalibration ? "default" : "outline"}
          size="icon"
          className="h-10 w-10 shadow-lg backdrop-blur-md bg-orange-500/10 border-orange-500/50 hover:bg-orange-500/20"
          onClick={(e) => { e.stopPropagation(); setShowCalibration(!showCalibration); }}
          title="Калибровка траектории"
        >
          <Maximize className="h-5 w-5 text-orange-500" />
        </Button>
        <Button
          variant={isFullscreen ? "default" : "outline"}
          size="icon"
          className="h-10 w-10 shadow-lg backdrop-blur-md"
          onClick={(e) => {
            e.stopPropagation();
            if (isFullscreen) exitFullscreen();
            else enterFullscreen();
          }}
          title={isFullscreen ? "Выйти с полноэкранного режима" : "Чертёж на полный экран"}
        >
          {isFullscreen ? <Minimize2 className="h-5 w-5" /> : <Maximize2 className="h-5 w-5" />}
        </Button>
      </div>

      {/* Calibration Panel */}
      {showCalibration && (
        <div
          className="absolute bottom-4 right-4 z-30 w-64 bg-background/90 backdrop-blur-xl p-4 rounded-xl border border-orange-500/30 shadow-2xl animate-in fade-in slide-in-from-bottom-4"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-bold flex items-center gap-2">
              <Maximize className="h-4 w-4 text-orange-500" />
              Калибровка траектории
            </h4>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setShowCalibration(false)}>
              <X className="h-3 w-3" />
            </Button>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-[10px] uppercase font-bold text-muted-foreground">
                <span>Масштаб</span>
                <span className="text-orange-500">{trajScale.toFixed(2)}x</span>
              </div>
              <input
                type="range" min="0.1" max="50" step="0.1"
                value={trajScale}
                onChange={(e) => setTrajScale(parseFloat(e.target.value))}
                className="w-full accent-orange-500"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-[10px] uppercase font-bold text-muted-foreground">
                <span>Смещение X</span>
                <span className="text-orange-500">{trajOffset.x}px</span>
              </div>
              <input
                type="range" min="-1000" max="1000" step="1"
                value={trajOffset.x}
                onChange={(e) => setTrajOffset(prev => ({ ...prev, x: parseInt(e.target.value) }))}
                className="w-full accent-orange-500"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-[10px] uppercase font-bold text-muted-foreground">
                <span>Смещение Y</span>
                <span className="text-orange-500">{trajOffset.y}px</span>
              </div>
              <input
                type="range" min="-1000" max="1000" step="1"
                value={trajOffset.y}
                onChange={(e) => setTrajOffset(prev => ({ ...prev, y: parseInt(e.target.value) }))}
                className="w-full accent-orange-500"
              />
            </div>

            {/* Калибровка масштаба по двум точкам */}
            <div className="pt-3 border-t border-border/50 space-y-2">
              <div className="flex items-center gap-2 text-[10px] font-bold text-muted-foreground uppercase">
                <Ruler className="h-3.5 w-3.5" />
                Масштаб по плану
              </div>
              <p className="text-[9px] text-muted-foreground leading-tight">
                Укажите два конца отрезка известной длины (напр. коридор 5 м)
              </p>
              {scaleCalibStep === 'idle' ? (
                <Button
                  variant="outline" size="sm" className="w-full h-8 text-[10px]"
                  onClick={() => setScaleCalibStep('A')}
                >
                  Указать отрезок
                </Button>
              ) : scaleCalibStep === 'A' ? (
                <p className="text-[10px] text-amber-600 font-medium">Кликните первую точку на плане</p>
              ) : scaleCalibStep === 'B' ? (
                <p className="text-[10px] text-amber-600 font-medium">Кликните вторую точку</p>
              ) : (
                <div className="space-y-2">
                  <input
                    type="text"
                    placeholder="Расстояние (м)"
                    value={scaleDistanceInput}
                    onChange={(e) => setScaleDistanceInput(e.target.value)}
                    className="w-full h-8 px-2 text-xs rounded border bg-background"
                  />
                  <div className="flex gap-1">
                    <Button size="sm" className="flex-1 h-7 text-[10px]" onClick={applyScaleCalibration}>
                      Применить
                    </Button>
                    <Button variant="ghost" size="sm" className="h-7 text-[10px]" onClick={() => { setScaleCalibStep('idle'); setScalePointA(null); setScalePointB(null); }}>
                      Отмена
                    </Button>
                  </div>
                </div>
              )}
              {scaleCalibStep !== 'idle' && scaleCalibStep !== 'input' && (
                <Button variant="ghost" size="sm" className="w-full h-6 text-[10px]" onClick={() => setScaleCalibStep('idle')}>
                  Отмена
                </Button>
              )}
            </div>

            <Button
              variant="outline" size="sm" className="w-full h-8 text-[10px] uppercase font-bold"
              onClick={() => { setTrajScale(1); setTrajOffset({ x: 0, y: 0 }); setScaleCalibStep('idle'); }}
            >
              Сбросить калибровку
            </Button>
          </div>
        </div>
      )}

      {/* Компас внизу чертежа — ориентация плана */}
      {(floorPlan || drawnPlan) && (
        <div
          className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-1 bg-background/90 backdrop-blur-md p-2 rounded-xl border border-border/50 shadow-lg"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center gap-2 text-[10px] font-bold text-muted-foreground uppercase">
            <Compass className="h-3.5 w-3.5" />
            Ориентация
          </div>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0"
              max="360"
              step="1"
              value={compassAngle}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setCompassAngle(v);
                try {
                  localStorage.setItem("compassAngle", String(v));
                } catch {
                  /* storage unavailable */
                }
              }}
              className="w-24 accent-primary"
            />
            <span className="text-xs font-mono w-10">{compassAngle}°</span>
          </div>
          <div className="relative w-14 h-14">
            <svg viewBox="0 0 100 100" className="w-full h-full">
              <circle cx="50" cy="50" r="45" fill="none" stroke="hsl(var(--border))" strokeWidth="2" />
              <text x="50" y="18" textAnchor="middle" fill="hsl(var(--muted-foreground))" fontSize="12" fontWeight="bold">С</text>
              <text x="50" y="95" textAnchor="middle" fill="hsl(var(--muted-foreground))" fontSize="12">Ю</text>
              <text x="8" y="54" textAnchor="middle" fill="hsl(var(--muted-foreground))" fontSize="12">З</text>
              <text x="92" y="54" textAnchor="middle" fill="hsl(var(--muted-foreground))" fontSize="12">В</text>
              <line
                x1="50"
                y1="50"
                x2={50 + 35 * Math.sin((compassAngle * Math.PI) / 180)}
                y2={50 - 35 * Math.cos((compassAngle * Math.PI) / 180)}
                stroke="hsl(var(--primary))"
                strokeWidth="3"
                strokeLinecap="round"
              />
              <circle cx="50" cy="50" r="4" fill="hsl(var(--primary))" />
            </svg>
          </div>
        </div>
      )}

      <svg
        ref={svgRef}
        viewBox={`0 0 ${viewBox.width} ${viewBox.height}`}
        className="w-full h-full transition-transform duration-75 ease-out"
        preserveAspectRatio="xMidYMid meet"
      >
        <g transform={`translate(${offset.x}, ${offset.y}) scale(${scale})`}>
          {/* Grid background */}
          <defs>
            <marker id="arrowhead-map" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
              <polygon points="0 0, 6 2, 0 4" fill="#22c55e" />
            </marker>
            <pattern id="trajectoryGrid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="hsl(var(--border))" strokeWidth="0.5" opacity="0.2" />
            </pattern>
            <filter id="trajectoryGlow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <defs id="heatmapDefs">
              <linearGradient id="heatmapGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#ff0000" stopOpacity="0.8" />
                <stop offset="25%" stopColor="#ff8000" stopOpacity="0.6" />
                <stop offset="50%" stopColor="#ffff00" stopOpacity="0.4" />
                <stop offset="75%" stopColor="#00ff00" stopOpacity="0.2" />
                <stop offset="100%" stopColor="#0000ff" stopOpacity="0.1" />
              </linearGradient>
              <radialGradient id="heatmapRadial">
                <stop offset="0%" stopColor="#ff0000" stopOpacity="0.8" />
                <stop offset="50%" stopColor="#ff8000" stopOpacity="0.4" />
                <stop offset="100%" stopColor="#ffff00" stopOpacity="0.1" />
              </radialGradient>
            </defs>
          </defs>

          {/* Group for Plan Scale - Centers content and applies scale */}
          <g transform={`translate(${viewBox.width / 2}, ${viewBox.height / 2}) scale(${planScale}) translate(${-viewBox.width / 2}, ${-viewBox.height / 2})`}>

            {/* Floor Plan / Drawn Plan */}
            {(floorPlan || drawnPlan) && (floorPlan ? !floorPlan.includes('application/pdf') : true) ? (
              <>
                {drawnPlan ? (
                  <g>
                    {(drawnPlan as DrawnPlanShape[]).map((s) => s.type === "rect" ?
                      <rect
                        key={s.id}
                        x={Math.min(s.points[0].x, s.points[1].x)}
                        y={Math.min(s.points[0].y, s.points[1].y)}
                        width={Math.abs(s.points[1].x - s.points[0].x)}
                        height={Math.abs(s.points[1].y - s.points[0].y)}
                        fill="rgba(56, 189, 248, 0.2)"
                        stroke="#38bdf8"
                        strokeWidth="2"
                      /> :
                      <line
                        key={s.id}
                        x1={s.points[0].x}
                        y1={s.points[0].y}
                        x2={s.points[1].x}
                        y2={s.points[1].y}
                        stroke="white"
                        strokeWidth="3"
                        strokeLinecap="round"
                      />
                    )}
                  </g>
                ) : floorPlan && (
                  <>
                    <rect x="0" y="0" width={viewBox.width} height={viewBox.height} fill="white" />
                    <image
                      href={floorPlan}
                      x="0"
                      y="0"
                      width={viewBox.width}
                      height={viewBox.height}
                      preserveAspectRatio="xMidYMid meet"
                      opacity="0.9"
                    />
                  </>
                )}
              </>
            ) : (
              <rect width="100%" height="100%" fill="url(#trajectoryGrid)" />
            )}

            {/* Overlay для калибровки масштаба */}
            {(scaleCalibStep === 'A' || scaleCalibStep === 'B') && (
              <rect
                x="0" y="0" width={viewBox.width} height={viewBox.height}
                fill="rgba(0,0,0,0.1)" cursor="crosshair"
                onClick={handleScaleCalibClick}
                onPointerDown={(e) => e.stopPropagation()}
              />
            )}
            {/* Overlay для указания направления */}
            {setDirectionMode && referencePoint && (
              <rect
                x="0" y="0" width={viewBox.width} height={viewBox.height}
                fill="rgba(34, 197, 94, 0.08)" cursor="crosshair"
                onClick={handleDirectionOverlayClick}
                onPointerDown={(e) => e.stopPropagation()}
              />
            )}
            {/* Маркеры точек калибровки */}
            {scalePointA && (
              <g>
                <circle cx={scalePointA.x} cy={scalePointA.y} r="8" fill="none" stroke="#22c55e" strokeWidth="2" />
                <text x={scalePointA.x} y={scalePointA.y - 12} textAnchor="middle" fill="#22c55e" fontSize="10" fontWeight="bold">A</text>
              </g>
            )}
            {scalePointB && (
              <g>
                <circle cx={scalePointB.x} cy={scalePointB.y} r="8" fill="none" stroke="#ef4444" strokeWidth="2" />
                <text x={scalePointB.x} y={scalePointB.y - 12} textAnchor="middle" fill="#ef4444" fontSize="10" fontWeight="bold">B</text>
              </g>
            )}

            {/* Heatmap or Trajectory visualization - INSIDE scaled group */}
            {showHeatmap ? (
              /* Heatmap visualization */
              generateHeatmapData.map((cell, index) => (
                <rect
                  key={`heatmap-${index}`}
                  x={cell.x}
                  y={cell.y}
                  width={cell.width}
                  height={cell.height}
                  fill={`rgba(255, ${Math.round(255 * (1 - cell.intensity))}, 0, ${0.3 + cell.intensity * 0.5})`}
                  opacity={cell.intensity * 0.8}
                />
              ))
            ) : (
              /* Multiple trajectory paths */
              (() => {
                const transformedTrajectories = getTransformedTrajectories;
                return transformedTrajectories.map((data, trajectoryIndex) => {
                  const corrected = correctedPaths.get(trajectoryIndex);
                  // При указанном направлении используем raw траекторию — pathfinding может давать артефакты
                  const pathPoints = usePathfinding && floorPlan && !directionPoint && corrected && corrected.length >= 2
                    ? corrected
                    : data.trajectory;
                  return (
                  <g key={`trajectory-${trajectoryIndex}`}>
                    {/* Trajectory path (с учётом стен при usePathfinding) */}
                    <path
                      d={getPathString(pathPoints)}
                      fill="none"
                      stroke={data.color}
                      strokeWidth={floorPlan ? 5 : 4}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeDasharray={data.ownerName.includes('Тестовые') ? "4 4" : "none"}
                      filter="url(#trajectoryGlow)"
                      className="animate-path-draw"
                      opacity={0.95}
                    />

                    {/* Trajectory points along the path */}
                    {data.trajectory
                      .filter((_, index) => index % 10 === 0)
                      .map((point, index) => (
                        <circle
                          key={`trajectory-point-${trajectoryIndex}-${index}`}
                          cx={point.x}
                          cy={point.y}
                          r="1.5"
                          fill={data.color}
                          opacity="0.7"
                        />
                      ))}
                  </g>
                );});
              })()
            )}

            {/* Start and end points for all trajectories */}
            {(() => {
              const transformedTrajectories = getTransformedTrajectories;
              return transformedTrajectories.map((data, index) => {
                if (data.trajectory.length === 0) return null;
                const endPoint = data.trajectory[data.trajectory.length - 1];

                return (
                  <g key={`points-${index}`}>
                    {/* Start point */}
                    {data.trajectory.length > 0 && (
                      <circle
                        cx={data.trajectory[0].x}
                        cy={data.trajectory[0].y}
                        r="3"
                        fill="hsl(var(--chart-2))"
                        stroke="white"
                        strokeWidth="2"
                      />
                    )}
                    {/* Reference point marker */}
                    {referencePoint && data.trajectory.length > 0 && (() => {
                      const refX = (referencePoint.x / 100) * viewBox.width;
                      const refY = (referencePoint.y / 100) * viewBox.height;
                      const startX = data.trajectory[0].x;
                      const startY = data.trajectory[0].y;
                      const distance = Math.sqrt(Math.pow(startX - refX, 2) + Math.pow(startY - refY, 2));

                      if (distance > 5) {
                        return (
                          <circle
                            cx={refX}
                            cy={refY}
                            r="4"
                            fill="none"
                            stroke="hsl(var(--primary))"
                            strokeWidth="2"
                            strokeDasharray="4 4"
                            opacity="0.7"
                          />
                        );
                      }
                      return null;
                    })()}
                    {/* End point */}
                    {data.trajectory.length > 1 && (
                      <circle
                        cx={endPoint.x}
                        cy={endPoint.y}
                        r="3"
                        fill="hsl(var(--destructive))"
                        stroke="white"
                        strokeWidth="2"
                      />
                    )}
                  </g>
                );
              });
            })()}

            {/* Точка отсчета на карте */}
            {referencePoint && (
              <g transform={`translate(${(referencePoint.x / 100) * viewBox.width}, ${(referencePoint.y / 100) * viewBox.height})`} style={{ pointerEvents: 'none' }}>
                <circle r="12" fill="rgba(239, 68, 68, 0.2)" className="animate-pulse" />
                <circle r="6" fill="rgb(239, 68, 68)" stroke="white" strokeWidth="2" />
                <text y="-15" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Точка отсчета</text>
              </g>
            )}
            {/* Направление движения на карте */}
            {referencePoint && directionPoint && (
              <g style={{ pointerEvents: 'none' }}>
                <line
                  x1={(referencePoint.x / 100) * viewBox.width}
                  y1={(referencePoint.y / 100) * viewBox.height}
                  x2={(directionPoint.x / 100) * viewBox.width}
                  y2={(directionPoint.y / 100) * viewBox.height}
                  stroke="#22c55e"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  markerEnd="url(#arrowhead-map)"
                />
                <circle
                  cx={(directionPoint.x / 100) * viewBox.width}
                  cy={(directionPoint.y / 100) * viewBox.height}
                  r="3"
                  fill="#22c55e"
                  stroke="white"
                  strokeWidth="1"
                />
                <text
                  x={(directionPoint.x / 100) * viewBox.width}
                  y={(directionPoint.y / 100) * viewBox.height - 8}
                  textAnchor="middle"
                  fill="#166534"
                  fontSize="9"
                  fontWeight="600"
                >
                  Направление
                </text>
              </g>
            )}

            {/* Turn points */}
            {(() => {
              const allTransformedTurnPoints = getAllTransformedTurnPoints;
              return allTransformedTurnPoints.map((turn, index) => (
                <g key={index}>
                  <circle
                    cx={turn.position[0]}
                    cy={turn.position[1]}
                    r={hoveredPoint === turn ? 1 : 0.6}
                    fill={turn.turn_type === 'left' ? 'hsl(var(--chart-5))' : 'hsl(var(--chart-4))'}
                    stroke="white"
                    strokeWidth="1.5"
                    className="cursor-pointer transition-all duration-200"
                    onMouseEnter={() => setHoveredPoint(turn)}
                    onMouseLeave={() => setHoveredPoint(null)}
                  />
                  <text
                    x={turn.position[0]}
                    y={turn.position[1] - 12}
                    textAnchor="middle"
                    fill="hsl(var(--foreground))"
                    fontSize="10"
                    fontWeight="bold"
                    className="pointer-events-none"
                  >
                    {index + 1}
                  </text>
                </g>
              ));
            })()}

            {/* Tooltip for turn points */}
            {hoveredPoint && (
              <g>
                <rect
                  x={hoveredPoint.position[0] + 15}
                  y={hoveredPoint.position[1] - 25}
                  width="120"
                  height="40"
                  rx="6"
                  fill="hsl(var(--card))"
                  stroke="hsl(var(--border))"
                  opacity="0.95"
                />
                <text
                  x={hoveredPoint.position[0] + 75}
                  y={hoveredPoint.position[1] - 10}
                  textAnchor="middle"
                  fill="hsl(var(--foreground))"
                  fontSize="12"
                  fontWeight="500"
                >
                  Поворот {(() => {
                    const allTurns = getAllTransformedTurnPoints;
                    const index = allTurns.findIndex(t => t === hoveredPoint);
                    return index !== -1 ? index + 1 : "?";
                  })()}
                </text>
                <text
                  x={hoveredPoint.position[0] + 75}
                  y={hoveredPoint.position[1] + 5}
                  textAnchor="middle"
                  fill="hsl(var(--muted-foreground))"
                  fontSize="10"
                >
                  {hoveredPoint.angle_degrees.toFixed(1)}° {hoveredPoint.turn_type === 'left' ? 'влево' : 'вправо'}
                </text>
              </g>
            )}

          </g> {/* End of Plan Scale Group */}
        </g>
      </svg>
    </div>
  );
};

export default TrajectoryMap;
