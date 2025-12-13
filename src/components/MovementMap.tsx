import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload, FileText, X, ZoomIn, ZoomOut } from "lucide-react";
import { toast } from "sonner";

interface Point {
  x: number;
  y: number;
  time?: string;
}

interface MovementMapProps {
  externalPath?: Point[];
  externalMapUrl?: string;
}

const defaultPathData: Point[] = [
  { x: 50, y: 400, time: "08:00" },
  { x: 120, y: 350, time: "08:15" },
  { x: 200, y: 300, time: "08:30" },
  { x: 280, y: 280, time: "09:00" },
  { x: 350, y: 200, time: "09:30" },
  { x: 420, y: 180, time: "10:00" },
  { x: 500, y: 150, time: "10:30" },
  { x: 580, y: 120, time: "11:00" },
  { x: 650, y: 100, time: "11:30" },
  { x: 720, y: 130, time: "12:00" },
];

const defaultZones = [
  { x: 80, y: 50, width: 180, height: 120, name: "Офис", type: "safe" },
  { x: 300, y: 80, width: 200, height: 150, name: "Цех А", type: "warning" },
  { x: 540, y: 60, width: 180, height: 130, name: "Склад", type: "safe" },
  { x: 100, y: 220, width: 160, height: 140, name: "Столовая", type: "safe" },
  { x: 300, y: 280, width: 220, height: 160, name: "Цех Б", type: "danger" },
  { x: 560, y: 240, width: 180, height: 120, name: "Лаборатория", type: "warning" },
];

const MovementMap = ({ externalPath, externalMapUrl }: MovementMapProps) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const txtInputRef = useRef<HTMLInputElement>(null);
  const [animatedPath, setAnimatedPath] = useState("");
  const [hoveredPoint, setHoveredPoint] = useState<Point | null>(null);
  const [mapImage, setMapImage] = useState<string | null>(externalMapUrl || null);
  const [pathData, setPathData] = useState<Point[]>(externalPath || defaultPathData);
  const [scale, setScale] = useState(1);
  const [viewBox, setViewBox] = useState({ width: 800, height: 480 });

  useEffect(() => {
    if (externalPath) {
      setPathData(externalPath);
    }
  }, [externalPath]);

  useEffect(() => {
    if (externalMapUrl) {
      setMapImage(externalMapUrl);
    }
  }, [externalMapUrl]);

  useEffect(() => {
    const pathString = pathData.reduce((acc, point, i) => {
      return acc + (i === 0 ? `M ${point.x} ${point.y}` : ` L ${point.x} ${point.y}`);
    }, "");
    setAnimatedPath(pathString);
  }, [pathData]);

  const handleMapUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        toast.error("Пожалуйста, загрузите изображение");
        return;
      }
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setViewBox({ width: img.width, height: img.height });
          setMapImage(event.target?.result as string);
          toast.success("Карта загружена");
        };
        img.src = event.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  };

  const handleTxtUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith(".txt")) {
        toast.error("Пожалуйста, загрузите .txt файл");
        return;
      }
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        const points = parseTxtCoordinates(text);
        if (points.length > 0) {
          setPathData(points);
          toast.success(`Загружено ${points.length} точек маршрута`);
        } else {
          toast.error("Не удалось прочитать координаты из файла");
        }
      };
      reader.readAsText(file);
    }
  };

  const parseTxtCoordinates = (text: string): Point[] => {
    const lines = text.trim().split("\n");
    const points: Point[] = [];
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      
      // Support formats: "x y", "x,y", "x;y", "x y time"
      const parts = trimmed.split(/[\s,;]+/);
      if (parts.length >= 2) {
        const x = parseFloat(parts[0]);
        const y = parseFloat(parts[1]);
        const time = parts[2] || undefined;
        
        if (!isNaN(x) && !isNaN(y)) {
          points.push({ x, y, time });
        }
      }
    }
    
    return points;
  };

  const clearMap = () => {
    setMapImage(null);
    setPathData(defaultPathData);
    setViewBox({ width: 800, height: 480 });
    setScale(1);
    toast.info("Карта сброшена");
  };

  const zoneColors = {
    safe: { fill: "hsl(var(--analytics-zone-safe) / 0.1)", stroke: "hsl(var(--analytics-zone-safe))" },
    warning: { fill: "hsl(var(--analytics-zone-warning) / 0.1)", stroke: "hsl(var(--analytics-zone-warning))" },
    danger: { fill: "hsl(var(--analytics-zone-danger) / 0.1)", stroke: "hsl(var(--analytics-zone-danger))" },
  };

  return (
    <div className="relative w-full h-full rounded-2xl bg-gradient-card border border-border/50 overflow-hidden">
      {/* Controls */}
      <div className="absolute top-4 left-4 z-10">
        <h3 className="text-lg font-semibold mb-2">Карта перемещений</h3>
        <div className="flex gap-3 text-sm mb-3">
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-full bg-analytics-safe" />
            <span className="text-muted-foreground">Безопасно</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-full bg-analytics-warning" />
            <span className="text-muted-foreground">Внимание</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-full bg-analytics-danger" />
            <span className="text-muted-foreground">Опасно</span>
          </div>
        </div>
        
        {/* Upload buttons */}
        <div className="flex gap-2 flex-wrap">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleMapUpload}
          />
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="h-3.5 w-3.5" />
            Загрузить карту
          </Button>
          
          <input
            ref={txtInputRef}
            type="file"
            accept=".txt"
            className="hidden"
            onChange={handleTxtUpload}
          />
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => txtInputRef.current?.click()}
          >
            <FileText className="h-3.5 w-3.5" />
            Загрузить маршрут
          </Button>
          
          {(mapImage || pathData !== defaultPathData) && (
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5 text-xs"
              onClick={clearMap}
            >
              <X className="h-3.5 w-3.5" />
              Сбросить
            </Button>
          )}
        </div>
      </div>

      {/* Zoom controls */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-1">
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8"
          onClick={() => setScale(Math.min(scale + 0.2, 3))}
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8"
          onClick={() => setScale(Math.max(scale - 0.2, 0.5))}
        >
          <ZoomOut className="h-4 w-4" />
        </Button>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${viewBox.width} ${viewBox.height}`}
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
        style={{ transform: `scale(${scale})`, transformOrigin: "center" }}
      >
        {/* Background image or grid */}
        {mapImage ? (
          <image
            href={mapImage}
            x="0"
            y="0"
            width={viewBox.width}
            height={viewBox.height}
            preserveAspectRatio="xMidYMid meet"
          />
        ) : (
          <>
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="hsl(var(--border))" strokeWidth="0.5" opacity="0.3" />
              </pattern>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
            
            {/* Default zones only shown when no custom map */}
            {defaultZones.map((zone) => (
              <g key={zone.name}>
                <rect
                  x={zone.x}
                  y={zone.y}
                  width={zone.width}
                  height={zone.height}
                  rx="12"
                  fill={zoneColors[zone.type as keyof typeof zoneColors].fill}
                  stroke={zoneColors[zone.type as keyof typeof zoneColors].stroke}
                  strokeWidth="2"
                  strokeDasharray="8 4"
                />
                <text
                  x={zone.x + zone.width / 2}
                  y={zone.y + zone.height / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="hsl(var(--muted-foreground))"
                  fontSize="14"
                  fontWeight="500"
                >
                  {zone.name}
                </text>
              </g>
            ))}
          </>
        )}

        {/* Glow filter for path */}
        <defs>
          <filter id="pathGlow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Movement path */}
        <path
          d={animatedPath}
          fill="none"
          stroke="hsl(var(--analytics-path))"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter="url(#pathGlow)"
          strokeDasharray="1000"
          className="animate-path-draw"
        />

        {/* Path points */}
        {pathData.map((point, i) => (
          <g key={i}>
            <circle
              cx={point.x}
              cy={point.y}
              r={hoveredPoint === point ? 10 : 6}
              fill="hsl(var(--background))"
              stroke="hsl(var(--analytics-path))"
              strokeWidth="2"
              className="cursor-pointer transition-all duration-200"
              onMouseEnter={() => setHoveredPoint(point)}
              onMouseLeave={() => setHoveredPoint(null)}
            />
            {hoveredPoint === point && (
              <g>
                <rect
                  x={point.x + 15}
                  y={point.y - 15}
                  width={point.time ? 60 : 80}
                  height="24"
                  rx="6"
                  fill="hsl(var(--card))"
                  stroke="hsl(var(--border))"
                />
                <text
                  x={point.x + (point.time ? 45 : 55)}
                  y={point.y + 2}
                  textAnchor="middle"
                  fill="hsl(var(--foreground))"
                  fontSize="12"
                  fontWeight="500"
                >
                  {point.time || `${Math.round(point.x)}, ${Math.round(point.y)}`}
                </text>
              </g>
            )}
          </g>
        ))}

        {/* Current position marker */}
        {pathData.length > 0 && (
          <>
            <circle
              cx={pathData[pathData.length - 1].x}
              cy={pathData[pathData.length - 1].y}
              r="12"
              fill="hsl(var(--analytics-marker))"
              opacity="0.3"
              className="animate-ping"
            />
            <circle
              cx={pathData[pathData.length - 1].x}
              cy={pathData[pathData.length - 1].y}
              r="8"
              fill="hsl(var(--analytics-marker))"
            />
          </>
        )}
      </svg>

      {/* File format hint */}
      {!mapImage && (
        <div className="absolute bottom-4 left-4 text-xs text-muted-foreground max-w-xs">
          Формат .txt: каждая строка — координаты x y (опционально время)
          <br />
          Пример: <code className="text-primary">100 200 08:30</code>
        </div>
      )}
    </div>
  );
};

export default MovementMap;
