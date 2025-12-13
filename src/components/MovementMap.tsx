import { useEffect, useRef, useState } from "react";

interface Point {
  x: number;
  y: number;
  time: string;
}

const pathData: Point[] = [
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

const zones = [
  { x: 80, y: 50, width: 180, height: 120, name: "Офис", type: "safe" },
  { x: 300, y: 80, width: 200, height: 150, name: "Цех А", type: "warning" },
  { x: 540, y: 60, width: 180, height: 130, name: "Склад", type: "safe" },
  { x: 100, y: 220, width: 160, height: 140, name: "Столовая", type: "safe" },
  { x: 300, y: 280, width: 220, height: 160, name: "Цех Б", type: "danger" },
  { x: 560, y: 240, width: 180, height: 120, name: "Лаборатория", type: "warning" },
];

const MovementMap = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [animatedPath, setAnimatedPath] = useState("");
  const [hoveredPoint, setHoveredPoint] = useState<Point | null>(null);

  useEffect(() => {
    // Create path string
    const pathString = pathData.reduce((acc, point, i) => {
      return acc + (i === 0 ? `M ${point.x} ${point.y}` : ` L ${point.x} ${point.y}`);
    }, "");
    setAnimatedPath(pathString);
  }, []);

  const zoneColors = {
    safe: { fill: "hsl(var(--analytics-zone-safe) / 0.1)", stroke: "hsl(var(--analytics-zone-safe))" },
    warning: { fill: "hsl(var(--analytics-zone-warning) / 0.1)", stroke: "hsl(var(--analytics-zone-warning))" },
    danger: { fill: "hsl(var(--analytics-zone-danger) / 0.1)", stroke: "hsl(var(--analytics-zone-danger))" },
  };

  return (
    <div className="relative w-full h-full rounded-2xl bg-gradient-card border border-border/50 overflow-hidden">
      <div className="absolute top-4 left-4 z-10">
        <h3 className="text-lg font-semibold mb-2">Карта перемещений</h3>
        <div className="flex gap-3 text-sm">
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
      </div>

      <svg
        ref={svgRef}
        viewBox="0 0 800 480"
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid */}
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

        {/* Zones */}
        {zones.map((zone) => (
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

        {/* Movement path */}
        <path
          d={animatedPath}
          fill="none"
          stroke="hsl(var(--analytics-path))"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter="url(#glow)"
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
                  width="60"
                  height="24"
                  rx="6"
                  fill="hsl(var(--card))"
                  stroke="hsl(var(--border))"
                />
                <text
                  x={point.x + 45}
                  y={point.y + 2}
                  textAnchor="middle"
                  fill="hsl(var(--foreground))"
                  fontSize="12"
                  fontWeight="500"
                >
                  {point.time}
                </text>
              </g>
            )}
          </g>
        ))}

        {/* Current position marker */}
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
      </svg>
    </div>
  );
};

export default MovementMap;
