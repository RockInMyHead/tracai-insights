import { useState, useEffect, useRef } from "react";
import Navbar from "@/components/Navbar";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Activity, Thermometer, Zap, Wind, Layers, Clock, AlertTriangle, CheckCircle2,
  Play, Pause, StopCircle, Download, Printer, Eye, EyeOff, Gauge, Hexagon,
  Droplets, Ruler, Cpu, Fan, Flame, ShieldAlert, Wifi, RefreshCw,
} from "lucide-react";

// ─── Mock data generators ──────────────────────────────────────────────

const rnd = (min: number, max: number) => Math.round((Math.random() * (max - min) + min) * 10) / 10;
const rndInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;

interface TemperaturePoint { time: string; value: number }
interface PowerPoint { time: string; laser1: number; laser2: number }

const buildTempHistory = (base: number, variance: number, count: number): TemperaturePoint[] => {
  const now = Date.now();
  return Array.from({ length: count }, (_, i) => ({
    time: new Date(now - (count - i) * 2000).toLocaleTimeString(),
    value: rnd(base - variance, base + variance),
  }));
};

const buildPowerHistory = (base: number, count: number): PowerPoint[] => {
  const now = Date.now();
  return Array.from({ length: count }, (_, i) => ({
    time: new Date(now - (count - i) * 2000).toLocaleTimeString(),
    laser1: rnd(base - 50, base + 50),
    laser2: rnd(base - 80, base + 30),
  }));
};

// ─── Mini Sparkline (pure SVG) ─────────────────────────────────────────

const SparkLine = ({ data, color = "#3b82f6", height = 40, width = 140 }: {
  data: { value: number }[]; color?: string; height?: number; width?: number;
}) => {
  if (data.length < 2) return null;
  const values = data.map(d => d.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * width},${height - ((v - min) / range) * height * 0.8 - height * 0.1}`);
  return (
    <svg width={width} height={height} className="flex-shrink-0">
      <polyline fill="none" stroke={color} strokeWidth="1.5" points={pts.join(" ")} />
    </svg>
  );
};

// ─── Build Plate Visualization ──────────────────────────────────────────

const BuildPlate = ({ layer, totalLayers }: { layer: number; totalLayers: number }) => {
  const pct = totalLayers > 0 ? layer / totalLayers : 0;
  const filled = Math.floor(pct * 100);
  const rows = 8;
  const cols = 8;
  const filledCount = Math.floor((rows * cols) * pct);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Build Plate {layer}/{totalLayers}</span>
        <span className="font-mono">{filled}%</span>
      </div>
      <div className="grid grid-cols-8 gap-[2px]">
        {Array.from({ length: rows * cols }, (_, i) => (
          <div
            key={i}
            className={`aspect-square rounded-sm transition-all duration-700 ${
              i < filledCount
                ? i > filledCount - 8
                  ? "bg-amber-400 shadow-[0_0_4px_rgba(251,191,36,0.6)]"
                  : "bg-amber-600/70"
                : "bg-muted/30 border border-muted/20"
            }`}
          />
        ))}
      </div>
    </div>
  );
};

// ─── Gauge Component ───────────────────────────────────────────────────

const MiniGauge = ({ value, max, label, unit, color }: {
  value: number; max: number; label: string; unit: string; color: string;
}) => {
  const pct = Math.min(value / max, 1);
  const angle = pct * 180;
  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="64" height="40" viewBox="0 0 64 40" className="overflow-visible">
        <path d="M6 36 A26 26 0 0 1 58 36" fill="none" stroke="hsl(var(--muted))" strokeWidth="4" strokeLinecap="round" />
        <path
          d="M6 36 A26 26 0 0 1 58 36"
          fill="none"
          stroke={color}
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={`${(angle / 180) * 81.68} 81.68`}
          style={{ transition: "stroke-dasharray 0.5s ease" }}
        />
        <line
          x1="32" y1="36" x2={32 + Math.sin((angle - 90) * (Math.PI / 180)) * 22}
          y2={36 - Math.cos((angle - 90) * (Math.PI / 180)) * 22}
          stroke={color} strokeWidth="2" strokeLinecap="round"
          style={{ transition: "all 0.5s ease" }}
        />
        <circle cx="32" cy="36" r="2.5" fill={color} />
      </svg>
      <span className="text-lg font-bold font-mono leading-none">{value}{unit}</span>
      <span className="text-[10px] text-muted-foreground">{label}</span>
    </div>
  );
};

// ─── Main Component ────────────────────────────────────────────────────

export default function Logs() {
  const [jobStatus, setJobStatus] = useState<"printing" | "paused" | "idle" | "complete">("printing");
  const [layer, setLayer] = useState(142);
  const totalLayers = 420;
  const [elapsed, setElapsed] = useState(7340); // seconds
  const [tempBed, setTempBed] = useState(165);
  const [tempChamber, setTempChamber] = useState(38);
  const [laserPower1, setLaserPower1] = useState(340);
  const [laserPower2, setLaserPower2] = useState(370);
  const [gasFlow, setGasFlow] = useState(3.2);
  const [oxygen, setOxygen] = useState(0.08);
  const [recoaterPos, setRecoaterPos] = useState(0);
  const [tempHistory, setTempHistory] = useState<TemperaturePoint[]>(() => buildTempHistory(165, 5, 30));
  const [powerHistory, setPowerHistory] = useState<PowerPoint[]>(() => buildPowerHistory(350, 30));
  const [alerts] = useState([
    { type: "warning" as const, msg: "Oxygen level elevated: 0.08%", time: "2 min ago" },
    { type: "info" as const, msg: "Layer 140 completed. 280 remaining.", time: "5 min ago" },
    { type: "warning" as const, msg: "Recoater speed deviation: +3.2%", time: "12 min ago" },
    { type: "success" as const, msg: "Laser calibration OK — all 4 beams aligned", time: "18 min ago" },
    { type: "info" as const, msg: "Filter replacement due in 14h", time: "22 min ago" },
  ]);

  // Simulate live updates
  useEffect(() => {
    if (jobStatus !== "printing") return;
    const t = setInterval(() => {
      setLayer(prev => Math.min(prev + rndInt(0, 2), totalLayers));
      setElapsed(prev => prev + 3);
      setTempBed(prev => Math.min(180, Math.max(155, prev + rnd(-0.8, 0.8))));
      setTempChamber(prev => Math.min(45, Math.max(30, prev + rnd(-0.4, 0.4))));
      setLaserPower1(prev => Math.min(400, Math.max(280, prev + rnd(-8, 8))));
      setLaserPower2(prev => Math.min(400, Math.max(280, prev + rnd(-10, 10))));
      setGasFlow(prev => Math.min(4.5, Math.max(2.0, prev + rnd(-0.15, 0.15))));
      setOxygen(prev => Math.min(0.15, Math.max(0.01, prev + rnd(-0.01, 0.01))));
      setRecoaterPos(prev => (prev + rnd(0, 2)) % 100);

      setTempHistory(prev => [...prev.slice(-29), { time: new Date().toLocaleTimeString(), value: rnd(162, 170) }]);
      setPowerHistory(prev => [...prev.slice(-29), {
        time: new Date().toLocaleTimeString(),
        laser1: rnd(300, 390),
        laser2: rnd(290, 380),
      }]);
    }, 2000);
    return () => clearInterval(t);
  }, [jobStatus]);

  const elapsedStr = () => {
    const h = Math.floor(elapsed / 3600);
    const m = Math.floor((elapsed % 3600) / 60);
    return `${h}h ${m}m`;
  };
  const eta = () => {
    const remaining = totalLayers - layer;
    const secPerLayer = layer > 0 ? elapsed / layer : 50;
    const etaSec = remaining * secPerLayer;
    const h = Math.floor(etaSec / 3600);
    const m = Math.floor((etaSec % 3600) / 60);
    return `${h}h ${m}m`;
  };
  const progress = totalLayers > 0 ? Math.round((layer / totalLayers) * 100) : 0;

  return (
    <div className="min-h-screen bg-gradient-dark">
      <Navbar />

      <main className="container mx-auto px-4 lg:px-6 pt-20 pb-12">
        {/* ───── Header ───── */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-3">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-primary/15 flex items-center justify-center">
              <Printer className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">SLM® 500Q — Operator Dashboard</h1>
              <p className="text-xs text-muted-foreground">Machine #SLM-007 · FW v4.2.1 · Build Job: IN718_TURBINE_V12</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={jobStatus === "printing" ? "default" : jobStatus === "paused" ? "secondary" : "outline"} className="gap-1.5 text-xs">
              <span className={`h-1.5 w-1.5 rounded-full ${
                jobStatus === "printing" ? "bg-green-400 animate-pulse" :
                jobStatus === "paused" ? "bg-amber-400" : "bg-muted-foreground"
              }`} />
              {jobStatus === "printing" ? "Printing" : jobStatus === "paused" ? "Paused" : "Idle"}
            </Badge>
            <Button size="sm" variant="ghost" className="h-8 w-8 p-0"><RefreshCw className="h-3.5 w-3.5" /></Button>
          </div>
        </div>

        {/* ───── Control Bar ───── */}
        <div className="flex flex-wrap items-center gap-2 mb-6 p-3 rounded-lg bg-background/40 border border-border/40">
          <Button size="sm" variant="ghost" className="gap-1.5 text-xs h-8">
            <Play className="h-3.5 w-3.5 fill-green-500 text-green-500" /> Start
          </Button>
          <Button size="sm" variant="ghost" className="gap-1.5 text-xs h-8">
            <Pause className="h-3.5 w-3.5 text-amber-400" /> Pause
          </Button>
          <Button size="sm" variant="ghost" className="gap-1.5 text-xs h-8 text-destructive">
            <StopCircle className="h-3.5 w-3.5" /> Abort
          </Button>
          <Separator orientation="vertical" className="h-6 mx-1" />
          <span className="text-xs text-muted-foreground">Elapsed: <span className="font-mono text-foreground">{elapsedStr()}</span></span>
          <Separator orientation="vertical" className="h-6 mx-1" />
          <span className="text-xs text-muted-foreground">ETA: <span className="font-mono text-foreground">{eta()}</span></span>
          <Separator orientation="vertical" className="h-6 mx-1" />
          <span className="text-xs text-muted-foreground">Layer: <span className="font-mono text-foreground">{layer}/{totalLayers}</span></span>
        </div>

        {/* ───── Main Grid ───── */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">

          {/* ── Column 1: Temperature / Environment ── */}
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <CardHeader className="pb-2 pt-3 px-4">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Thermometer className="h-3.5 w-3.5 text-red-400" /> Temperature
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0">
                <div className="grid grid-cols-2 gap-3 mb-3">
                  <MiniGauge value={Math.round(tempBed)} max={200} label="Bed" unit="°C" color="#ef4444" />
                  <MiniGauge value={Math.round(tempChamber)} max={60} label="Chamber" unit="°C" color="#f59e0b" />
                </div>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between"><span className="text-muted-foreground">Feed hopper</span><span className="font-mono">32°C</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Recirculator</span><span className="font-mono">28°C</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Coolant</span><span className="font-mono">24°C</span></div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2 pt-3 px-4">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Wind className="h-3.5 w-3.5 text-cyan-400" /> Atmosphere
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0 space-y-2.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-muted-foreground">N₂ Flow</span>
                  <span className="font-mono font-medium">{gasFlow.toFixed(1)} L/min</span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-muted-foreground">O₂</span>
                  <span className={`font-mono font-medium ${oxygen > 0.1 ? "text-red-400" : "text-green-400"}`}>{oxygen.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-muted-foreground">Pressure</span>
                  <span className="font-mono font-medium">{(rnd(980, 1020)).toFixed(0)} mbar</span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-muted-foreground">Humidity</span>
                  <span className="font-mono font-medium">{rnd(0.1, 1.5).toFixed(1)}%</span>
                </div>
                <Progress value={oxygen > 0.1 ? 75 : 25} className="h-1" />
              </CardContent>
            </Card>
          </div>

          {/* ── Column 2: Laser / Process ── */}
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <CardHeader className="pb-2 pt-3 px-4">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Zap className="h-3.5 w-3.5 text-yellow-400" /> Lasers
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0">
                <div className="grid grid-cols-2 gap-2 mb-3">
                  <div className="text-center p-2 rounded-lg bg-background/60">
                    <span className="text-[10px] text-muted-foreground block">Laser 1</span>
                    <span className="text-lg font-bold font-mono text-yellow-400">{Math.round(laserPower1)}</span>
                    <span className="text-[10px] text-muted-foreground"> W</span>
                  </div>
                  <div className="text-center p-2 rounded-lg bg-background/60">
                    <span className="text-[10px] text-muted-foreground block">Laser 2</span>
                    <span className="text-lg font-bold font-mono text-yellow-400">{Math.round(laserPower2)}</span>
                    <span className="text-[10px] text-muted-foreground"> W</span>
                  </div>
                  <div className="text-center p-2 rounded-lg bg-background/60">
                    <span className="text-[10px] text-muted-foreground block">Laser 3</span>
                    <span className="text-lg font-bold font-mono text-yellow-400">{rndInt(280, 390)}</span>
                    <span className="text-[10px] text-muted-foreground"> W</span>
                  </div>
                  <div className="text-center p-2 rounded-lg bg-background/60">
                    <span className="text-[10px] text-muted-foreground block">Laser 4</span>
                    <span className="text-lg font-bold font-mono text-yellow-400">{rndInt(280, 390)}</span>
                    <span className="text-[10px] text-muted-foreground"> W</span>
                  </div>
                </div>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Spot size</span>
                    <span className="font-mono">{rnd(70, 90).toFixed(0)} µm</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Scan speed</span>
                    <span className="font-mono">{rndInt(700, 1200)} mm/s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Hatch spacing</span>
                    <span className="font-mono">{rnd(80, 120).toFixed(0)} µm</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2 pt-3 px-4">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Activity className="h-3.5 w-3.5 text-emerald-400" /> Process
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0 space-y-2.5 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Layer thickness</span>
                  <span className="font-mono">40 µm</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Recoater speed</span>
                  <span className="font-mono">{rnd(80, 200).toFixed(0)} mm/s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Recoater pos</span>
                  <span className="font-mono">{recoaterPos.toFixed(0)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Platform Z</span>
                  <span className="font-mono">{(layer * 0.04).toFixed(2)} mm</span>
                </div>
                <Separator />
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground flex items-center gap-1"><Droplets className="h-3 w-3" /> Material</span>
                  <span className="font-mono">Inconel 718</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Used</span>
                  <span className="font-mono">{rnd(1.2, 2.0).toFixed(1)} kg</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* ── Column 3: Build Plate + Progress ── */}
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <CardHeader className="pb-2 pt-3 px-4">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Layers className="h-3.5 w-3.5 text-amber-400" /> Build Progress
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0">
                <BuildPlate layer={layer} totalLayers={totalLayers} />
                <div className="mt-3 space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="font-mono font-medium">{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2 pt-3 px-4">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Clock className="h-3.5 w-3.5 text-blue-400" /> Timeline
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0 space-y-2 text-xs">
                <div className="flex justify-between"><span className="text-muted-foreground">Started</span><span className="font-mono">2026-06-10 08:42</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Elapsed</span><span className="font-mono font-medium">{elapsedStr()}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">ETA</span><span className="font-mono font-medium">{eta()}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Est. finish</span><span className="font-mono">2026-06-12 ~14:00</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Layers/min</span><span className="font-mono">{layer > 0 ? (layer / (elapsed / 60)).toFixed(2) : "—"}</span></div>
              </CardContent>
            </Card>
          </div>

          {/* ── Column 4: Alerts / Events ── */}
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <CardHeader className="pb-2 pt-3 px-4">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <ShieldAlert className="h-3.5 w-3.5 text-amber-400" /> Alerts & Events
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0 space-y-2">
                {alerts.map((a, i) => (
                  <div key={i} className="flex items-start gap-2.5 text-xs p-2 rounded-lg bg-background/40">
                    {a.type === "warning" && <AlertTriangle className="h-3.5 w-3.5 text-amber-400 mt-0.5 flex-shrink-0" />}
                    {a.type === "info" && <Activity className="h-3.5 w-3.5 text-blue-400 mt-0.5 flex-shrink-0" />}
                    {a.type === "success" && <CheckCircle2 className="h-3.5 w-3.5 text-green-400 mt-0.5 flex-shrink-0" />}
                    <div>
                      <p className="text-foreground/90">{a.msg}</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">{a.time}</p>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2 pt-3 px-4">
                <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                  <Cpu className="h-3.5 w-3.5 text-purple-400" /> System
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0 space-y-1.5 text-xs">
                <div className="flex justify-between"><span className="text-muted-foreground">CPU load</span><span className="font-mono">{rnd(20, 60).toFixed(0)}%</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Memory</span><span className="font-mono">{rndInt(4, 12)} GB</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Uptime</span><span className="font-mono">{rndInt(12, 48)}d {rndInt(0, 23)}h</span></div>
                <Progress value={rnd(25, 65)} className="h-1 mt-1" />
              </CardContent>
            </Card>
          </div>
        </div>

        {/* ───── Charts Row ───── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          <Card>
            <CardHeader className="pb-2 pt-3 px-4">
              <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                <Thermometer className="h-3.5 w-3.5 text-red-400" /> Bed Temperature (last 60s)
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4 pt-0">
              <div className="flex items-end gap-1 h-24">
                {tempHistory.map((pt, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center justify-end gap-0.5 group relative">
                    <div
                      className="w-full rounded-t-sm transition-all duration-500"
                      style={{
                        height: `${((pt.value - 155) / 25) * 100}%`,
                        backgroundColor: pt.value > 175 ? "#ef4444" : pt.value > 168 ? "#f59e0b" : "#3b82f6",
                        opacity: 0.6 + (i / tempHistory.length) * 0.4,
                      }}
                    />
                  </div>
                ))}
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                <span>{tempHistory[0]?.time || ""}</span>
                <span>{tempHistory[tempHistory.length - 1]?.time || ""}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2 pt-3 px-4">
              <CardTitle className="text-xs font-medium flex items-center gap-1.5">
                <Zap className="h-3.5 w-3.5 text-yellow-400" /> Laser Power Timeline
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4 pt-0">
              <div className="flex items-end gap-1 h-24">
                {powerHistory.map((pt, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center justify-end">
                    <div
                      className="w-1/2 rounded-t-sm transition-all duration-500"
                      style={{
                        height: `${((pt.laser1 - 250) / 200) * 100}%`,
                        backgroundColor: "#eab308",
                        opacity: 0.5 + (i / powerHistory.length) * 0.5,
                      }}
                    />
                  </div>
                ))}
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                <span>{powerHistory[0]?.time || ""}</span>
                <span>{powerHistory[powerHistory.length - 1]?.time || ""}</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ───── Job Info Table ───── */}
        <Card>
          <CardHeader className="pb-3 pt-4 px-5">
            <CardTitle className="text-sm font-medium">Job Parameters</CardTitle>
            <CardDescription className="text-xs">IN718_TURBINE_V12 — SLM 500Q</CardDescription>
          </CardHeader>
          <CardContent className="px-5 pb-5 pt-0">
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-xs">
              {[ 
                ["Material", "Inconel 718 (IN718)"],
                ["Layer thickness", "40 µm"],
                ["Laser power", "4 × 350 W (avg)"],
                ["Build volume", "250 × 250 × 300 mm"],
                ["Scan strategy", "Stripes 67° rot"],
                ["Support type", "Block + Tree"],
                ["Gas", "Argon 5.0 (99.999%)"],
                ["Platform preheat", "200°C"],
                ["Recoater type", "Soft rubber blade"],
                ["Filter grade", "HEPA H13"],
                ["Job priority", "High"],
                ["Operator", "A. Smith"],
              ].map(([label, value]) => (
                <div key={label} className="p-2 rounded-lg bg-background/40 border border-border/20">
                  <p className="text-muted-foreground mb-0.5">{label}</p>
                  <p className="font-medium">{value}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* ───── Footer ───── */}
        <div className="mt-6 text-center text-[10px] text-muted-foreground/50">
          SLM® 500Q · Machine #SLM-007 · FW v4.2.1 · Last calibration: 2026-06-08 · All values simulated
        </div>
      </main>
    </div>
  );
}
