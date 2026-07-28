import { ChangeEvent, PointerEvent, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft, CheckCircle2, Download, GitBranch, Loader2,
  MousePointer2, Network, Plus, RefreshCw, Trash2, Upload,
} from "lucide-react";
import Navbar from "@/components/Navbar";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  FloorplanGraphEdge, FloorplanGraphNode, FloorplanTopologyGraph, apiClient,
} from "@/lib/api";

const MAP_ID = "kerama_marazzi_2025";
const PLAN_URL = "/floorplans/kerama-marazzi-2025.png";
const ADMIN_SESSION_KEY = "trackai_admin_authenticated";

type Tool = "select" | "add-node" | "connect";

const length = (points: number[][]) => points.slice(1).reduce((sum, point, index) => {
  const prior = points[index];
  return sum + Math.hypot(point[0] - prior[0], point[1] - prior[1]);
}, 0);

const downloadJson = (graph: FloorplanTopologyGraph) => {
  const blob = new Blob([JSON.stringify(graph, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${graph.map_id}.topology-graph.v1.json`;
  anchor.click();
  URL.revokeObjectURL(url);
};

export default function AdminGraphEditor() {
  const [graph, setGraph] = useState<FloorplanTopologyGraph | null>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [connectFrom, setConnectFrom] = useState<string | null>(null);
  const [minimumEdge, setMinimumEdge] = useState(1.5);
  const [scalePixels, setScalePixels] = useState(8);
  const [showMask, setShowMask] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const svgRef = useRef<SVGSVGElement>(null);
  const authenticated = sessionStorage.getItem(ADMIN_SESSION_KEY) === "true";

  const nodeById = useMemo(
    () => new Map((graph?.nodes || []).map((node) => [node.id, node])),
    [graph],
  );
  const enabledEdges = graph?.edges.filter((edge) => edge.enabled) || [];
  const validation = useMemo(() => {
    if (!graph) return { orphanNodes: 0, invalidEdges: 0 };
    const ids = new Set(graph.nodes.filter((node) => node.enabled).map((node) => node.id));
    const degrees = new Map<string, number>();
    let invalidEdges = 0;
    graph.edges.filter((edge) => edge.enabled).forEach((edge) => {
      if (!ids.has(edge.from) || !ids.has(edge.to) || edge.points.length < 2) invalidEdges += 1;
      degrees.set(edge.from, (degrees.get(edge.from) || 0) + 1);
      degrees.set(edge.to, (degrees.get(edge.to) || 0) + 1);
    });
    return {
      orphanNodes: graph.nodes.filter((node) => node.enabled && !degrees.get(node.id)).length,
      invalidEdges,
    };
  }, [graph]);

  const generate = async () => {
    setBusy(true);
    setError("");
    try {
      setGraph(await apiClient.generateFloorplanTopologyGraph(
        MAP_ID, minimumEdge, scalePixels,
      ));
      setSelectedNode(null);
      setSelectedEdge(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Ошибка генерации графа");
    } finally {
      setBusy(false);
    }
  };

  const pointFromEvent = (event: PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg || !graph) return null;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const matrix = svg.getScreenCTM()?.inverse();
    if (!matrix) return null;
    const transformed = point.matrixTransform(matrix);
    return {
      x: Math.max(0, Math.min(graph.width, transformed.x)),
      y: Math.max(0, Math.min(graph.height, transformed.y)),
    };
  };

  const addNode = (x: number, y: number) => {
    if (!graph) return;
    const next = Math.max(
      0,
      ...graph.nodes.map((node) => Number(node.id.match(/\d+$/)?.[0] || 0)),
    ) + 1;
    const node: FloorplanGraphNode = {
      id: `node_${String(next).padStart(4, "0")}`,
      kind: "manual",
      x: Math.round(x * 1000) / 1000,
      y: Math.round(y * 1000) / 1000,
      degree: 0,
      enabled: true,
    };
    setGraph({ ...graph, nodes: [...graph.nodes, node] });
    setSelectedNode(node.id);
  };

  const connectNodes = (to: string) => {
    if (!graph || !connectFrom || connectFrom === to) {
      setConnectFrom(to);
      return;
    }
    const fromNode = nodeById.get(connectFrom);
    const toNode = nodeById.get(to);
    if (!fromNode || !toNode) return;
    const next = Math.max(
      0,
      ...graph.edges.map((edge) => Number(edge.id.match(/\d+$/)?.[0] || 0)),
    ) + 1;
    const points = [[fromNode.x, fromNode.y], [toNode.x, toNode.y]];
    const edge: FloorplanGraphEdge = {
      id: `edge_${String(next).padStart(5, "0")}`,
      from: fromNode.id,
      to: toNode.id,
      points,
      length_meters: length(points) * graph.meters_per_pixel,
      minimum_width_meters: null,
      median_width_meters: null,
      bidirectional: true,
      enabled: true,
    };
    setGraph({ ...graph, edges: [...graph.edges, edge] });
    setConnectFrom(null);
    setSelectedEdge(edge.id);
  };

  const handleCanvasPointer = (event: PointerEvent<SVGSVGElement>) => {
    if (event.target !== event.currentTarget && tool !== "add-node") return;
    const point = pointFromEvent(event);
    if (point && tool === "add-node") addNode(point.x, point.y);
  };

  const moveSelectedNode = (event: PointerEvent<SVGCircleElement>, nodeId: string) => {
    if (tool !== "select" || event.buttons !== 1 || !graph) return;
    const point = pointFromEvent(event as unknown as PointerEvent<SVGSVGElement>);
    if (!point) return;
    const nodes = graph.nodes.map((node) => (
      node.id === nodeId ? { ...node, x: point.x, y: point.y } : node
    ));
    const moved = nodes.find((node) => node.id === nodeId);
    const edges = graph.edges.map((edge) => {
      if (!moved || (edge.from !== nodeId && edge.to !== nodeId)) return edge;
      const points = edge.points.map((item) => [...item]);
      if (edge.from === nodeId) points[0] = [moved.x, moved.y];
      if (edge.to === nodeId) points[points.length - 1] = [moved.x, moved.y];
      return {
        ...edge,
        points,
        length_meters: length(points) * graph.meters_per_pixel,
      };
    });
    setGraph({ ...graph, nodes, edges });
  };

  const removeSelection = () => {
    if (!graph) return;
    if (selectedEdge) {
      setGraph({
        ...graph,
        edges: graph.edges.filter((edge) => edge.id !== selectedEdge),
      });
      setSelectedEdge(null);
      return;
    }
    if (selectedNode) {
      setGraph({
        ...graph,
        nodes: graph.nodes.filter((node) => node.id !== selectedNode),
        edges: graph.edges.filter(
          (edge) => edge.from !== selectedNode && edge.to !== selectedNode,
        ),
      });
      setSelectedNode(null);
    }
  };

  const importGraph = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text()) as FloorplanTopologyGraph;
      if (
        payload.schema_version !== "trackai.floorplan_graph.v1"
        || !Array.isArray(payload.nodes)
        || !Array.isArray(payload.edges)
      ) throw new Error("Неподдерживаемая схема графа");
      setGraph(payload);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось импортировать JSON");
    } finally {
      event.target.value = "";
    }
  };

  if (!authenticated) {
    return (
      <div className="min-h-screen bg-slate-950 text-white">
        <Navbar />
        <main className="mx-auto max-w-xl px-6 pt-28">
          <Alert className="border-amber-500/30 bg-amber-500/10 text-amber-100">
            <AlertTitle>Нужна авторизация администратора</AlertTitle>
            <AlertDescription className="mt-3">
              Сначала войдите в админ-панель.
              <Button asChild variant="link" className="text-amber-200">
                <Link to="/admin">Перейти к входу</Link>
              </Button>
            </AlertDescription>
          </Alert>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <Navbar />
      <main className="mx-auto max-w-[1800px] px-5 pb-10 pt-24">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <Button asChild variant="ghost" className="-ml-3 mb-1 text-slate-400">
              <Link to="/admin"><ArrowLeft className="mr-2 h-4 w-4" />Админ-панель</Link>
            </Button>
            <h1 className="text-3xl font-semibold">Редактор графа проходов</h1>
            <p className="mt-1 text-slate-400">
              Зелёная support-mask → узлы развилок → corridor-edges → production JSON
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Label className="cursor-pointer">
              <Input className="hidden" type="file" accept=".json" onChange={importGraph} />
              <span className="inline-flex h-10 items-center rounded-md border border-slate-700 px-4 text-sm hover:bg-slate-800">
                <Upload className="mr-2 h-4 w-4" />Импорт JSON
              </span>
            </Label>
            <Button disabled={!graph} onClick={() => graph && downloadJson(graph)}>
              <Download className="mr-2 h-4 w-4" />Скачать production JSON
            </Button>
          </div>
        </div>

        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertTitle>Ошибка</AlertTitle><AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="grid gap-4 xl:grid-cols-[310px_minmax(0,1fr)_300px]">
          <Card className="border-slate-800 bg-slate-900 text-slate-50">
            <CardHeader><CardTitle className="text-lg">Генерация</CardTitle></CardHeader>
            <CardContent className="space-y-5">
              <div className="space-y-2">
                <Label>Минимальная длина ребра, м</Label>
                <Input type="number" min={0.25} max={20} step={0.25}
                  value={minimumEdge}
                  onChange={(event) => setMinimumEdge(Number(event.target.value))}
                  className="border-slate-700 bg-slate-950" />
              </div>
              <div className="space-y-2">
                <Label>Шаг графа, px</Label>
                <Input type="number" min={2} max={32} step={2}
                  value={scalePixels}
                  onChange={(event) => setScalePixels(Number(event.target.value))}
                  className="border-slate-700 bg-slate-950" />
              </div>
              <Button className="w-full" onClick={generate} disabled={busy}>
                {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
                Построить из зелёной маски
              </Button>
              <div className="space-y-2 border-t border-slate-800 pt-4">
                <Label>Инструменты</Label>
                {([
                  ["select", MousePointer2, "Выбор и перемещение"],
                  ["add-node", Plus, "Добавить узел"],
                  ["connect", GitBranch, "Соединить два узла"],
                ] as const).map(([value, Icon, label]) => (
                  <Button key={value} variant={tool === value ? "default" : "outline"}
                    className="w-full justify-start" onClick={() => {
                      setTool(value); setConnectFrom(null);
                    }}>
                    <Icon className="mr-2 h-4 w-4" />{label}
                  </Button>
                ))}
                <Button variant="destructive" className="w-full justify-start"
                  disabled={!selectedNode && !selectedEdge} onClick={removeSelection}>
                  <Trash2 className="mr-2 h-4 w-4" />Удалить выбранное
                </Button>
              </div>
              <Button variant="outline" className="w-full" onClick={() => setShowMask(!showMask)}>
                {showMask ? "Скрыть зелёную маску" : "Показать зелёную маску"}
              </Button>
            </CardContent>
          </Card>

          <Card className="overflow-hidden border-slate-800 bg-slate-900">
            <div className="relative aspect-[5298/3743] min-h-[620px] overflow-auto bg-white">
              {graph ? (
                <svg ref={svgRef} viewBox={`0 0 ${graph.width} ${graph.height}`}
                  className="h-full w-full touch-none" onPointerDown={handleCanvasPointer}>
                  <image href={PLAN_URL} width={graph.width} height={graph.height} />
                  {showMask && (
                    <image href={apiClient.getFloorplanSupportMaskUrl(MAP_ID)}
                      width={graph.width} height={graph.height}
                      opacity={0.22} style={{ filter: "sepia(1) saturate(8) hue-rotate(75deg)" }} />
                  )}
                  {graph.edges.map((edge) => (
                    <polyline key={edge.id}
                      points={edge.points.map((point) => point.join(",")).join(" ")}
                      fill="none"
                      stroke={selectedEdge === edge.id ? "#f59e0b" : edge.enabled ? "#2563eb" : "#64748b"}
                      strokeWidth={selectedEdge === edge.id ? 11 : 6}
                      opacity={edge.enabled ? 0.9 : 0.3}
                      onPointerDown={(event) => {
                        event.stopPropagation(); setSelectedEdge(edge.id); setSelectedNode(null);
                      }} />
                  ))}
                  {graph.nodes.map((node) => (
                    <circle key={node.id} cx={node.x} cy={node.y}
                      r={selectedNode === node.id ? 17 : node.kind === "junction" ? 12 : 9}
                      fill={connectFrom === node.id ? "#f97316" : node.kind === "junction" ? "#ef4444" : "#10b981"}
                      stroke="white" strokeWidth={4}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        if (tool === "connect") connectNodes(node.id);
                        else { setSelectedNode(node.id); setSelectedEdge(null); }
                      }}
                      onPointerMove={(event) => moveSelectedNode(event, node.id)} />
                  ))}
                </svg>
              ) : (
                <div className="flex h-full items-center justify-center text-center text-slate-500">
                  <div><Network className="mx-auto mb-3 h-12 w-12" />
                    Нажмите «Построить из зелёной маски»
                  </div>
                </div>
              )}
            </div>
          </Card>

          <Card className="border-slate-800 bg-slate-900 text-slate-50">
            <CardHeader><CardTitle className="text-lg">Проверка графа</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-lg bg-slate-950 p-3">
                  <div className="text-2xl font-semibold">{graph?.nodes.length || 0}</div>
                  <div className="text-xs text-slate-400">узлов</div>
                </div>
                <div className="rounded-lg bg-slate-950 p-3">
                  <div className="text-2xl font-semibold">{enabledEdges.length}</div>
                  <div className="text-xs text-slate-400">рёбер</div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between"><span>Сиротские узлы</span>
                  <Badge variant={validation.orphanNodes ? "destructive" : "secondary"}>
                    {validation.orphanNodes}
                  </Badge>
                </div>
                <div className="flex justify-between"><span>Некорректные рёбра</span>
                  <Badge variant={validation.invalidEdges ? "destructive" : "secondary"}>
                    {validation.invalidEdges}
                  </Badge>
                </div>
                {graph && !validation.orphanNodes && !validation.invalidEdges && (
                  <div className="flex items-center gap-2 rounded-lg bg-emerald-500/10 p-3 text-sm text-emerald-300">
                    <CheckCircle2 className="h-4 w-4" />JSON структурно валиден
                  </div>
                )}
              </div>
              {(selectedNode || selectedEdge) && (
                <div className="border-t border-slate-800 pt-4 text-sm">
                  <Label>Выбрано</Label>
                  <div className="mt-2 break-all rounded bg-slate-950 p-3 font-mono text-xs">
                    {selectedNode || selectedEdge}
                  </div>
                </div>
              )}
              <p className="text-xs leading-5 text-slate-500">
                Экспорт не меняет production-маску. Файл можно проверить, сохранить в Git
                и затем подключить к map-matching отдельным релизом.
              </p>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
