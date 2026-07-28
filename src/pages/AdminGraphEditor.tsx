import { ChangeEvent, PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft, CheckCircle2, Download, GitBranch, Loader2,
  MousePointer2, Plus, RefreshCw, RotateCcw, Save, Trash2, Upload,
  ZoomIn, ZoomOut,
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
const PLAN_WIDTH = 5298;
const PLAN_HEIGHT = 3743;
const ADMIN_SESSION_KEY = "trackai_admin_authenticated";

type Tool = "select" | "add-node" | "connect";
type DragState = {
  nodeId: string;
  pointerId: number;
  offsetX: number;
  offsetY: number;
};
type CompetingBranches = {
  ambiguous: boolean;
  divergence_anchor: number;
  divergence_point: number[];
  selected: number[][];
  alternative: number[][];
};

const findCompetingBranches = (value: unknown): CompetingBranches | null => {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const candidate = record.competing_branches;
  if (candidate && typeof candidate === "object") {
    const branch = candidate as Partial<CompetingBranches>;
    if (
      Array.isArray(branch.selected)
      && Array.isArray(branch.alternative)
      && Array.isArray(branch.divergence_point)
    ) return branch as CompetingBranches;
  }
  for (const child of Object.values(record)) {
    const found = findCompetingBranches(child);
    if (found) return found;
  }
  return null;
};

const length = (points: number[][]) => points.slice(1).reduce((sum, point, index) => {
  const prior = points[index];
  return sum + Math.hypot(point[0] - prior[0], point[1] - prior[1]);
}, 0);

const nextNumericId = (items: { id: string }[]) => Math.max(
  0,
  ...items.map((item) => Number(item.id.match(/\d+$/)?.[0] || 0)),
) + 1;

const withNodeDegrees = (graph: FloorplanTopologyGraph): FloorplanTopologyGraph => {
  const degrees = new Map<string, number>();
  graph.edges.filter((edge) => edge.enabled).forEach((edge) => {
    degrees.set(edge.from, (degrees.get(edge.from) || 0) + 1);
    degrees.set(edge.to, (degrees.get(edge.to) || 0) + 1);
  });
  return {
    ...graph,
    nodes: graph.nodes.map((node) => ({ ...node, degree: degrees.get(node.id) || 0 })),
    validation: {
      ...graph.validation,
      node_count: graph.nodes.length,
      edge_count: graph.edges.length,
      disabled_nodes: graph.nodes.filter((node) => !node.enabled).length,
      disabled_edges: graph.edges.filter((edge) => !edge.enabled).length,
    },
  };
};

const closestEdgeProjection = (
  x: number,
  y: number,
  edges: FloorplanGraphEdge[],
) => {
  let closest: {
    edge: FloorplanGraphEdge;
    segmentIndex: number;
    x: number;
    y: number;
    distance: number;
  } | null = null;
  edges.filter((edge) => edge.enabled).forEach((edge) => {
    edge.points.slice(1).forEach((to, segmentIndex) => {
      const from = edge.points[segmentIndex];
      const dx = to[0] - from[0];
      const dy = to[1] - from[1];
      const denominator = dx * dx + dy * dy;
      const ratio = denominator
        ? Math.max(0, Math.min(1, ((x - from[0]) * dx + (y - from[1]) * dy) / denominator))
        : 0;
      const projectedX = from[0] + ratio * dx;
      const projectedY = from[1] + ratio * dy;
      const distance = Math.hypot(x - projectedX, y - projectedY);
      if (!closest || distance < closest.distance) {
        closest = {
          edge,
          segmentIndex,
          x: projectedX,
          y: projectedY,
          distance,
        };
      }
    });
  });
  return closest;
};

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
  const [mapZoom, setMapZoom] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [savedMessage, setSavedMessage] = useState("");
  const [competingBranches, setCompetingBranches] = useState<CompetingBranches | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const authenticated = sessionStorage.getItem(ADMIN_SESSION_KEY) === "true";

  const loadProductionGraph = async () => {
    setBusy(true);
    setError("");
    setSavedMessage("");
    try {
      setGraph(await apiClient.getProductionFloorplanTopologyGraph(MAP_ID));
      setSelectedNode(null);
      setSelectedEdge(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Ошибка загрузки production-графа");
    } finally {
      setBusy(false);
    }
  };

  const saveProductionGraph = async () => {
    if (!graph) return;
    if (validation.orphanNodes || validation.invalidEdges) {
      setError("Исправьте сиротские узлы и некорректные рёбра перед сохранением");
      return;
    }
    setBusy(true);
    setError("");
    setSavedMessage("");
    try {
      const saved = await apiClient.saveProductionFloorplanTopologyGraph(
        MAP_ID,
        graph,
      );
      setGraph(saved);
      setSavedMessage("Граф сохранён в production и кеш matcher обновлён");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Ошибка сохранения production-графа");
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (authenticated) void loadProductionGraph();
    // Authentication is session-scoped and fixed for this page lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authenticated]);

  const nodeById = useMemo(
    () => new Map((graph?.nodes || []).map((node) => [node.id, node])),
    [graph],
  );
  const visibleNodes = graph?.nodes.filter((node) => node.kind !== "attachment") || [];
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

  const pointFromClient = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg || !graph) return null;
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
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
    const next = nextNumericId(graph.nodes);
    const node: FloorplanGraphNode = {
      id: `node_${String(next).padStart(4, "0")}`,
      kind: "manual",
      x: Math.round(x * 1000) / 1000,
      y: Math.round(y * 1000) / 1000,
      degree: 0,
      enabled: true,
    };
    const projection = closestEdgeProjection(x, y, graph.edges);
    if (!projection || projection.distance > 500) {
      setGraph(withNodeDegrees({ ...graph, nodes: [...graph.nodes, node] }));
      setError("Узел создан, но ближайшее ребро находится слишком далеко. Переместите узел ближе и соедините его вручную.");
    } else {
      const sourceEdge = projection.edge;
      const leftPoints = [
        ...sourceEdge.points.slice(0, projection.segmentIndex + 1),
        [node.x, node.y],
      ];
      const rightPoints = [
        [node.x, node.y],
        ...sourceEdge.points.slice(projection.segmentIndex + 1),
      ];
      const edges = graph.edges.map((edge) => edge.id === sourceEdge.id ? {
        ...edge,
        to: node.id,
        points: leftPoints,
        length_meters: length(leftPoints) * graph.meters_per_pixel,
      } : edge);
      edges.push({
        ...sourceEdge,
        id: `edge_${String(nextNumericId(edges)).padStart(5, "0")}`,
        from: node.id,
        points: rightPoints,
        length_meters: length(rightPoints) * graph.meters_per_pixel,
      });
      setGraph(withNodeDegrees({ ...graph, nodes: [...graph.nodes, node], edges }));
      setError("");
    }
    setSelectedNode(node.id);
    setSelectedEdge(null);
    setTool("select");
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
    setGraph(withNodeDegrees({ ...graph, edges: [...graph.edges, edge] }));
    setConnectFrom(null);
    setSelectedEdge(edge.id);
  };

  const handleCanvasPointer = (event: PointerEvent<SVGSVGElement>) => {
    if (event.target !== event.currentTarget && tool !== "add-node") return;
    const point = pointFromClient(event.clientX, event.clientY);
    if (point && tool === "add-node") addNode(point.x, point.y);
  };

  const beginNodeDrag = (event: PointerEvent<SVGCircleElement>, node: FloorplanGraphNode) => {
    if (tool !== "select") return;
    const point = pointFromClient(event.clientX, event.clientY);
    if (!point) return;
    dragRef.current = {
      nodeId: node.id,
      pointerId: event.pointerId,
      offsetX: node.x - point.x,
      offsetY: node.y - point.y,
    };
    svgRef.current?.setPointerCapture(event.pointerId);
  };

  const moveDraggedNode = (event: PointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const point = pointFromClient(event.clientX, event.clientY);
    if (!point) return;
    setGraph((current) => {
      if (!current) return current;
      const x = Math.max(0, Math.min(current.width, point.x + drag.offsetX));
      const y = Math.max(0, Math.min(current.height, point.y + drag.offsetY));
      const nodes = current.nodes.map((node) => (
        node.id === drag.nodeId ? { ...node, x, y } : node
      ));
      const edges = current.edges.map((edge) => {
        if (edge.from !== drag.nodeId && edge.to !== drag.nodeId) return edge;
        const points = edge.points.map((item) => [...item]);
        if (edge.from === drag.nodeId) points[0] = [x, y];
        if (edge.to === drag.nodeId) points[points.length - 1] = [x, y];
        return {
          ...edge,
          points,
          length_meters: length(points) * current.meters_per_pixel,
        };
      });
      return { ...current, nodes, edges };
    });
  };

  const endNodeDrag = (event: PointerEvent<SVGSVGElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    if (svgRef.current?.hasPointerCapture(event.pointerId)) {
      svgRef.current.releasePointerCapture(event.pointerId);
    }
  };

  const removeSelection = () => {
    if (!graph) return;
    if (selectedEdge) {
      setGraph(withNodeDegrees({
        ...graph,
        edges: graph.edges.filter((edge) => edge.id !== selectedEdge),
      }));
      setSelectedEdge(null);
      return;
    }
    if (selectedNode) {
      setGraph(withNodeDegrees({
        ...graph,
        nodes: graph.nodes.filter((node) => node.id !== selectedNode),
        edges: graph.edges.filter(
          (edge) => edge.from !== selectedNode && edge.to !== selectedNode,
        ),
      }));
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

  const importDiagnostics = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text()) as unknown;
      const branches = findCompetingBranches(payload);
      if (!branches) throw new Error("В файле нет диагностики competing_branches");
      setCompetingBranches(branches);
      setMapZoom((value) => Math.max(value, 1.5));
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось импортировать диагностику");
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
            <h1 className="text-3xl font-semibold">Чертёж</h1>
            <p className="mt-1 text-slate-400">
              План Kerama Marazzi → узлы проходов → corridor-edges → production JSON
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              onClick={saveProductionGraph}
              disabled={!graph || busy || Boolean(validation.orphanNodes || validation.invalidEdges)}
              className="bg-emerald-600 hover:bg-emerald-500"
            >
              {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4" />}
              Сохранить в production
            </Button>
            <Button variant="outline" onClick={loadProductionGraph} disabled={busy}>
              {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
              Production-граф
            </Button>
            <Label className="cursor-pointer">
              <Input className="hidden" type="file" accept=".json" onChange={importDiagnostics} />
              <span className="inline-flex h-10 items-center rounded-md border border-amber-500/50 px-4 text-sm text-amber-200 hover:bg-amber-500/10">
                <Upload className="mr-2 h-4 w-4" />Диагностика маршрута
              </span>
            </Label>
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
        {savedMessage && (
          <Alert className="mb-4 border-emerald-500/30 bg-emerald-500/10 text-emerald-100">
            <CheckCircle2 className="h-4 w-4" />
            <AlertTitle>Сохранено</AlertTitle>
            <AlertDescription>{savedMessage}</AlertDescription>
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
            </CardContent>
          </Card>

          <Card className="overflow-hidden border-slate-800 bg-slate-900">
            <div className="flex h-12 items-center justify-end gap-2 border-b border-slate-800 px-3">
              <Button variant="outline" size="icon" disabled={mapZoom <= 1}
                aria-label="Уменьшить масштаб"
                onClick={() => setMapZoom((value) => Math.max(1, value - 0.25))}>
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Badge variant="secondary" className="min-w-16 justify-center">
                {Math.round(mapZoom * 100)}%
              </Badge>
              <Button variant="outline" size="icon" disabled={mapZoom >= 4}
                aria-label="Увеличить масштаб"
                onClick={() => setMapZoom((value) => Math.min(4, value + 0.25))}>
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="icon" aria-label="Сбросить масштаб"
                disabled={mapZoom === 1} onClick={() => setMapZoom(1)}>
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
            <div className="relative aspect-[5298/3743] min-h-[620px] overflow-auto bg-white">
              <svg ref={svgRef}
                viewBox={`0 0 ${graph?.width || PLAN_WIDTH} ${graph?.height || PLAN_HEIGHT}`}
                className="block touch-none"
                style={{
                  width: `${mapZoom * 100}%`,
                  height: "auto",
                  minHeight: `${mapZoom * 620}px`,
                }}
                onPointerDown={handleCanvasPointer}
                onPointerMove={moveDraggedNode}
                onPointerUp={endNodeDrag}
                onPointerCancel={endNodeDrag}>
                <image href={PLAN_URL}
                  width={graph?.width || PLAN_WIDTH}
                  height={graph?.height || PLAN_HEIGHT} />
                {(graph?.edges || []).map((edge) => (
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
                {competingBranches && (
                  <>
                    <polyline
                      points={competingBranches.selected.map((point) => point.join(",")).join(" ")}
                      fill="none" stroke="#22c55e" strokeWidth={18} opacity={0.9}
                      strokeLinecap="round" strokeLinejoin="round" />
                    <polyline
                      points={competingBranches.alternative.map((point) => point.join(",")).join(" ")}
                      fill="none" stroke="#f97316" strokeWidth={18} opacity={0.9}
                      strokeDasharray="30 18" strokeLinecap="round" strokeLinejoin="round" />
                    <circle
                      cx={competingBranches.divergence_point[0]}
                      cy={competingBranches.divergence_point[1]}
                      r={28} fill="#ef4444" stroke="white" strokeWidth={8} />
                  </>
                )}
                {visibleNodes.map((node) => (
                    <circle key={node.id} cx={node.x} cy={node.y}
                      r={selectedNode === node.id ? 17 : ["junction", "turn"].includes(node.kind) ? 12 : 9}
                      fill={connectFrom === node.id ? "#f97316" : ["junction", "turn"].includes(node.kind) ? "#ef4444" : "#10b981"}
                      stroke="white" strokeWidth={4}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        if (tool === "connect") connectNodes(node.id);
                        else {
                          setSelectedNode(node.id);
                          setSelectedEdge(null);
                          beginNodeDrag(event, node);
                        }
                      }} />
                ))}
              </svg>
              {!graph && (
                <div className="pointer-events-none absolute inset-x-0 top-4 flex justify-center">
                  <div className="rounded-lg border border-slate-300 bg-white/90 px-4 py-2 text-sm text-slate-600 shadow">
                    План загружен. Нажмите «Построить из зелёной маски», чтобы показать граф.
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
                  <div className="text-2xl font-semibold">{visibleNodes.length}</div>
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
                {graph?.production_validation && (
                  <div className={`rounded-lg border p-3 text-sm ${
                    graph.production_validation.file_sha256_matches
                    && graph.production_validation.geometry_sha256
                      === graph.production_validation.embedded_geometry_sha256
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
                      : "border-red-500/30 bg-red-500/10 text-red-200"
                  }`}>
                    <div className="font-medium">
                      {graph.production_validation.file_sha256_matches
                      && graph.production_validation.geometry_sha256
                        === graph.production_validation.embedded_geometry_sha256
                        ? "Редактор показывает точный production-граф"
                        : "SHA или геометрия production-графа не совпадает"}
                    </div>
                    <div className="mt-2 break-all font-mono text-[10px] opacity-75">
                      file: {graph.production_validation.file_sha256}<br />
                      geometry: {graph.production_validation.geometry_sha256}
                    </div>
                  </div>
                )}
                {competingBranches && (
                  <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-100">
                    <div className="font-medium">Неоднозначная развилка</div>
                    <div className="mt-1 text-xs text-amber-200/80">
                      Зелёная линия — выбранная ветка, оранжевая — ближайший конкурент.
                      Красная точка — место расхождения, anchor {competingBranches.divergence_anchor}.
                    </div>
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
