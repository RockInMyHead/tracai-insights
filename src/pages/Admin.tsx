import {
    FormEvent,
    MouseEvent,
    WheelEvent,
    PointerEvent,
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import Navbar from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { apiClient, TrackingTask } from "@/lib/api";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
    MousePointer2,
    Image as ImageIcon,
    Video,
    User,
    Clock,
    CheckCircle2,
    AlertCircle,
    Loader2,
    RefreshCw,
    Gauge,
    ZoomIn,
    ZoomOut,
    RotateCcw,
    Eye,
    EyeOff,
    Trash2,
    Download,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";

const ADMIN_LOGIN = "adm!n";
const ADMIN_PASSWORD = "bdyltrcrjvgfybz123";
const ADMIN_SESSION_KEY = "trackai_admin_authenticated";
const ADMIN_READ_TASK_IDS_KEY = "trackai_admin_read_task_ids";

function getTaskSource(task: TrackingTask) {
    const source = String(task.map_context?.client_source || "").toLowerCase();
    return source === "desktop" ? "desktop" : "web";
}

function TaskSourceBadge({ task }: { task: TrackingTask }) {
    const source = getTaskSource(task);
    const isDesktop = source === "desktop";

    return (
        <Badge
            variant="outline"
            className={`text-[9px] px-1.5 py-0 uppercase tracking-wider shrink-0 ${
                isDesktop
                    ? "border-cyan-400/50 text-cyan-300 bg-cyan-500/10"
                    : "border-violet-400/50 text-violet-300 bg-violet-500/10"
            }`}
            title={isDesktop ? "Загрузка из десктопного приложения" : "Загрузка с веб-сайта"}
        >
            {isDesktop ? "Desktop" : "Web"}
        </Badge>
    );
}

function loadReadTaskIdsFromStorage(): Set<string> {
    try {
        const raw = localStorage.getItem(ADMIN_READ_TASK_IDS_KEY);
        if (!raw) return new Set();
        const parsed = JSON.parse(raw) as unknown;
        if (!Array.isArray(parsed)) return new Set();
        return new Set(parsed.filter((id): id is string => typeof id === "string"));
    } catch {
        return new Set();
    }
}

function persistReadTaskIds(ids: Set<string>) {
    try {
        localStorage.setItem(ADMIN_READ_TASK_IDS_KEY, JSON.stringify([...ids]));
    } catch {
        /* ignore */
    }
}

const PLAYBACK_RATES = ["0.25", "0.5", "0.75", "1", "1.25", "1.5", "2"] as const;

/** Как в TrajectoryMap: система координат плана 800×600, цвет и толщина линии */
const PLAN_VB_W = 800;
const PLAN_VB_H = 600;
const TRAJECTORY_STROKE = "#3b82f6";
const TRAJECTORY_START_FILL = "#22c55e";

function migrateLegacyPlanPoints(points: Array<{ x: number; y: number }>): Array<{ x: number; y: number }> {
    if (points.length === 0) return points;
    const maxX = Math.max(...points.map((p) => p.x));
    const maxY = Math.max(...points.map((p) => p.y));
    if (maxX <= 100.5 && maxY <= 100.5) {
        return points.map((p) => ({
            x: (p.x / 100) * PLAN_VB_W,
            y: (p.y / 100) * PLAN_VB_H,
        }));
    }
    return points;
}

/** Несколько подпутей в одном `d`: без отрезков между концом одного видео и началом следующего (объединение по сотруднику). */
function planPathDFromSegments(segments: Array<Array<{ x: number; y: number }>>): string {
    const chunks: string[] = [];
    for (const seg of segments) {
        if (seg.length < 2) continue;
        chunks.push(
            seg.reduce(
                (acc, pt, i) => acc + (i === 0 ? `M ${pt.x} ${pt.y}` : ` L ${pt.x} ${pt.y}`),
                ""
            )
        );
    }
    return chunks.join(" ");
}

function firstPlanPoint(segments: Array<Array<{ x: number; y: number }>>): { x: number; y: number } | null {
    for (const s of segments) {
        if (s.length > 0) return s[0];
    }
    return null;
}

function lastPlanPoint(segments: Array<Array<{ x: number; y: number }>>): { x: number; y: number } | null {
    for (let i = segments.length - 1; i >= 0; i--) {
        const s = segments[i];
        if (s.length > 0) return s[s.length - 1];
    }
    return null;
}

const PLAN_ZOOM_MIN = 0.5;
const PLAN_ZOOM_MAX = 12;

const TRAJECTORY_STROKE_DEFAULT = 5;
const TRAJECTORY_STROKE_MIN = 0.75;
const TRAJECTORY_STROKE_MAX = 12;

function clientToRootSvg(
    svg: SVGSVGElement,
    clientX: number,
    clientY: number
): { x: number; y: number } | null {
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = svg.getScreenCTM()?.inverse();
    if (!ctm) return null;
    const q = pt.matrixTransform(ctm);
    return { x: q.x, y: q.y };
}

/**
 * Экран → координаты плана 800×600.
 * Важно: getScreenCTM() на корневом <svg> не учитывает transform на вложенных <g> (pan/zoom),
 * поэтому используем CTM слоя контента — он включает полную цепочку до экрана.
 */
function clientToPlanCoords(
    svg: SVGSVGElement,
    planLayer: SVGGElement | null,
    clientX: number,
    clientY: number,
    planZoom: number,
    planPan: { x: number; y: number }
): { x: number; y: number } | null {
    const owner = planLayer?.ownerSVGElement;
    if (planLayer && owner) {
        const pt = owner.createSVGPoint();
        pt.x = clientX;
        pt.y = clientY;
        const ctm = planLayer.getScreenCTM();
        if (ctm) {
            try {
                const q = pt.matrixTransform(ctm.inverse());
                return { x: q.x, y: q.y };
            } catch {
                /* inverse может не существовать при вырожденной матрице */
            }
        }
    }
    const R = clientToRootSvg(svg, clientX, clientY);
    if (!R) return null;
    const Qx = R.x - planPan.x;
    const Qy = R.y - planPan.y;
    const cx = PLAN_VB_W / 2;
    const cy = PLAN_VB_H / 2;
    const s = planZoom;
    return {
        x: cx + (Qx - cx) / s,
        y: cy + (Qy - cy) / s,
    };
}

/** % reference_point → SVG 800×600 с учётом letterbox растра (как у пользователя в натуральном viewBox). */
function referencePercentToPlanSvg(
    pct: { x: number; y: number },
    natural: { w: number; h: number }
): { x: number; y: number } {
    if (natural.w > 0 && natural.h > 0) {
        const s = Math.min(PLAN_VB_W / natural.w, PLAN_VB_H / natural.h);
        const cw = natural.w * s;
        const ch = natural.h * s;
        const ox = (PLAN_VB_W - cw) / 2;
        const oy = (PLAN_VB_H - ch) / 2;
        return {
            x: ox + (pct.x / 100) * cw,
            y: oy + (pct.y / 100) * ch,
        };
    }
    return {
        x: (pct.x / 100) * PLAN_VB_W,
        y: (pct.y / 100) * PLAN_VB_H,
    };
}

const Admin = () => {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [error, setError] = useState("");
    const [tasks, setTasks] = useState<TrackingTask[]>([]);
    const [isLoadingData, setIsLoadingData] = useState(false);
    const [dataError, setDataError] = useState("");
    const [selectedTaskId, setSelectedTaskId] = useState("");
    /** Раздельные цепочки точек (одна задача = один сегмент; объединение по сотруднику = несколько). */
    const [manualSegments, setManualSegments] = useState<Array<Array<{ x: number; y: number }>>>([]);
    const manualPointsFlat = useMemo(() => manualSegments.flat(), [manualSegments]);
    const adminTrajectoryPathD = useMemo(() => planPathDFromSegments(manualSegments), [manualSegments]);
    const adminFirstPlanPt = useMemo(() => firstPlanPoint(manualSegments), [manualSegments]);
    const adminLastPlanPt = useMemo(() => lastPlanPoint(manualSegments), [manualSegments]);
    const [playbackRate, setPlaybackRate] = useState("1");
    const [videoPreviewRetry, setVideoPreviewRetry] = useState(0);
    const [isSavingTrajectory, setIsSavingTrajectory] = useState(false);
    const [trajectoryNotice, setTrajectoryNotice] = useState("");
    /** Масштаб чертежа (центр 800×600), как planScale в TrajectoryMap */
    const [planZoom, setPlanZoom] = useState(1);
    /** Сдвиг чертежа по X/Y в координатах viewBox */
    const [planPan, setPlanPan] = useState({ x: 0, y: 0 });
    const [isPlanPanning, setIsPlanPanning] = useState(false);
    const isPlanPanningRef = useRef(false);
    const lastPanClientRef = useRef({ x: 0, y: 0 });
    const suppressPlanClickRef = useRef(false);
    /** Толщина линии траектории в единицах SVG (не влияет на сохранённые координаты) */
    const [trajectoryStrokeWidth, setTrajectoryStrokeWidth] = useState(TRAJECTORY_STROKE_DEFAULT);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const planContainerRef = useRef<SVGSVGElement | null>(null);
    /** Слой с изображением и траекторией (система координат 800×600); для корректного hit-test при pan/zoom */
    const planLayerRef = useRef<SVGGElement | null>(null);
    /** Натуральный размер растрового плана — для совпадения ref-point с TrajectoryMap */
    const [floorPlanNaturalSize, setFloorPlanNaturalSize] = useState({ w: 0, h: 0 });
    /** Прочитанные задачи (только в этом браузере) */
    const [readTaskIds, setReadTaskIds] = useState<Set<string>>(loadReadTaskIdsFromStorage);
    /** По умолчанию только текущая задача; склейка по сотруднику — только по явному согласию админа */
    const [mergeEmployeeTrajectories, setMergeEmployeeTrajectories] = useState(false);
    /** ID задач, чьи ручные траектории не участвуют в объединении (только режим объединения) */
    const [excludedMergeTaskIds, setExcludedMergeTaskIds] = useState<string[]>([]);
    const [deletingTaskId, setDeletingTaskId] = useState<string | null>(null);
    const [deleteTaskError, setDeleteTaskError] = useState("");
    const [isClearingDatabase, setIsClearingDatabase] = useState(false);

    useEffect(() => {
        const savedSession = sessionStorage.getItem(ADMIN_SESSION_KEY);
        setIsAuthenticated(savedSession === "true");
    }, []);

    const handleLogin = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (username === ADMIN_LOGIN && password === ADMIN_PASSWORD) {
            sessionStorage.setItem(ADMIN_SESSION_KEY, "true");
            setIsAuthenticated(true);
            setError("");
            return;
        }

        setError("Неверный логин или пароль");
    };

    const handleLogout = () => {
        sessionStorage.removeItem(ADMIN_SESSION_KEY);
        setIsAuthenticated(false);
        setPassword("");
        setError("");
        setDataError("");
        setSelectedTaskId("");
        setManualSegments([]);
        setPlanZoom(1);
        setPlanPan({ x: 0, y: 0 });
        setTrajectoryStrokeWidth(TRAJECTORY_STROKE_DEFAULT);
        setTrajectoryNotice("");
    };

    const loadAdminData = async () => {
        setIsLoadingData(true);
        setDataError("");
        setDeleteTaskError("");
        try {
            const tasksResponse = await apiClient.getAdminTasks();
            setTasks(tasksResponse || []);
        } catch (loadError) {
            const message =
                loadError instanceof Error
                    ? loadError.message
                    : "Не удалось загрузить данные задач";
            setDataError(message);
        } finally {
            setIsLoadingData(false);
        }
    };

    useEffect(() => {
        if (!isAuthenticated) return;
        loadAdminData();
    }, [isAuthenticated]);

    const selectedTask = useMemo(
        () => tasks.find((t) => t.id === selectedTaskId) ?? null,
        [tasks, selectedTaskId]
    );

    const unreadTaskCount = useMemo(
        () => tasks.filter((t) => !readTaskIds.has(t.id)).length,
        [tasks, readTaskIds]
    );

    /** Задачи того же сотрудника до выбранной включительно — для настройки объединения */
    const mergeTimelineForSelection = useMemo(() => {
        if (!selectedTaskId || !selectedTask?.employee_name) return [];
        const emp = selectedTask.employee_name;
        const list = tasks
            .filter((t) => t.employee_name === emp)
            .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
        const idx = list.findIndex((t) => t.id === selectedTaskId);
        if (idx < 0) return [];
        return list.slice(0, idx + 1);
    }, [tasks, selectedTaskId, selectedTask?.employee_name]);

    const markTaskRead = (taskId: string) => {
        if (!taskId) return;
        setReadTaskIds((prev) => {
            if (prev.has(taskId)) return prev;
            const next = new Set(prev);
            next.add(taskId);
            persistReadTaskIds(next);
            return next;
        });
    };

    const toggleTaskReadFlag = (taskId: string, e: MouseEvent<HTMLButtonElement>) => {
        e.stopPropagation();
        setReadTaskIds((prev) => {
            const next = new Set(prev);
            if (next.has(taskId)) next.delete(taskId);
            else next.add(taskId);
            persistReadTaskIds(next);
            return next;
        });
    };

    const selectedVideoUrl = selectedTask
        ? `${apiClient.getUploadedVideoPreviewUrl(selectedTask.id)}?v=${videoPreviewRetry}`
        : "";
    const selectedOriginalVideoUrl = selectedTask
        ? apiClient.getUploadedVideoUrl(selectedTask.id)
        : "";

    const floorPlanData = selectedTask?.map_context?.floor_plan_data;

    useEffect(() => {
        if (!floorPlanData || floorPlanData.includes("application/pdf")) {
            setFloorPlanNaturalSize({ w: 0, h: 0 });
            return;
        }
        const img = new Image();
        img.onload = () => {
            if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                setFloorPlanNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
            }
        };
        img.onerror = () => setFloorPlanNaturalSize({ w: 0, h: 0 });
        img.src = floorPlanData;
    }, [floorPlanData]);

    const adminReferenceSvg = useMemo(() => {
        const ref = selectedTask?.map_context?.reference_point;
        if (!ref) return null;
        const hasRasterPlan =
            !!floorPlanData && !floorPlanData.includes("application/pdf");
        return referencePercentToPlanSvg(ref, hasRasterPlan ? floorPlanNaturalSize : { w: 0, h: 0 });
    }, [selectedTask, floorPlanData, floorPlanNaturalSize]);

    useEffect(() => {
        const v = videoRef.current;
        if (!v) return;
        const rate = Number.parseFloat(playbackRate);
        if (!Number.isFinite(rate) || rate <= 0) return;
        v.playbackRate = rate;
    }, [playbackRate, selectedVideoUrl, selectedTaskId]);

    const handlePlanCanvasClick = (event: MouseEvent<SVGSVGElement>) => {
        if (event.button !== 0) return;
        if (event.shiftKey) return;
        if (suppressPlanClickRef.current) {
            suppressPlanClickRef.current = false;
            return;
        }
        const svg = planContainerRef.current;
        if (!svg) return;
        const p = clientToPlanCoords(svg, planLayerRef.current, event.clientX, event.clientY, planZoom, planPan);
        if (!p) return;
        setManualSegments((prev) => {
            const pt = { x: Number(p.x.toFixed(2)), y: Number(p.y.toFixed(2)) };
            if (prev.length === 0) return [[pt]];
            const next = prev.map((s) => [...s]);
            next[next.length - 1] = [...next[next.length - 1], pt];
            return next;
        });
        setTrajectoryNotice("");
    };

    const handlePlanPointerDown = (e: PointerEvent<SVGSVGElement>) => {
        const wantPan = e.button === 1 || e.button === 2 || (e.button === 0 && e.shiftKey);
        if (!wantPan) return;
        e.preventDefault();
        e.stopPropagation();
        isPlanPanningRef.current = true;
        setIsPlanPanning(true);
        lastPanClientRef.current = { x: e.clientX, y: e.clientY };
        e.currentTarget.setPointerCapture(e.pointerId);
    };

    const handlePlanPointerMove = (e: PointerEvent<SVGSVGElement>) => {
        if (!isPlanPanningRef.current) return;
        e.preventDefault();
        suppressPlanClickRef.current = true;
        const svg = planContainerRef.current;
        if (!svg) return;
        const cur = clientToRootSvg(svg, e.clientX, e.clientY);
        const last = clientToRootSvg(svg, lastPanClientRef.current.x, lastPanClientRef.current.y);
        if (!cur || !last) return;
        lastPanClientRef.current = { x: e.clientX, y: e.clientY };
        setPlanPan((prev) => ({
            x: prev.x + (cur.x - last.x),
            y: prev.y + (cur.y - last.y),
        }));
    };

    const handlePlanPointerUp = (e: PointerEvent<SVGSVGElement>) => {
        if (!isPlanPanningRef.current) return;
        e.preventDefault();
        isPlanPanningRef.current = false;
        setIsPlanPanning(false);
        try {
            e.currentTarget.releasePointerCapture(e.pointerId);
        } catch {
            /* already released */
        }
    };

    const handlePlanWheel = (e: WheelEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        const delta = e.deltaY > 0 ? -0.12 : 0.12;
        setPlanZoom((z) => {
            const next = Math.round((z + delta) * 100) / 100;
            return Math.min(PLAN_ZOOM_MAX, Math.max(PLAN_ZOOM_MIN, next));
        });
    };

    const handleUndoPoint = () => {
        setManualSegments((prev) => {
            if (prev.length === 0) return [];
            const next = prev.map((s) => [...s]);
            let i = next.length - 1;
            while (i >= 0 && next[i].length === 0) i -= 1;
            if (i < 0) return [];
            next[i] = next[i].slice(0, -1);
            while (next.length > 0 && next[next.length - 1].length === 0) {
                next.pop();
            }
            return next;
        });
    };

    const handleClearPoints = () => {
        setManualSegments([]);
    };

    /** Только GET ручной траектории; не затрагивает загрузку/конвертацию видео на сервере. */
    const fetchManualTrajectoryForTask = useCallback(
        async (taskId: string, mergePrevious: boolean, excludedFromMerge: string[]) => {
            if (!taskId) return;
            try {
                if (mergePrevious) {
                    const selected = tasks.find((t) => t.id === taskId) ?? null;
                    if (selected && selected.employee_name) {
                        const emp = selected.employee_name;
                        const tasksForEmployee = tasks
                            .filter((t) => t.employee_name === emp)
                            .sort(
                                (a, b) =>
                                    new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
                            );

                        const uptoIndex = tasksForEmployee.findIndex((t) => t.id === taskId);
                        if (uptoIndex >= 0) {
                            const sliceIds = tasksForEmployee.slice(0, uptoIndex + 1).map((t) => t.id);
                            const idsToLoad = sliceIds.filter((id) => !excludedFromMerge.includes(id));
                            const excludedCount = sliceIds.length - idsToLoad.length;
                            const responses = await Promise.all(
                                idsToLoad.map((id) => apiClient.getManualTrajectory(id).catch((_) => null))
                            );
                            const segments: Array<Array<{ x: number; y: number }>> = [];
                            for (const res of responses) {
                                if (res && res.exists && Array.isArray(res.trajectory)) {
                                    const raw = res.trajectory
                                        .filter((p) => Array.isArray(p) && p.length >= 2)
                                        .map((p) => ({ x: Number(p[0]), y: Number(p[1]) }));
                                    if (raw.length > 0) {
                                        segments.push(migrateLegacyPlanPoints(raw));
                                    }
                                }
                            }
                            if (segments.length > 0) {
                                setManualSegments(segments);
                                setTrajectoryNotice(
                                    excludedCount > 0
                                        ? `Загружена объединённая ручная траектория из ${idsToLoad.length} задач (${excludedCount} исключено)`
                                        : `Загружена объединённая ручная траектория из ${idsToLoad.length} задач`
                                );
                                return;
                            }
                        }
                    }
                }

                const manual = await apiClient.getManualTrajectory(taskId);
                if (manual.exists && Array.isArray(manual.trajectory)) {
                    const raw = manual.trajectory
                        .filter((p) => Array.isArray(p) && p.length >= 2)
                        .map((p) => ({ x: Number(p[0]), y: Number(p[1]) }));
                    setManualSegments([migrateLegacyPlanPoints(raw)]);
                    setTrajectoryNotice("Загружена ранее сохраненная ручная траектория");
                }
            } catch (e) {
                console.warn("Manual trajectory error:", e);
            }
        },
        [tasks]
    );

    const toggleMergeExclusion = (taskIdToToggle: string) => {
        setExcludedMergeTaskIds((prev) => {
            const next = prev.includes(taskIdToToggle)
                ? prev.filter((x) => x !== taskIdToToggle)
                : [...prev, taskIdToToggle];
            if (selectedTaskId && mergeEmployeeTrajectories) {
                void fetchManualTrajectoryForTask(selectedTaskId, true, next);
            }
            return next;
        });
    };

    const handleDeleteTaskFromList = async (taskId: string, e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDeleteTaskError("");
        if (
            !window.confirm(
                "Удалить эту задачу и файл видео с сервера? Ручная траектория и результат анализа для этого видео будут удалены. Действие необратимо."
            )
        ) {
            return;
        }
        setDeletingTaskId(taskId);
        try {
            await apiClient.deleteAdminTask(taskId);
            setReadTaskIds((prev) => {
                if (!prev.has(taskId)) return prev;
                const next = new Set(prev);
                next.delete(taskId);
                persistReadTaskIds(next);
                return next;
            });
            if (selectedTaskId === taskId) {
                setSelectedTaskId("");
                setManualSegments([]);
                setTrajectoryNotice("");
                setExcludedMergeTaskIds([]);
            }
            await loadAdminData();
        } catch (err) {
            const msg = err instanceof Error ? err.message : "Не удалось удалить";
            setDeleteTaskError(msg);
        } finally {
            setDeletingTaskId(null);
        }
    };

    const handleClearDatabase = async () => {
        setDeleteTaskError("");
        if (
            !window.confirm(
                "Удалить ВСЕ записи из базы данных SQLite (таблицы задач и планов)? Файлы видео на сервере и ручные траектории в JSON не удаляются. Действие необратимо."
            )
        ) {
            return;
        }
        if (!window.confirm("Подтвердите полную очистку БД.")) {
            return;
        }
        setIsClearingDatabase(true);
        try {
            await apiClient.clearAdminDatabase();
            setTasks([]);
            setSelectedTaskId("");
            setManualSegments([]);
            setTrajectoryNotice("");
            setExcludedMergeTaskIds([]);
            setReadTaskIds(new Set());
            persistReadTaskIds(new Set());
        } catch (e) {
            const msg = e instanceof Error ? e.message : "Не удалось очистить базу";
            setDeleteTaskError(msg);
        } finally {
            setIsClearingDatabase(false);
        }
    };

    const handleSelectTask = async (taskId: string) => {
        setSelectedTaskId(taskId);
        markTaskRead(taskId);
        setPlaybackRate("1");
        setPlanZoom(1);
        setPlanPan({ x: 0, y: 0 });
        setVideoPreviewRetry(0);
        setTrajectoryStrokeWidth(TRAJECTORY_STROKE_DEFAULT);
        setManualSegments([]);
        setTrajectoryNotice("");
        setExcludedMergeTaskIds([]);
        if (!taskId) return;

        try {
            const fullTask = await apiClient.getAdminTask(taskId);
            setTasks((prev) => prev.map((t) => (t.id === taskId ? fullTask : t)));
        } catch (e) {
            console.warn("Failed to load full task details:", e);
        }

        await fetchManualTrajectoryForTask(taskId, mergeEmployeeTrajectories, []);
    };

    const handleSaveManualTrajectory = async () => {
        if (!selectedTaskId) {
            setTrajectoryNotice("Сначала выберите задачу");
            return;
        }
        if (manualPointsFlat.length < 2) {
            setTrajectoryNotice("Добавьте минимум 2 точки траектории");
            return;
        }
        setIsSavingTrajectory(true);
        setTrajectoryNotice("");
        try {
            const trajectory = manualPointsFlat.map((p) => [p.x, p.y, 0]);
            await apiClient.saveManualTrajectory(selectedTaskId, trajectory);
            setTrajectoryNotice("Ручная траектория сохранена. Пользователь получит ее после нажатия \"Далее\".");
        } catch (e) {
            const message = e instanceof Error ? e.message : "Не удалось сохранить траекторию";
            setTrajectoryNotice(message);
        } finally {
            setIsSavingTrajectory(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-950 text-slate-50 font-sans selection:bg-primary/30">
            <Navbar />
            <main className="container mx-auto px-6 pt-24 pb-12">
                <div className="mx-auto max-w-7xl">
                    {!isAuthenticated ? (
                        <Card className="mx-auto max-w-md bg-slate-900 border-slate-800 shadow-2xl">
                            <CardHeader className="text-center">
                                <CardTitle className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
                                    TrackAI Admin
                                </CardTitle>
                                <CardDescription className="text-slate-400">Панель управления анализом траекторий</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <form onSubmit={handleLogin} className="space-y-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="admin-login">Логин</Label>
                                        <Input
                                            id="admin-login"
                                            value={username}
                                            onChange={(event) => setUsername(event.target.value)}
                                            placeholder="Введите логин"
                                            autoComplete="username"
                                            className="bg-slate-800 border-slate-700 focus:ring-blue-500"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="admin-password">Пароль</Label>
                                        <Input
                                            id="admin-password"
                                            type="password"
                                            value={password}
                                            onChange={(event) => setPassword(event.target.value)}
                                            placeholder="Введите пароль"
                                            autoComplete="current-password"
                                            className="bg-slate-800 border-slate-700 focus:ring-blue-500"
                                        />
                                    </div>
                                    {error && (
                                        <div className="flex items-center gap-2 p-3 rounded-md bg-destructive/10 text-destructive text-sm animate-in fade-in zoom-in duration-200">
                                            <AlertCircle className="h-4 w-4" />
                                            {error}
                                        </div>
                                    )}
                                    <Button type="submit" className="w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 border-none shadow-lg transition-all hover:scale-[1.02] active:scale-[0.98]">
                                        Войти
                                    </Button>
                                </form>
                            </CardContent>
                        </Card>
                    ) : (
                        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between bg-slate-900/50 backdrop-blur-xl p-6 rounded-2xl border border-slate-800/50">
                                <div>
                                    <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">Админ-панель</h1>
                                    <p className="text-slate-400">Управление задачами трассировки в реальном времени</p>
                                </div>
                                <div className="flex flex-wrap gap-3">
                                    <Button 
                                        variant="outline" 
                                        onClick={loadAdminData} 
                                        disabled={isLoadingData}
                                        className="bg-slate-800/50 border-slate-700 hover:bg-slate-700 transition-all gap-2"
                                    >
                                        {isLoadingData ? <Loader2 className="h-4 w-4 animate-spin text-blue-400" /> : <RefreshCw className="h-4 w-4" />}
                                        Обновить данные
                                    </Button>
                                    <Button
                                        type="button"
                                        variant="destructive"
                                        onClick={handleClearDatabase}
                                        disabled={isClearingDatabase || isLoadingData}
                                        className="gap-2 font-semibold tracking-wide"
                                    >
                                        {isClearingDatabase ? (
                                            <Loader2 className="h-4 w-4 animate-spin" />
                                        ) : (
                                            <Trash2 className="h-4 w-4" />
                                        )}
                                        ОЧИСТИТЬ
                                    </Button>
                                    <Button 
                                        variant="ghost" 
                                        onClick={handleLogout}
                                        className="text-slate-400 hover:text-white hover:bg-white/5"
                                    >
                                        Выйти
                                    </Button>
                                </div>
                            </div>

                            {dataError && (
                                <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive flex items-center gap-3">
                                    <AlertCircle className="h-5 w-5" />
                                    <p className="text-sm font-medium">{dataError}</p>
                                </div>
                            )}

                            <div className="grid gap-8 lg:grid-cols-4">
                                {/* Tasks Sidebar */}
                                <Card className="lg:col-span-1 bg-slate-900/50 border-slate-800/50 overflow-hidden">
                                    <CardHeader className="bg-slate-800/30">
                                        <CardTitle className="text-lg">Задачи (запросы)</CardTitle>
                                        <CardDescription className="space-y-1">
                                            <span>Выберите одного из активных пользователей</span>
                                            {tasks.length > 0 && (
                                                <span className="block text-xs">
                                                    {unreadTaskCount > 0 ? (
                                                        <span className="text-sky-400/90">
                                                            Непрочитанных: {unreadTaskCount}
                                                        </span>
                                                    ) : (
                                                        <span className="text-slate-500">Все задачи просмотрены</span>
                                                    )}
                                                </span>
                                            )}
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent className="p-0">
                                        {deleteTaskError && (
                                            <div className="px-4 py-2 text-[11px] text-red-400 border-b border-red-500/20 bg-red-950/30">
                                                {deleteTaskError}
                                            </div>
                                        )}
                                        <div className="flex flex-col max-h-[600px] overflow-y-auto divide-y divide-slate-800/50 custom-scrollbar">
                                            {tasks.length === 0 ? (
                                                <div className="p-8 text-center text-slate-500">
                                                    <Clock className="h-10 w-10 mx-auto mb-3 opacity-20" />
                                                    <p className="text-sm">Активных задач пока нет</p>
                                                </div>
                                            ) : (
                                                tasks.map((task) => {
                                                    const isRead = readTaskIds.has(task.id);
                                                    return (
                                                    <div
                                                        key={task.id}
                                                        className={`flex gap-0.5 p-2 pl-0 transition-all hover:bg-white/5 ${
                                                            selectedTaskId === task.id ? "bg-blue-600/10 border-l-2 border-l-blue-500" : ""
                                                        } ${!isRead ? "bg-sky-950/25" : ""}`}
                                                    >
                                                        <button
                                                            type="button"
                                                            onClick={() => handleSelectTask(task.id)}
                                                            className="flex min-w-0 flex-1 flex-col gap-1.5 rounded-r-md px-3 py-2 text-left active:bg-white/10"
                                                        >
                                                        <div className="flex items-center justify-between gap-2">
                                                            <div className="flex items-center gap-2 min-w-0">
                                                                {!isRead && (
                                                                    <span
                                                                        className="h-2 w-2 shrink-0 rounded-full bg-sky-400 shadow-[0_0_6px_rgba(56,189,248,0.7)]"
                                                                        title="Непрочитано"
                                                                    />
                                                                )}
                                                                <User className="h-3.5 w-3.5 text-blue-400 shrink-0" />
                                                                <span
                                                                    className={`text-sm truncate ${
                                                                        isRead ? "font-medium text-slate-300" : "font-semibold text-white"
                                                                    }`}
                                                                >
                                                                    {task.employee_name || "Инкогнито"}
                                                                </span>
                                                            </div>
                                                            <div className="flex shrink-0 items-center gap-1">
                                                                <TaskSourceBadge task={task} />
                                                                <Badge
                                                                    variant="outline"
                                                                    className={`text-[10px] px-1.5 py-0 uppercase tracking-wider shrink-0 ${
                                                                        task.status === "completed" ? "border-emerald-500/50 text-emerald-400 bg-emerald-500/5" :
                                                                        task.status === "error" ? "border-red-500/50 text-red-400 bg-red-500/5" :
                                                                        "border-blue-500/50 text-blue-400 bg-blue-500/5 animate-pulse"
                                                                    }`}
                                                                >
                                                                    {task.status === "queued" ? "В очереди" :
                                                                     task.status === "processing" ? "Обработка" :
                                                                     task.status === "completed" ? "Готово" :
                                                                     task.status === "error" ? "Ошибка" : task.status}
                                                                </Badge>
                                                            </div>
                                                        </div>
                                                        <div
                                                            className={`flex items-center gap-1.5 text-[11px] truncate ${
                                                                isRead ? "text-slate-500" : "text-slate-400"
                                                            }`}
                                                        >
                                                            <Video className="h-3 w-3 shrink-0" />
                                                            <span className="truncate">{task.original_filename}</span>
                                                        </div>
                                                        <div
                                                            className={`text-[10px] mt-0.5 ${
                                                                isRead ? "text-slate-600" : "text-slate-500"
                                                            }`}
                                                        >
                                                            {new Date(task.created_at).toLocaleString('ru-RU')}
                                                        </div>
                                                        </button>
                                                        <div className="flex shrink-0 flex-col items-center gap-0.5 pt-0.5">
                                                            <button
                                                                type="button"
                                                                onClick={(e) => toggleTaskReadFlag(task.id, e)}
                                                                className="rounded-md p-1.5 text-slate-500 transition-colors hover:bg-white/10 hover:text-slate-200"
                                                                title={
                                                                    isRead
                                                                        ? "Пометить как непрочитанное"
                                                                        : "Пометить как прочитанное"
                                                                }
                                                            >
                                                                {isRead ? (
                                                                    <EyeOff className="h-3.5 w-3.5" />
                                                                ) : (
                                                                    <Eye className="h-3.5 w-3.5 text-sky-400" />
                                                                )}
                                                            </button>
                                                            <Button
                                                                type="button"
                                                                variant="ghost"
                                                                size="icon"
                                                                disabled={deletingTaskId === task.id}
                                                                className="h-8 w-8 text-slate-500 hover:text-red-400 hover:bg-red-950/40"
                                                                title="Удалить задачу и видео с сервера"
                                                                onClick={(e) => handleDeleteTaskFromList(task.id, e)}
                                                            >
                                                                {deletingTaskId === task.id ? (
                                                                    <Loader2 className="h-4 w-4 animate-spin" />
                                                                ) : (
                                                                    <Trash2 className="h-4 w-4" />
                                                                )}
                                                            </Button>
                                                        </div>
                                                    </div>
                                                    );
                                                })
                                            )}
                                        </div>
                                    </CardContent>
                                </Card>

                                {/* Drawing Area */}
                                <Card className="lg:col-span-3 bg-slate-900 border-slate-800 shadow-xl overflow-hidden flex flex-col">
                                    {selectedTask ? (
                                        <>
                                            <CardHeader className="bg-slate-800/30 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-slate-800/50">
                                                <div className="flex flex-col">
                                                    <CardTitle className="text-xl flex items-center gap-3 flex-wrap">
                                                        <Badge className="bg-blue-600 text-white hover:bg-blue-600">Активная сессия</Badge>
                                                        {selectedTask.employee_name || "Без имени"}
                                                    </CardTitle>
                                                    <CardDescription className="flex items-center gap-2 mt-1">
                                                        <Video className="h-3 w-3" /> {selectedTask.original_filename}
                                                    </CardDescription>
                                                    <p className="text-[11px] text-slate-500 mt-2 max-w-xl">
                                                        Слева — просмотр записи (ускорение без влияния на траекторию). Справа — кликайте по чертежу, чтобы отметить путь.
                                                    </p>
                                                    <div className="flex items-start gap-2.5 mt-3 max-w-xl rounded-lg border border-slate-700/60 bg-slate-900/50 px-3 py-2">
                                                        <Checkbox
                                                            id="admin-merge-employee-traj"
                                                            className="mt-0.5 border-slate-500 data-[state=checked]:bg-sky-600 data-[state=checked]:border-sky-600"
                                                            checked={mergeEmployeeTrajectories}
                                                            onCheckedChange={(v) => {
                                                                const next = v === true;
                                                                setMergeEmployeeTrajectories(next);
                                                                if (selectedTaskId) {
                                                                    void fetchManualTrajectoryForTask(
                                                                        selectedTaskId,
                                                                        next,
                                                                        excludedMergeTaskIds
                                                                    );
                                                                }
                                                            }}
                                                        />
                                                        <Label
                                                            htmlFor="admin-merge-employee-traj"
                                                            className="text-[11px] text-slate-400 font-normal leading-snug cursor-pointer"
                                                        >
                                                            Объединить ручную траекторию со всеми предыдущими задачами этого сотрудника (по дате). Иначе загружается только выбранное видео.
                                                        </Label>
                                                    </div>
                                                    {mergeEmployeeTrajectories &&
                                                        mergeTimelineForSelection.length > 0 &&
                                                        selectedTask?.employee_name && (
                                                            <div className="mt-3 max-w-xl rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 space-y-2">
                                                                <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                                                                    Участие видео в объединении
                                                                </p>
                                                                <p className="text-[10px] text-slate-500 leading-snug">
                                                                    Снимите галочку, чтобы не подмешивать ручную траекторию этого ролика в общую линию.
                                                                </p>
                                                                <div className="max-h-[min(200px,28vh)] overflow-y-auto space-y-1.5 pr-1">
                                                                    {mergeTimelineForSelection.map((t) => {
                                                                        const included = !excludedMergeTaskIds.includes(t.id);
                                                                        return (
                                                                            <div
                                                                                key={t.id}
                                                                                className="flex items-start gap-2 rounded-md border border-slate-800/80 bg-slate-900/80 px-2 py-1.5"
                                                                            >
                                                                                <Checkbox
                                                                                    id={`merge-inc-${t.id}`}
                                                                                    className="mt-0.5 border-slate-500 data-[state=checked]:bg-emerald-600 data-[state=checked]:border-emerald-600"
                                                                                    checked={included}
                                                                                    onCheckedChange={() => toggleMergeExclusion(t.id)}
                                                                                />
                                                                                <Label
                                                                                    htmlFor={`merge-inc-${t.id}`}
                                                                                    className="text-[11px] text-slate-300 font-normal leading-snug cursor-pointer flex-1 min-w-0"
                                                                                >
                                                                                    <span className="block truncate text-slate-200">
                                                                                        {t.original_filename}
                                                                                    </span>
                                                                                    <span className="block text-[10px] text-slate-500 tabular-nums">
                                                                                        {new Date(t.created_at).toLocaleString("ru-RU")}
                                                                                    </span>
                                                                                </Label>
                                                                            </div>
                                                                        );
                                                                    })}
                                                                </div>
                                                            </div>
                                                        )}
                                                </div>
                                            </CardHeader>
                                            <CardContent className="p-0 bg-black/40 relative">
                                                <div className="p-4 sm:p-6">
                                                    <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.35fr)] xl:gap-8 xl:items-start">
                                                        {/* Видео: только просмотр, без разметки */}
                                                        <div className="flex flex-col gap-3 min-w-0">
                                                            <div className="flex items-center justify-between gap-3 text-sm font-medium text-slate-300">
                                                                <div className="flex items-center gap-2 min-w-0">
                                                                    <Video className="h-4 w-4 text-sky-400 shrink-0" />
                                                                    <span>Видео</span>
                                                                </div>
                                                                {selectedTask && (
                                                                    <Button
                                                                        asChild
                                                                        size="sm"
                                                                        variant="outline"
                                                                        className="h-8 gap-2 border-slate-700 bg-slate-900/80 text-slate-100 hover:bg-slate-800"
                                                                    >
                                                                        <a
                                                                            href={selectedOriginalVideoUrl}
                                                                            download={selectedTask.original_filename || `${selectedTask.id}.mp4`}
                                                                        >
                                                                            <Download className="h-4 w-4" />
                                                                            Скачать
                                                                        </a>
                                                                    </Button>
                                                                )}
                                                            </div>
                                                            <div className="overflow-hidden rounded-xl border border-slate-800 bg-black shadow-2xl">
                                                                <video
                                                                    key={`${selectedTaskId}-${videoPreviewRetry}`}
                                                                    ref={videoRef}
                                                                    src={selectedVideoUrl}
                                                                    controls
                                                                    playsInline
                                                                    className="block h-[min(560px,55vh)] w-full object-contain bg-black"
                                                                    onLoadedMetadata={(e) => {
                                                                        const rate = Number.parseFloat(playbackRate);
                                                                        if (Number.isFinite(rate) && rate > 0) {
                                                                            e.currentTarget.playbackRate = rate;
                                                                        }
                                                                    }}
                                                                    onError={() => {
                                                                        if (!selectedTaskId) return;
                                                                        window.setTimeout(() => {
                                                                            setVideoPreviewRetry((v) => v + 1);
                                                                        }, 5000);
                                                                    }}
                                                                />
                                                            </div>
                                                            <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-800/80 bg-slate-900/80 px-3 py-2.5">
                                                                <div className="flex items-center gap-2 text-slate-400">
                                                                    <Gauge className="h-4 w-4 shrink-0" />
                                                                    <span className="text-xs font-medium">Скорость</span>
                                                                </div>
                                                                <Select value={playbackRate} onValueChange={setPlaybackRate}>
                                                                    <SelectTrigger className="h-9 w-[120px] border-slate-700 bg-slate-800/80 text-slate-100">
                                                                        <SelectValue placeholder="1×" />
                                                                    </SelectTrigger>
                                                                    <SelectContent className="bg-slate-900 border-slate-700 text-slate-100">
                                                                        {PLAYBACK_RATES.map((r) => (
                                                                            <SelectItem key={r} value={r}>
                                                                                {r === "1" ? "1× (норма)" : `${r}×`}
                                                                            </SelectItem>
                                                                        ))}
                                                                    </SelectContent>
                                                                </Select>
                                                                <p className="text-[11px] text-slate-500 ml-auto">
                                                                    Траектория рисуется только на чертеже
                                                                </p>
                                                            </div>
                                                        </div>

                                                        {/* Чертеж: единственное место для траектории */}
                                                        <div className="flex flex-col gap-3 min-w-0">
                                                            <div className="flex items-center gap-2 text-base font-semibold text-slate-200">
                                                                <ImageIcon className="h-5 w-5 text-emerald-400 shrink-0" />
                                                                Чертеж
                                                                {!selectedTask.map_context?.floor_plan_data && !selectedTask.map_context?.drawn_plan && (
                                                                    <Badge variant="outline" className="text-[10px] border-amber-500/40 text-amber-400/90">
                                                                        план не загружен — сетка
                                                                    </Badge>
                                                                )}
                                                            </div>
                                                            <div
                                                                className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950 w-full min-h-[320px] sm:min-h-[440px] h-[min(760px,82vh)] shadow-2xl group"
                                                                onWheel={handlePlanWheel}
                                                            >
                                                                <div
                                                                    className="absolute top-2 right-2 z-10 flex flex-col items-stretch gap-1 rounded-lg border border-slate-700/80 bg-slate-900/95 p-1.5 shadow-lg backdrop-blur-sm"
                                                                    onClick={(ev) => ev.stopPropagation()}
                                                                    onWheel={(ev) => ev.stopPropagation()}
                                                                >
                                                                    <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-400 text-center px-1">
                                                                        План
                                                                    </span>
                                                                    <div className="flex gap-1 justify-center">
                                                                        <Button
                                                                            type="button"
                                                                            variant="secondary"
                                                                            size="icon"
                                                                            className="h-8 w-8 shrink-0 border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700"
                                                                            title="Увеличить"
                                                                            onClick={() =>
                                                                                setPlanZoom((z) =>
                                                                                    Math.min(PLAN_ZOOM_MAX, Math.round((z + 0.15) * 100) / 100)
                                                                                )
                                                                            }
                                                                        >
                                                                            <ZoomIn className="h-4 w-4" />
                                                                        </Button>
                                                                        <Button
                                                                            type="button"
                                                                            variant="secondary"
                                                                            size="icon"
                                                                            className="h-8 w-8 shrink-0 border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700"
                                                                            title="Уменьшить"
                                                                            onClick={() =>
                                                                                setPlanZoom((z) =>
                                                                                    Math.max(PLAN_ZOOM_MIN, Math.round((z - 0.15) * 100) / 100)
                                                                                )
                                                                            }
                                                                        >
                                                                            <ZoomOut className="h-4 w-4" />
                                                                        </Button>
                                                                        <Button
                                                                            type="button"
                                                                            variant="secondary"
                                                                            size="icon"
                                                                            className="h-8 w-8 shrink-0 border-slate-600 bg-slate-800 text-slate-100 hover:bg-slate-700"
                                                                            title="Сброс масштаба и положения"
                                                                            onClick={() => {
                                                                                setPlanZoom(1);
                                                                                setPlanPan({ x: 0, y: 0 });
                                                                            }}
                                                                        >
                                                                            <RotateCcw className="h-4 w-4" />
                                                                        </Button>
                                                                    </div>
                                                                    <span className="text-center font-mono text-[11px] tabular-nums text-slate-300">
                                                                        {Math.round(planZoom * 100)}%
                                                                    </span>
                                                                    <div className="mt-1.5 border-t border-slate-700/80 pt-1.5 space-y-1.5 w-[8.5rem]">
                                                                        <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-400 text-center block">
                                                                            Линия
                                                                        </span>
                                                                        <Slider
                                                                            value={[trajectoryStrokeWidth]}
                                                                            min={TRAJECTORY_STROKE_MIN}
                                                                            max={TRAJECTORY_STROKE_MAX}
                                                                            step={0.25}
                                                                            onValueChange={(v) =>
                                                                                setTrajectoryStrokeWidth(v[0] ?? TRAJECTORY_STROKE_DEFAULT)
                                                                            }
                                                                            className="w-full"
                                                                        />
                                                                        <span className="text-center font-mono text-[10px] tabular-nums text-slate-400 block">
                                                                            {trajectoryStrokeWidth.toFixed(2)} px
                                                                        </span>
                                                                    </div>
                                                                </div>
                                                                <svg
                                                                    ref={planContainerRef}
                                                                    className={`w-full h-full touch-none select-none ${
                                                                        isPlanPanning ? "cursor-grabbing" : "cursor-crosshair"
                                                                    }`}
                                                                    viewBox={`0 0 ${PLAN_VB_W} ${PLAN_VB_H}`}
                                                                    preserveAspectRatio="xMidYMid meet"
                                                                    onClick={handlePlanCanvasClick}
                                                                    onPointerDown={handlePlanPointerDown}
                                                                    onPointerMove={handlePlanPointerMove}
                                                                    onPointerUp={handlePlanPointerUp}
                                                                    onPointerCancel={handlePlanPointerUp}
                                                                    onContextMenu={(ev) => ev.preventDefault()}
                                                                >
                                                                    <defs>
                                                                        <pattern
                                                                            id="adminTrajectoryGrid"
                                                                            width="20"
                                                                            height="20"
                                                                            patternUnits="userSpaceOnUse"
                                                                        >
                                                                            <path
                                                                                d="M 20 0 L 0 0 0 20"
                                                                                fill="none"
                                                                                stroke="rgba(148, 163, 184, 0.35)"
                                                                                strokeWidth="0.5"
                                                                            />
                                                                        </pattern>
                                                                        <filter id="adminTrajectoryGlow">
                                                                            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                                                                            <feMerge>
                                                                                <feMergeNode in="coloredBlur" />
                                                                                <feMergeNode in="SourceGraphic" />
                                                                            </feMerge>
                                                                        </filter>
                                                                    </defs>
                                                                    <g transform={`translate(${planPan.x} ${planPan.y})`}>
                                                                    <g
                                                                        transform={`translate(${PLAN_VB_W / 2} ${PLAN_VB_H / 2}) scale(${planZoom}) translate(${-PLAN_VB_W / 2} ${-PLAN_VB_H / 2})`}
                                                                    >
                                                                    <g ref={planLayerRef}>
                                                                    {selectedTask.map_context?.floor_plan_data || selectedTask.map_context?.drawn_plan ? (
                                                                        <>
                                                                            {selectedTask.map_context?.floor_plan_data ? (
                                                                                <>
                                                                                    <rect x="0" y="0" width={PLAN_VB_W} height={PLAN_VB_H} fill="white" />
                                                                                    <image
                                                                                        href={selectedTask.map_context.floor_plan_data}
                                                                                        x="0"
                                                                                        y="0"
                                                                                        width={PLAN_VB_W}
                                                                                        height={PLAN_VB_H}
                                                                                        preserveAspectRatio="xMidYMid meet"
                                                                                        opacity={0.9}
                                                                                    />
                                                                                </>
                                                                            ) : (
                                                                                <rect width="100%" height="100%" fill="url(#adminTrajectoryGrid)" />
                                                                            )}

                                                                            {selectedTask.map_context?.drawn_plan &&
                                                                                Array.isArray(selectedTask.map_context.drawn_plan) &&
                                                                                selectedTask.map_context.drawn_plan.map(
                                                                                    (shape: { id: string; type: string; points: { x: number; y: number }[] }) => (
                                                                                        <g key={shape.id}>
                                                                                            {shape.type === "rect" ? (
                                                                                                <rect
                                                                                                    x={Math.min(shape.points[0].x, shape.points[1].x)}
                                                                                                    y={Math.min(shape.points[0].y, shape.points[1].y)}
                                                                                                    width={Math.abs(shape.points[1].x - shape.points[0].x)}
                                                                                                    height={Math.abs(shape.points[1].y - shape.points[0].y)}
                                                                                                    fill="rgba(56, 189, 248, 0.2)"
                                                                                                    stroke="#38bdf8"
                                                                                                    strokeWidth="2"
                                                                                                />
                                                                                            ) : (
                                                                                                <line
                                                                                                    x1={shape.points[0].x}
                                                                                                    y1={shape.points[0].y}
                                                                                                    x2={shape.points[1].x}
                                                                                                    y2={shape.points[1].y}
                                                                                                    stroke="white"
                                                                                                    strokeWidth="3"
                                                                                                    strokeLinecap="round"
                                                                                                    opacity={0.8}
                                                                                                />
                                                                                            )}
                                                                                        </g>
                                                                                    )
                                                                                )}
                                                                        </>
                                                                    ) : (
                                                                        <rect width="100%" height="100%" fill="url(#adminTrajectoryGrid)" />
                                                                    )}

                                                                    {adminReferenceSvg && (
                                                                        <g
                                                                            transform={`translate(${adminReferenceSvg.x}, ${adminReferenceSvg.y})`}
                                                                            style={{ pointerEvents: "none" }}
                                                                        >
                                                                            <circle r="12" fill="rgba(239, 68, 68, 0.2)" className="animate-pulse" />
                                                                            <circle r="6" fill="rgb(239, 68, 68)" stroke="white" strokeWidth="2" />
                                                                        </g>
                                                                    )}

                                                                    {adminTrajectoryPathD !== "" && (
                                                                        <path
                                                                            d={adminTrajectoryPathD}
                                                                            fill="none"
                                                                            stroke={TRAJECTORY_STROKE}
                                                                            strokeWidth={trajectoryStrokeWidth}
                                                                            strokeLinecap="round"
                                                                            strokeLinejoin="round"
                                                                            filter="url(#adminTrajectoryGlow)"
                                                                            opacity={0.95}
                                                                        />
                                                                    )}
                                                                    {manualPointsFlat
                                                                        .filter((_, index) => index % 10 === 0)
                                                                        .map((point, index) => (
                                                                            <circle
                                                                                key={`pp-sample-${index}`}
                                                                                cx={point.x}
                                                                                cy={point.y}
                                                                                r="1.5"
                                                                                fill={TRAJECTORY_STROKE}
                                                                                opacity={0.7}
                                                                            />
                                                                        ))}
                                                                    {adminFirstPlanPt && (
                                                                        <circle
                                                                            cx={adminFirstPlanPt.x}
                                                                            cy={adminFirstPlanPt.y}
                                                                            r="3"
                                                                            fill={TRAJECTORY_START_FILL}
                                                                            stroke="white"
                                                                            strokeWidth="2"
                                                                        />
                                                                    )}
                                                                    {manualPointsFlat.length > 1 && adminLastPlanPt && (
                                                                        <circle
                                                                            cx={adminLastPlanPt.x}
                                                                            cy={adminLastPlanPt.y}
                                                                            r="3"
                                                                            fill="#ef4444"
                                                                            stroke="white"
                                                                            strokeWidth="2"
                                                                        />
                                                                    )}
                                                                    </g>
                                                                    </g>
                                                                    </g>
                                                                </svg>
                                                                {manualPointsFlat.length === 0 && (
                                                                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                                                        <div className="text-center bg-black/55 backdrop-blur-md px-4 py-3 rounded-xl border border-white/10 max-w-[90%] opacity-90">
                                                                            <MousePointer2 className="h-7 w-7 mx-auto mb-2 text-emerald-400" />
                                                                            <p className="text-xs sm:text-sm text-slate-200">Кликайте по плану, чтобы отметить траекторию</p>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                            <div className="flex flex-wrap items-center justify-between gap-3 text-slate-400 px-0.5 pt-1">
                                                                <p className="text-sm font-medium">
                                                                    Точек на чертеже: <span className="text-emerald-400 tabular-nums">{manualPointsFlat.length}</span>
                                                                </p>
                                                                <p className="text-xs sm:text-sm italic opacity-75 text-right max-w-[min(100%,28rem)] leading-snug">
                                                                    {!selectedTask.map_context?.floor_plan_data && !selectedTask.map_context?.drawn_plan
                                                                        ? `Сетка ${PLAN_VB_W}×${PLAN_VB_H}. Масштаб: +/− или колесо. Сдвиг: средняя/правая кнопка или Shift+ЛКМ.`
                                                                        : "Масштаб: +/− или колесо. Сдвиг плана: средняя/правая кнопка мыши или Shift+перетаскивание левой."}
                                                                </p>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="bg-slate-800/50 backdrop-blur-md border-t border-slate-800/80 p-6 flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                                                    <div className="flex gap-2">
                                                        <Button 
                                                            variant="outline" 
                                                            size="sm"
                                                            onClick={handleUndoPoint} 
                                                            disabled={manualPointsFlat.length === 0}
                                                            className="bg-slate-800/50 border-slate-700 hover:bg-slate-700 h-10 px-4"
                                                        >
                                                            Отменить точку
                                                        </Button>
                                                        <Button 
                                                            variant="ghost" 
                                                            size="sm"
                                                            onClick={handleClearPoints} 
                                                            disabled={manualPointsFlat.length === 0}
                                                            className="text-slate-400 hover:text-red-400 h-10 px-4"
                                                        >
                                                            Очистить всё
                                                        </Button>
                                                    </div>
                                                    
                                                    <div className="flex flex-col md:items-end gap-3">
                                                        {trajectoryNotice && (
                                                            <div className="flex items-center gap-2 text-xs font-medium text-blue-400 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 animate-in fade-in slide-in-from-right-4 duration-300">
                                                                <CheckCircle2 className="h-3 w-3" />
                                                                {trajectoryNotice}
                                                            </div>
                                                        )}
                                                        <Button 
                                                            onClick={handleSaveManualTrajectory} 
                                                            disabled={isSavingTrajectory || !selectedTaskId || manualPointsFlat.length < 2}
                                                            className="bg-blue-600 hover:bg-blue-500 text-white font-bold h-12 px-8 rounded-xl shadow-lg shadow-blue-600/20 transition-all hover:scale-[1.03] active:scale-[0.97]"
                                                        >
                                                            {isSavingTrajectory ? (
                                                                <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Сохранение...</>
                                                            ) : (
                                                                <><MousePointer2 className="h-4 w-4 mr-2" /> Опубликовать результат далее</>
                                                            )}
                                                        </Button>
                                                    </div>
                                                </div>
                                            </CardContent>
                                        </>
                                    ) : (
                                        <div className="flex flex-col items-center justify-center p-20 text-center space-y-4">
                                            <div className="h-20 w-20 rounded-3xl bg-slate-800/50 border border-slate-700 flex items-center justify-center mb-2">
                                                <MousePointer2 className="h-10 w-10 text-slate-600" />
                                            </div>
                                            <div className="space-y-1">
                                                <h3 className="text-xl font-bold text-slate-300">Ожидание выбора</h3>
                                                <p className="text-slate-500 max-w-sm mx-auto">Выберите задачу из списка слева, чтобы "подхватить" сессию пользователя и нарисовать траекторию</p>
                                            </div>
                                        </div>
                                    )}
                                </Card>
                            </div>
                        </div>
                    )}
                </div>
            </main>

            <style>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 4px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: transparent;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: rgba(255, 255, 255, 0.2);
                }
                .bg-gradient-dark {
                    background: radial-gradient(circle at top right, #0f172a, #020617);
                }
            `}</style>
        </div>
    );
};

export default Admin;
