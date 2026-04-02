import { FormEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import Navbar from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { apiClient, TrackingTask } from "@/lib/api";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { MousePointer2, Image as ImageIcon, Video, User, Clock, CheckCircle2, AlertCircle, Loader2, RefreshCw, Gauge } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const ADMIN_LOGIN = "adm!n";
const ADMIN_PASSWORD = "bdyltrcrjvgfybz123";
const ADMIN_SESSION_KEY = "trackai_admin_authenticated";

const PLAYBACK_RATES = ["0.25", "0.5", "0.75", "1", "1.25", "1.5", "2"] as const;

const Admin = () => {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [error, setError] = useState("");
    const [tasks, setTasks] = useState<TrackingTask[]>([]);
    const [isLoadingData, setIsLoadingData] = useState(false);
    const [dataError, setDataError] = useState("");
    const [selectedTaskId, setSelectedTaskId] = useState("");
    const [manualPoints, setManualPoints] = useState<Array<{ x: number; y: number }>>([]);
    const [playbackRate, setPlaybackRate] = useState("1");
    const [isSavingTrajectory, setIsSavingTrajectory] = useState(false);
    const [trajectoryNotice, setTrajectoryNotice] = useState("");
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const planContainerRef = useRef<SVGSVGElement | null>(null);

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
        setManualPoints([]);
        setTrajectoryNotice("");
    };

    const loadAdminData = async () => {
        setIsLoadingData(true);
        setDataError("");
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

    const selectedVideoUrl = selectedTask
        ? apiClient.getUploadedVideoUrl(selectedTask.id)
        : "";

    useEffect(() => {
        const v = videoRef.current;
        if (!v) return;
        const rate = Number.parseFloat(playbackRate);
        if (!Number.isFinite(rate) || rate <= 0) return;
        v.playbackRate = rate;
    }, [playbackRate, selectedVideoUrl, selectedTaskId]);

    const handlePlanCanvasClick = (event: MouseEvent<SVGSVGElement>) => {
        if (!planContainerRef.current) return;
        const svg = planContainerRef.current;
        const rect = svg.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;

        // Convert screen coordinates to viewBox coordinates (0-100 scale for consistency)
        const x = ((event.clientX - rect.left) / rect.width) * 100;
        const y = ((event.clientY - rect.top) / rect.height) * 100;

        setManualPoints((prev) => [...prev, { x: Number(x.toFixed(3)), y: Number(y.toFixed(3)) }]);
        setTrajectoryNotice("");
    };

    const handleUndoPoint = () => {
        setManualPoints((prev) => prev.slice(0, -1));
    };

    const handleClearPoints = () => {
        setManualPoints([]);
    };

    const handleSelectTask = async (taskId: string) => {
        setSelectedTaskId(taskId);
        setPlaybackRate("1");
        setManualPoints([]);
        setTrajectoryNotice("");
        if (!taskId) return;
        try {
            const manual = await apiClient.getManualTrajectory(taskId);
            if (manual.exists && Array.isArray(manual.trajectory)) {
                setManualPoints(
                    manual.trajectory
                        .filter((p) => Array.isArray(p) && p.length >= 2)
                        .map((p) => ({ x: Number(p[0]), y: Number(p[1]) }))
                );
                setTrajectoryNotice("Загружена ранее сохраненная ручная траектория");
            }
        } catch (e) {
            console.warn("Manual trajectory error:", e);
        }
    };

    const handleSaveManualTrajectory = async () => {
        if (!selectedTaskId) {
            setTrajectoryNotice("Сначала выберите задачу");
            return;
        }
        if (manualPoints.length < 2) {
            setTrajectoryNotice("Добавьте минимум 2 точки траектории");
            return;
        }
        setIsSavingTrajectory(true);
        setTrajectoryNotice("");
        try {
            const trajectory = manualPoints.map((p) => [p.x, p.y, 0]);
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
                                <div className="flex gap-3">
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
                                        <CardDescription>Выберите одного из активных пользователей</CardDescription>
                                    </CardHeader>
                                    <CardContent className="p-0">
                                        <div className="flex flex-col max-h-[600px] overflow-y-auto divide-y divide-slate-800/50 custom-scrollbar">
                                            {tasks.length === 0 ? (
                                                <div className="p-8 text-center text-slate-500">
                                                    <Clock className="h-10 w-10 mx-auto mb-3 opacity-20" />
                                                    <p className="text-sm">Активных задач пока нет</p>
                                                </div>
                                            ) : (
                                                tasks.map((task) => (
                                                    <button
                                                        key={task.id}
                                                        onClick={() => handleSelectTask(task.id)}
                                                        className={`flex flex-col gap-1.5 p-4 text-left transition-all hover:bg-white/5 active:bg-white/10 ${
                                                            selectedTaskId === task.id ? "bg-blue-600/10 border-l-2 border-l-blue-500" : ""
                                                        }`}
                                                    >
                                                        <div className="flex items-center justify-between gap-2">
                                                            <div className="flex items-center gap-2 min-w-0">
                                                                <User className="h-3.5 w-3.5 text-blue-400 shrink-0" />
                                                                <span className="font-semibold text-sm truncate">
                                                                    {task.employee_name || "Инкогнито"}
                                                                </span>
                                                            </div>
                                                            <Badge 
                                                                variant="outline" 
                                                                className={`text-[10px] px-1.5 py-0 uppercase tracking-wider ${
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
                                                        <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
                                                            <Video className="h-3 w-3" />
                                                            <span className="truncate">{task.original_filename}</span>
                                                        </div>
                                                        <div className="text-[10px] text-slate-600 mt-0.5">
                                                            {new Date(task.created_at).toLocaleString('ru-RU')}
                                                        </div>
                                                    </button>
                                                ))
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
                                                </div>
                                            </CardHeader>
                                            <CardContent className="p-0 bg-black/40 relative">
                                                <div className="p-4 sm:p-6">
                                                    <div className="grid grid-cols-1 gap-6 xl:grid-cols-2 xl:gap-8 xl:items-start">
                                                        {/* Видео: только просмотр, без разметки */}
                                                        <div className="flex flex-col gap-3 min-w-0">
                                                            <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
                                                                <Video className="h-4 w-4 text-sky-400 shrink-0" />
                                                                Видео
                                                            </div>
                                                            <div className="overflow-hidden rounded-xl border border-slate-800 bg-black shadow-2xl">
                                                                <video
                                                                    key={selectedTaskId}
                                                                    ref={videoRef}
                                                                    src={selectedVideoUrl}
                                                                    controls
                                                                    playsInline
                                                                    className="block h-[min(520px,52vh)] w-full object-contain bg-black"
                                                                    onLoadedMetadata={(e) => {
                                                                        const rate = Number.parseFloat(playbackRate);
                                                                        if (Number.isFinite(rate) && rate > 0) {
                                                                            e.currentTarget.playbackRate = rate;
                                                                        }
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
                                                            <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
                                                                <ImageIcon className="h-4 w-4 text-emerald-400 shrink-0" />
                                                                Чертеж
                                                                {!selectedTask.map_context?.floor_plan_data && !selectedTask.map_context?.drawn_plan && (
                                                                    <Badge variant="outline" className="text-[10px] border-amber-500/40 text-amber-400/90">
                                                                        план не загружен — сетка
                                                                    </Badge>
                                                                )}
                                                            </div>
                                                            <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950 aspect-[16/9] max-h-[min(520px,52vh)] min-h-[240px] shadow-2xl group">
                                                                <svg
                                                                    ref={planContainerRef}
                                                                    className="w-full h-full cursor-crosshair touch-none"
                                                                    viewBox="0 0 100 100"
                                                                    preserveAspectRatio="none"
                                                                    onClick={handlePlanCanvasClick}
                                                                >
                                                                    <defs>
                                                                        <pattern id="adminGrid" width="5" height="5" patternUnits="userSpaceOnUse">
                                                                            <path d="M 5 0 L 0 0 0 5" fill="none" stroke="white" strokeWidth="0.2" opacity="0.05" />
                                                                        </pattern>
                                                                    </defs>
                                                                    <rect width="100%" height="100%" fill="url(#adminGrid)" />

                                                                    {selectedTask.map_context?.floor_plan_data && (
                                                                        <image
                                                                            href={selectedTask.map_context.floor_plan_data}
                                                                            width="100%"
                                                                            height="100%"
                                                                            preserveAspectRatio="xMidYMid meet"
                                                                            opacity="0.6"
                                                                        />
                                                                    )}

                                                                    {selectedTask.map_context?.drawn_plan && Array.isArray(selectedTask.map_context.drawn_plan) && selectedTask.map_context.drawn_plan.map((shape: { id: string; type: string; points: { x: number; y: number }[] }) => (
                                                                        <g key={shape.id}>
                                                                            {shape.type === "rect" ? (
                                                                                <rect
                                                                                    x={(Math.min(shape.points[0].x, shape.points[1].x) / 800) * 100}
                                                                                    y={(Math.min(shape.points[0].y, shape.points[1].y) / 600) * 100}
                                                                                    width={(Math.abs(shape.points[1].x - shape.points[0].x) / 800) * 100}
                                                                                    height={(Math.abs(shape.points[1].y - shape.points[0].y) / 600) * 100}
                                                                                    fill="rgba(56, 189, 248, 0.15)"
                                                                                    stroke="#0ea5e9"
                                                                                    strokeWidth="0.3"
                                                                                />
                                                                            ) : (
                                                                                <line
                                                                                    x1={(shape.points[0].x / 800) * 100}
                                                                                    y1={(shape.points[0].y / 600) * 100}
                                                                                    x2={(shape.points[1].x / 800) * 100}
                                                                                    y2={(shape.points[1].y / 600) * 100}
                                                                                    stroke="white"
                                                                                    strokeWidth="0.5"
                                                                                    strokeLinecap="round"
                                                                                    opacity="0.8"
                                                                                />
                                                                            )}
                                                                        </g>
                                                                    ))}

                                                                    {selectedTask.map_context?.reference_point && (
                                                                        <g transform={`translate(${selectedTask.map_context.reference_point.x}, ${selectedTask.map_context.reference_point.y})`}>
                                                                            <circle r="1.5" fill="rgba(239, 68, 68, 0.3)" className="animate-ping" />
                                                                            <circle r="0.8" fill="#ef4444" stroke="white" strokeWidth="0.2" />
                                                                        </g>
                                                                    )}

                                                                    {manualPoints.length > 1 && (
                                                                        <polyline
                                                                            points={manualPoints.map((p) => `${p.x},${p.y}`).join(" ")}
                                                                            fill="none"
                                                                            stroke="#22d3ee"
                                                                            strokeWidth="0.8"
                                                                            strokeLinecap="round"
                                                                            strokeLinejoin="round"
                                                                            className="drop-shadow-[0_0_8px_rgba(34,211,238,0.8)]"
                                                                        />
                                                                    )}
                                                                    {manualPoints.map((p, idx) => (
                                                                        <circle
                                                                            key={`pp-${idx}`}
                                                                            cx={p.x}
                                                                            cy={p.y}
                                                                            r="1"
                                                                            fill={idx === manualPoints.length - 1 ? "#f43f5e" : "#06b6d4"}
                                                                            className={idx === manualPoints.length - 1 ? "animate-pulse" : ""}
                                                                        />
                                                                    ))}
                                                                </svg>
                                                                {manualPoints.length === 0 && (
                                                                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                                                        <div className="text-center bg-black/55 backdrop-blur-md px-4 py-3 rounded-xl border border-white/10 max-w-[90%] opacity-90">
                                                                            <MousePointer2 className="h-7 w-7 mx-auto mb-2 text-emerald-400" />
                                                                            <p className="text-xs sm:text-sm text-slate-200">Кликайте по плану, чтобы отметить траекторию</p>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                            <div className="flex flex-wrap items-center justify-between gap-2 text-slate-400 px-0.5">
                                                                <p className="text-xs font-medium">
                                                                    Точек на чертеже: <span className="text-emerald-400">{manualPoints.length}</span>
                                                                </p>
                                                                <p className="text-[11px] italic opacity-70 text-right">
                                                                    {!selectedTask.map_context?.floor_plan_data && !selectedTask.map_context?.drawn_plan
                                                                        ? "Нет плана пользователя — координаты в условной сетке 0–100"
                                                                        : "Траектория в координатах плана"}
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
                                                            disabled={manualPoints.length === 0}
                                                            className="bg-slate-800/50 border-slate-700 hover:bg-slate-700 h-10 px-4"
                                                        >
                                                            Отменить точку
                                                        </Button>
                                                        <Button 
                                                            variant="ghost" 
                                                            size="sm"
                                                            onClick={handleClearPoints} 
                                                            disabled={manualPoints.length === 0}
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
                                                            disabled={isSavingTrajectory || !selectedTaskId || manualPoints.length < 2}
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
