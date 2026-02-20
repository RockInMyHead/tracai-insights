import { useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Square, Slash, Save, Trash2, Undo2, MousePointer2, X } from "lucide-react";
import { toast } from "sonner";
import { apiClient, Plan } from "@/lib/api";

interface Point {
    x: number;
    y: number;
}

interface Shape {
    id: string;
    type: 'rect' | 'line';
    points: Point[];
    color?: string;
}

interface PlanEditorProps {
    onSave?: (plan: Plan) => void;
    onCancel?: () => void;
}

const PlanEditor = ({ onSave, onCancel }: PlanEditorProps) => {
    const [shapes, setShapes] = useState<Shape[]>([]);
    const [currentShape, setCurrentShape] = useState<Shape | null>(null);
    const [mode, setMode] = useState<'select' | 'rect' | 'line'>('rect');
    const [planName, setPlanName] = useState("");
    const [isSaving, setIsSaving] = useState(false);
    const svgRef = useRef<SVGSVGElement>(null);

    const getCoordinates = (e: React.MouseEvent | React.TouchEvent): Point | null => {
        if (!svgRef.current) return null;
        const svg = svgRef.current;
        const rect = svg.getBoundingClientRect();

        let clientX, clientY;
        if ('clientX' in e) {
            clientX = e.clientX;
            clientY = e.clientY;
        } else {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        // Convert screen coordinates to SVG coordinates
        const scaleX = 800 / rect.width;
        const scaleY = 600 / rect.height;

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    };

    const handleMouseDown = (e: React.MouseEvent) => {
        if (mode === 'select') return;

        const coords = getCoordinates(e);
        if (!coords) return;

        setCurrentShape({
            id: Date.now().toString(),
            type: mode as 'rect' | 'line',
            points: [coords, coords]
        });
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!currentShape) return;

        const coords = getCoordinates(e);
        if (!coords) return;

        setCurrentShape(prev => {
            if (!prev) return null;
            return {
                ...prev,
                points: [prev.points[0], coords]
            };
        });
    };

    const handleMouseUp = () => {
        if (currentShape) {
            // Only add if it has some size
            const dist = Math.sqrt(
                Math.pow(currentShape.points[1].x - currentShape.points[0].x, 2) +
                Math.pow(currentShape.points[1].y - currentShape.points[0].y, 2)
            );

            if (dist > 5) {
                setShapes(prev => [...prev, currentShape]);
            }
            setCurrentShape(null);
        }
    };

    const clearAll = () => {
        if (window.confirm("Очистить чертеж?")) {
            setShapes([]);
        }
    };

    const undo = () => {
        setShapes(prev => prev.slice(0, -1));
    };

    const savePlan = async () => {
        if (!planName) {
            toast.error("Введите название плана");
            return;
        }
        if (shapes.length === 0) {
            toast.error("Чертеж пуст");
            return;
        }

        setIsSaving(true);
        try {
            // Create a static SVG for preview
            const previewSvg = `
        <svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
          ${shapes.map(s => s.type === 'rect' ?
                `<rect x="${Math.min(s.points[0].x, s.points[1].x)}" y="${Math.min(s.points[0].y, s.points[1].y)}" width="${Math.abs(s.points[1].x - s.points[0].x)}" height="${Math.abs(s.points[1].y - s.points[0].y)}" fill="rgba(56, 189, 248, 0.2)" stroke="#38bdf8" stroke-width="2" />` :
                `<line x1="${s.points[0].x}" y1="${s.points[0].y}" x2="${s.points[1].x}" y2="${s.points[1].y}" stroke="white" stroke-width="3" stroke-linecap="round" />`
            ).join('')}
        </svg>
      `;

            const newPlan: Plan = {
                name: planName,
                data: shapes,
                preview_svg: previewSvg
            };

            const result = await apiClient.savePlan(newPlan);
            toast.success(`План "${planName}" успешно сохранен`);
            if (onSave) onSave({ ...newPlan, id: result.id });
        } catch (error) {
            toast.error("Ошибка при сохранении плана");
            console.error(error);
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm p-4">
            <Card className="w-full max-w-5xl h-[90vh] flex flex-col overflow-hidden shadow-2xl border-primary/20">
                <CardHeader className="py-3 px-4 flex flex-row items-center justify-between border-b bg-muted/50">
                    <div className="flex items-center gap-4">
                        <CardTitle className="text-lg font-bold flex items-center gap-2">
                            <MousePointer2 className="h-5 w-5 text-primary" />
                            Конструктор плана
                        </CardTitle>
                        <div className="flex items-center gap-1 bg-background p-1 rounded-md border shadow-sm">
                            <Button
                                variant={mode === 'rect' ? "default" : "ghost"}
                                size="sm"
                                onClick={() => setMode('rect')}
                                className="h-8 px-3 gap-2"
                                title="Помещение (Прямоугольник)"
                            >
                                <Square className="h-4 w-4" />
                                <span className="text-xs hidden sm:inline">Помещение</span>
                            </Button>
                            <Button
                                variant={mode === 'line' ? "default" : "ghost"}
                                size="sm"
                                onClick={() => setMode('line')}
                                className="h-8 px-3 gap-2"
                                title="Стена (Линия)"
                            >
                                <Slash className="h-4 w-4" />
                                <span className="text-xs hidden sm:inline">Стена</span>
                            </Button>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <Button variant="outline" size="sm" onClick={undo} disabled={shapes.length === 0} className="h-8">
                            <Undo2 className="h-4 w-4 sm:mr-1" /> <span className="hidden sm:inline">Отмена</span>
                        </Button>
                        <Button variant="ghost" size="icon" onClick={onCancel} className="h-8 w-8">
                            <X className="h-5 w-5" />
                        </Button>
                    </div>
                </CardHeader>

                <CardContent className="p-0 flex-1 relative bg-slate-950 overflow-hidden select-none">
                    <svg
                        ref={svgRef}
                        className="w-full h-full cursor-crosshair touch-none"
                        viewBox="0 0 800 600"
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                    >
                        <defs>
                            <pattern id="editorGrid" width="40" height="40" patternUnits="userSpaceOnUse">
                                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="white" strokeWidth="0.5" opacity="0.1" />
                            </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill="url(#editorGrid)" />

                        {/* Render existing shapes */}
                        {shapes.map((shape) => (
                            <g key={shape.id}>
                                {shape.type === 'rect' ? (
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
                                    />
                                )}
                            </g>
                        ))}

                        {/* Render current shape being drawn */}
                        {currentShape && (
                            <g opacity="0.6">
                                {currentShape.type === 'rect' ? (
                                    <rect
                                        x={Math.min(currentShape.points[0].x, currentShape.points[1].x)}
                                        y={Math.min(currentShape.points[0].y, currentShape.points[1].y)}
                                        width={Math.abs(currentShape.points[1].x - currentShape.points[0].x)}
                                        height={Math.abs(currentShape.points[1].y - currentShape.points[0].y)}
                                        fill="rgba(56, 189, 248, 0.4)"
                                        stroke="#38bdf8"
                                        strokeWidth="2"
                                        strokeDasharray="4"
                                    />
                                ) : (
                                    <line
                                        x1={currentShape.points[0].x}
                                        y1={currentShape.points[0].y}
                                        x2={currentShape.points[1].x}
                                        y2={currentShape.points[1].y}
                                        stroke="white"
                                        strokeWidth="3"
                                        strokeDasharray="4"
                                    />
                                )}
                            </g>
                        )}
                    </svg>

                    <div className="absolute bottom-4 left-4 bg-background/80 backdrop-blur-md p-2 rounded-lg border text-[10px] text-muted-foreground">
                        Зажмите и тяните, чтобы рисовать. Прямоугольники — помещения, линии — стены.
                    </div>
                </CardContent>

                <CardFooter className="py-4 px-6 border-t bg-muted/30 flex gap-4">
                    <Input
                        placeholder="Название плана (напр. Склад А1)"
                        value={planName}
                        onChange={(e) => setPlanName(e.target.value)}
                        className="flex-1 bg-background"
                    />
                    <div className="flex gap-2">
                        <Button variant="outline" onClick={clearAll} className="text-destructive hover:bg-destructive/10">
                            Очистить
                        </Button>
                        <Button onClick={savePlan} disabled={isSaving || shapes.length === 0} className="px-8">
                            {isSaving ? "Сохранение..." : <><Save className="h-4 w-4 mr-2" /> Сохранить</>}
                        </Button>
                    </div>
                </CardFooter>
            </Card>
        </div>
    );
};

export default PlanEditor;
