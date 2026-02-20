import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { apiClient, Plan } from "@/lib/api";
import { Trash2, CheckCircle2, Map as MapIcon, Loader2 } from "lucide-react";
import { toast } from "sonner";

interface PlanLibraryProps {
    onSelect: (plan: Plan) => void;
    selectedId?: number;
}

const PlanLibrary = ({ onSelect, selectedId }: PlanLibraryProps) => {
    const [plans, setPlans] = useState<Plan[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    const loadPlans = async () => {
        try {
            setIsLoading(true);
            const data = await apiClient.getPlans();
            setPlans(data);
        } catch (error) {
            console.error(error);
            toast.error("Ошибка при загрузке планов");
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        loadPlans();
    }, []);

    const deletePlan = async (e: React.MouseEvent, id: number) => {
        e.stopPropagation();
        if (!window.confirm("Удалить этот план?")) return;

        try {
            await apiClient.deletePlan(id);
            setPlans(prev => prev.filter(p => p.id !== id));
            toast.success("План удален");
        } catch (error) {
            toast.error("Ошибка при удалении");
        }
    };

    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center py-12 gap-3">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <p className="text-muted-foreground animate-pulse">Загрузка библиотеки планов...</p>
            </div>
        );
    }

    if (plans.length === 0) {
        return (
            <div className="text-center py-12 border-2 border-dashed rounded-xl bg-muted/20">
                <MapIcon className="h-12 w-12 text-muted-foreground mx-auto mb-4 opacity-20" />
                <p className="text-muted-foreground font-medium">Библиотека пуста</p>
                <p className="text-xs text-muted-foreground mt-1">Нарисуйте свой первый план, чтобы он появился здесь</p>
            </div>
        );
    }

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {plans.map((plan) => (
                <Card
                    key={plan.id}
                    className={`group cursor-pointer transition-all duration-300 hover:shadow-xl hover:-translate-y-1 border-primary/10 overflow-hidden relative ${selectedId === plan.id ? 'ring-2 ring-primary bg-primary/5 border-transparent' : ''}`}
                    onClick={() => onSelect(plan)}
                >
                    <div className="h-36 bg-slate-900 overflow-hidden p-4 flex items-center justify-center relative bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:20px_20px]">
                        <div
                            className="w-full h-full pointer-events-none scale-[0.8]"
                            dangerouslySetInnerHTML={{ __html: plan.preview_svg || "" }}
                        />

                        {/* Overlay for selected state */}
                        {selectedId === plan.id && (
                            <div className="absolute inset-0 bg-primary/10 flex items-center justify-center">
                                <div className="bg-primary text-primary-foreground rounded-full p-1 shadow-lg scale-125">
                                    <CheckCircle2 className="h-6 w-6" />
                                </div>
                            </div>
                        )}
                    </div>

                    <CardContent className="p-4">
                        <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0">
                                <h3 className="font-bold text-sm truncate group-hover:text-primary transition-colors">
                                    {plan.name}
                                </h3>
                                <time className="text-[10px] text-muted-foreground mt-1 block">
                                    {plan.created_at ? new Date(plan.created_at).toLocaleString('ru-RU', {
                                        day: '2-digit',
                                        month: 'long',
                                        year: 'numeric'
                                    }) : ""}
                                </time>
                            </div>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-muted-foreground hover:bg-destructive/10 hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity"
                                onClick={(e) => deletePlan(e, plan.id!)}
                            >
                                <Trash2 className="h-4 w-4" />
                            </Button>
                        </div>

                        <div className="mt-3 pt-3 border-t border-border/50 flex items-center justify-between">
                            <span className="text-[10px] uppercase font-bold tracking-widest text-muted-foreground/60">
                                Hand Drawn
                            </span>
                            <Button variant="link" size="sm" className="h-auto p-0 text-xs font-semibold">
                                Выбрать
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            ))}
        </div>
    );
};

export default PlanLibrary;
