import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

interface Worker {
  id: string;
  name: string;
  role: string;
  zone: string;
  status: "active" | "idle" | "offline";
  lastSeen: string;
}

const workers: Worker[] = [
  { id: "1", name: "Иванов А.С.", role: "Оператор", zone: "Цех А", status: "active", lastSeen: "Сейчас" },
  { id: "2", name: "Петров В.М.", role: "Техник", zone: "Склад", status: "active", lastSeen: "2 мин назад" },
  { id: "3", name: "Сидоров К.П.", role: "Инженер", zone: "Офис", status: "idle", lastSeen: "15 мин назад" },
  { id: "4", name: "Козлов Д.И.", role: "Оператор", zone: "Цех Б", status: "active", lastSeen: "Сейчас" },
  { id: "5", name: "Новиков Е.А.", role: "Мастер", zone: "—", status: "offline", lastSeen: "2 часа назад" },
];

const WorkersList = () => {
  const statusColors = {
    active: "bg-analytics-safe",
    idle: "bg-analytics-warning",
    offline: "bg-muted-foreground",
  };

  const statusLabels = {
    active: "Активен",
    idle: "Неактивен",
    offline: "Офлайн",
  };

  return (
    <div className="p-6 rounded-2xl bg-gradient-card border border-border/50">
      <h3 className="text-lg font-semibold mb-4">Сотрудники</h3>
      <div className="space-y-3">
        {workers.map((worker) => (
          <div 
            key={worker.id}
            className="flex items-center gap-3 p-3 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors cursor-pointer"
          >
            <div className="relative">
              <Avatar className="h-10 w-10">
                <AvatarFallback className="bg-primary/10 text-primary text-sm">
                  {worker.name.split(" ").map(n => n[0]).join("")}
                </AvatarFallback>
              </Avatar>
              <span 
                className={`absolute bottom-0 right-0 h-3 w-3 rounded-full border-2 border-card ${statusColors[worker.status]}`} 
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-medium truncate">{worker.name}</div>
              <div className="text-sm text-muted-foreground">{worker.role}</div>
            </div>
            <div className="text-right">
              <Badge variant="secondary" className="mb-1">{worker.zone}</Badge>
              <div className="text-xs text-muted-foreground">{worker.lastSeen}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default WorkersList;
