import Navbar from "@/components/Navbar";
import MovementMap from "@/components/MovementMap";
import VideoUpload from "@/components/VideoUpload";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar, Download, Filter, Play, Pause, RotateCcw } from "lucide-react";
import { useState } from "react";

interface PathPoint {
  x: number;
  y: number;
  time?: string;
}

const timelineData = [
  { time: "08:00", zone: "Офис", duration: "30 мин", type: "safe" },
  { time: "08:30", zone: "Цех А", duration: "1 ч 30 мин", type: "warning" },
  { time: "10:00", zone: "Столовая", duration: "45 мин", type: "safe" },
  { time: "10:45", zone: "Цех Б", duration: "1 ч 15 мин", type: "danger" },
  { time: "12:00", zone: "Склад", duration: "В процессе", type: "safe" },
];

const Analytics = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [analyzedPath, setAnalyzedPath] = useState<PathPoint[] | undefined>(undefined);

  const handleVideoAnalyzed = (coordinates: PathPoint[]) => {
    setAnalyzedPath(coordinates);
  };
  return (
    <div className="min-h-screen bg-gradient-dark">
      <Navbar />
      
      <main className="container mx-auto px-6 pt-24 pb-12">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between mb-6 gap-4">
          <div>
            <h1 className="text-2xl font-bold mb-1">Аналитика движения</h1>
            <p className="text-muted-foreground">Отслеживание маршрутов и времени нахождения в зонах</p>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            <Select defaultValue="ivanov">
              <SelectTrigger className="w-48 bg-secondary border-border">
                <SelectValue placeholder="Выберите сотрудника" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ivanov">Иванов А.С.</SelectItem>
                <SelectItem value="petrov">Петров В.М.</SelectItem>
                <SelectItem value="sidorov">Сидоров К.П.</SelectItem>
                <SelectItem value="kozlov">Козлов Д.И.</SelectItem>
              </SelectContent>
            </Select>
            
            <Button variant="outline" className="gap-2">
              <Calendar className="h-4 w-4" />
              Сегодня
            </Button>
            
            <Button variant="outline" size="icon">
              <Filter className="h-4 w-4" />
            </Button>
            
            <Button variant="outline" size="icon">
              <Download className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Main content grid */}
        <div className="grid lg:grid-cols-4 gap-6">
          {/* Map */}
          <div className="lg:col-span-3 h-[500px]">
            <MovementMap externalPath={analyzedPath} />
          </div>
          
          {/* Right sidebar */}
          <div className="space-y-6">
            {/* Video Upload */}
            <VideoUpload onVideoAnalyzed={handleVideoAnalyzed} />
            
            {/* Timeline */}
            <div className="p-6 rounded-2xl bg-gradient-card border border-border/50">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">История</h3>
                <div className="flex gap-1">
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="h-8 w-8"
                    onClick={() => setIsPlaying(!isPlaying)}
                  >
                    {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  </Button>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              <div className="space-y-4">
                {timelineData.map((item, i) => (
                  <div key={i} className="relative pl-6 pb-4 last:pb-0">
                    {/* Timeline line */}
                    {i < timelineData.length - 1 && (
                      <div className="absolute left-2 top-3 bottom-0 w-px bg-border" />
                    )}
                    
                    {/* Dot */}
                    <div className={`absolute left-0 top-1.5 h-4 w-4 rounded-full border-2 border-card ${
                      item.type === "safe" ? "bg-analytics-safe" :
                      item.type === "warning" ? "bg-analytics-warning" :
                      "bg-analytics-danger"
                    }`} />
                    
                    <div className="text-sm text-muted-foreground mb-1">{item.time}</div>
                    <div className="font-medium mb-1">{item.zone}</div>
                    <Badge variant="secondary" className="text-xs">{item.duration}</Badge>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="p-4 rounded-xl bg-secondary/30 border border-border/30">
            <div className="text-2xl font-bold text-gradient mb-1">4.2 км</div>
            <div className="text-sm text-muted-foreground">Пройдено за день</div>
          </div>
          <div className="p-4 rounded-xl bg-secondary/30 border border-border/30">
            <div className="text-2xl font-bold text-gradient mb-1">5</div>
            <div className="text-sm text-muted-foreground">Посещено зон</div>
          </div>
          <div className="p-4 rounded-xl bg-secondary/30 border border-border/30">
            <div className="text-2xl font-bold text-gradient mb-1">1 ч 15 мин</div>
            <div className="text-sm text-muted-foreground">В опасных зонах</div>
          </div>
          <div className="p-4 rounded-xl bg-secondary/30 border border-border/30">
            <div className="text-2xl font-bold text-gradient mb-1">0</div>
            <div className="text-sm text-muted-foreground">Нарушений</div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Analytics;
