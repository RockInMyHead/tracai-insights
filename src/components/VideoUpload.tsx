import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload, Video, X, Play, Pause, Loader2 } from "lucide-react";
import { toast } from "sonner";

interface VideoUploadProps {
  onVideoAnalyzed?: (coordinates: { x: number; y: number; time?: string }[]) => void;
}

const VideoUpload = ({ onVideoAnalyzed }: VideoUploadProps) => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("video/")) {
        toast.error("Пожалуйста, загрузите видео файл");
        return;
      }
      
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
      setVideoName(file.name);
      toast.success(`Видео "${file.name}" загружено`);
    }
  };

  const clearVideo = () => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setVideoUrl(null);
    setVideoName("");
    setIsPlaying(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    toast.info("Видео удалено");
  };

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleAnalyze = async () => {
    if (!videoUrl) return;
    
    setIsAnalyzing(true);
    toast.info("Начат анализ видео...");
    
    // Simulate analysis - in production this would call an AI/CV backend
    await new Promise((resolve) => setTimeout(resolve, 2000));
    
    // Generate mock coordinates based on video duration
    const duration = videoRef.current?.duration || 60;
    const points: { x: number; y: number; time?: string }[] = [];
    const numPoints = Math.min(Math.floor(duration / 5), 20);
    
    for (let i = 0; i < numPoints; i++) {
      const progress = i / numPoints;
      points.push({
        x: 50 + Math.random() * 700 * progress,
        y: 400 - Math.random() * 300 * progress,
        time: formatTime(i * 5),
      });
    }
    
    setIsAnalyzing(false);
    toast.success(`Распознано ${points.length} точек движения`);
    
    if (onVideoAnalyzed) {
      onVideoAnalyzed(points);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="p-6 rounded-2xl bg-gradient-card border border-border/50">
      <div className="flex items-center gap-2 mb-4">
        <Video className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold">Видео для распознавания</h3>
      </div>
      
      {!videoUrl ? (
        <div className="border-2 border-dashed border-border rounded-xl p-8 text-center">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={handleVideoUpload}
          />
          
          <div className="flex flex-col items-center gap-3">
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
              <Upload className="h-6 w-6 text-primary" />
            </div>
            <div>
              <p className="font-medium mb-1">Загрузите видео</p>
              <p className="text-sm text-muted-foreground">
                Поддерживаются форматы MP4, WebM, MOV
              </p>
            </div>
            <Button
              variant="outline"
              className="gap-2"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="h-4 w-4" />
              Выбрать файл
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Video preview */}
          <div className="relative rounded-xl overflow-hidden bg-secondary/50">
            <video
              ref={videoRef}
              src={videoUrl}
              className="w-full h-48 object-contain"
              onEnded={() => setIsPlaying(false)}
            />
            
            {/* Play overlay */}
            <button
              onClick={togglePlay}
              className="absolute inset-0 flex items-center justify-center bg-background/20 hover:bg-background/30 transition-colors"
            >
              <div className="h-12 w-12 rounded-full bg-background/80 flex items-center justify-center">
                {isPlaying ? (
                  <Pause className="h-6 w-6 text-foreground" />
                ) : (
                  <Play className="h-6 w-6 text-foreground ml-1" />
                )}
              </div>
            </button>
          </div>
          
          {/* File info */}
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground truncate max-w-[200px]">
              {videoName}
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="gap-1 text-destructive hover:text-destructive"
              onClick={clearVideo}
            >
              <X className="h-4 w-4" />
              Удалить
            </Button>
          </div>
          
          {/* Analyze button */}
          <Button
            className="w-full gap-2"
            onClick={handleAnalyze}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Анализ...
              </>
            ) : (
              <>
                <Video className="h-4 w-4" />
                Распознать движение
              </>
            )}
          </Button>
          
          <p className="text-xs text-muted-foreground text-center">
            ИИ проанализирует видео и определит траекторию движения
          </p>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
