import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Play, Pause, RotateCcw, Download, Loader2 } from "lucide-react";

export type TrajPoint = number[] | { x: number; y: number; z?: number };

function trajXY(pt: TrajPoint): [number, number] {
  if (Array.isArray(pt)) {
    return [Number(pt[0]), Number(pt[1])];
  }
  return [pt.x, pt.y];
}

interface RealtimeTrajectoryVideoProps {
  videoUrl?: string;
  trajectory?: TrajPoint[];
  turnPoints?: Array<{ trajectory_index?: number; turn_type?: string; [key: string]: unknown }>;
  scaleFactor?: number;
}

const RealtimeTrajectoryVideo: React.FC<RealtimeTrajectoryVideoProps> = ({
  videoUrl,
  trajectory = [],
  turnPoints = [],
  scaleFactor = 12.306
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTrajectoryPoint, setCurrentTrajectoryPoint] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  // Draw trajectory on canvas overlay
  const drawTrajectory = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || trajectory.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate video display size to match CSS
    const videoRect = video.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    const scaleX = canvasRect.width / videoRect.width;
    const scaleY = canvasRect.height / videoRect.height;

    // Draw trajectory path
    const pointsToDraw = Math.min(currentTrajectoryPoint + 1, trajectory.length);
    
    if (trajectory.length > 1 && pointsToDraw >= 2) {
      ctx.beginPath();
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Calculate trajectory bounds for auto-scaling
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (let i = 0; i < pointsToDraw; i++) {
        const pt = trajectory[i];
        const [x, y] = trajXY(pt);
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
      
      const trajWidth = maxX - minX || 1;
      const trajHeight = maxY - minY || 1;
      const padding = 50;
      
      // Scale to fit canvas with padding
      const drawScaleX = (canvas.width - padding * 2) / trajWidth;
      const drawScaleY = (canvas.height - padding * 2) / trajHeight;
      const drawScale = Math.min(drawScaleX, drawScaleY, 100); // Max scale limit
      
      const offsetX = (canvas.width - trajWidth * drawScale) / 2 - minX * drawScale;
      const offsetY = (canvas.height - trajHeight * drawScale) / 2 - minY * drawScale;

      for (let i = 0; i < pointsToDraw; i++) {
        const point = trajectory[i];
        const [xCoord, yCoord] = trajXY(point);
        
        // Scale trajectory points to canvas size
        const x = xCoord * drawScale + offsetX;
        const y = canvas.height - (yCoord * drawScale + offsetY);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Draw current position dot
      if (pointsToDraw > 0) {
        const currentPoint = trajectory[pointsToDraw - 1];
        const [xCoord, yCoord] = trajXY(currentPoint);
        
        const x = xCoord * drawScale + offsetX;
        const y = canvas.height - (yCoord * drawScale + offsetY);
        
        ctx.beginPath();
        ctx.fillStyle = '#ef4444';
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw start point
        const startPoint = trajectory[0];
        const [startXCoord, startYCoord] = trajXY(startPoint);
        const startX = startXCoord * drawScale + offsetX;
        const startY = canvas.height - (startYCoord * drawScale + offsetY);
        
        ctx.beginPath();
        ctx.fillStyle = '#10b981';
        ctx.arc(startX, startY, 8, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw turn points
      turnPoints.forEach((turn) => {
        const ti = turn.trajectory_index ?? 0;
        if (ti < pointsToDraw) {
          const turnPoint = trajectory[ti];
          if (turnPoint) {
            const [tx, ty] = trajXY(turnPoint);
            const x = (tx / scaleFactor) * scaleX + canvas.width / 2;
            const y = canvas.height - (ty / scaleFactor) * scaleY - 50;
            
            ctx.beginPath();
            ctx.fillStyle = turn.turn_type === 'left' ? '#f59e0b' : '#8b5cf6';
            ctx.arc(x, y, 10, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw turn angle
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`${String(turn.angle_degrees ?? "")}°`, x, y - 15);
          }
        }
      });
    }
  }, [trajectory, turnPoints, currentTrajectoryPoint, scaleFactor]);

  // Update trajectory point based on video time
  useEffect(() => {
    if (!videoRef.current || trajectory.length === 0) return;

    const video = videoRef.current;
    const duration = video.duration || 1;
    const currentTime = video.currentTime;
    const progress = currentTime / duration;
    const pointIndex = Math.floor(progress * trajectory.length);
    
    setCurrentTrajectoryPoint(Math.min(pointIndex, trajectory.length - 1));
  }, [currentTime, trajectory.length]);

  // Redraw when trajectory or time changes
  useEffect(() => {
    drawTrajectory();
  }, [drawTrajectory, currentTrajectoryPoint, duration]);

  // Handle video time update
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
      
      // Обновляем позицию траектории на основе времени видео
      if (duration > 0 && trajectory.length > 0) {
        const video = videoRef.current;
        const timeRatio = video.currentTime / video.duration;
        const pointIndex = Math.floor(timeRatio * trajectory.length);
        setCurrentTrajectoryPoint(Math.min(pointIndex, trajectory.length - 1));
      }
    }
  };

  // Handle video loaded
  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      drawTrajectory();
    }
  };

  // Play/Pause toggle
  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Restart video
  const handleRestart = () => {
    const video = videoRef.current;
    if (!video) return;
    
    video.currentTime = 0;
    setCurrentTime(0);
    setCurrentTrajectoryPoint(0);
    drawTrajectory();
  };

  // Handle video ended
  const handleEnded = () => {
    setIsPlaying(false);
  };

  // Download trajectory as image
  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement('a');
    link.download = 'trajectory-video.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  if (!videoUrl) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Видео с траекторией</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 bg-muted rounded-lg">
            <p className="text-muted-foreground">Видео не выбрано</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle>Видео с траекторией</CardTitle>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRestart}
              disabled={!videoUrl}
            >
              <RotateCcw className="h-4 w-4 mr-1" />
              Сброс
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
              disabled={!trajectory || trajectory.length === 0}
            >
              <Download className="h-4 w-4 mr-1" />
              Скачать кадр
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Video with canvas overlay */}
        <div className="relative rounded-lg overflow-hidden bg-black">
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full"
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onEnded={handleEnded}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
          />
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          <Button
            variant="default"
            size="sm"
            onClick={togglePlay}
            disabled={!videoUrl}
          >
            {isPlaying ? (
              <>
                <Pause className="h-4 w-4 mr-1" />
                Пауза
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-1" />
                Воспроизвести
              </>
            )}
          </Button>

          {/* Progress bar */}
          <div className="flex-1">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>{Math.floor(currentTime)}с</span>
              <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all"
                  style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
                />
              </div>
              <span>{Math.floor(duration)}с</span>
            </div>
          </div>

          {/* Stats */}
          <div className="text-sm text-muted-foreground">
            Точка: {currentTrajectoryPoint + 1} / {trajectory.length}
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4 text-sm">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span>Старт</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span>Текущая позиция</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span>Траектория</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-amber-500" />
            <span>Поворот</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default RealtimeTrajectoryVideo;
