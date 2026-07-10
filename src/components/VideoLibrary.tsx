import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Video, Play, FileVideo, Calendar, HardDrive, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { apiClient, VideoListItem } from "@/lib/api";

interface VideoLibraryProps {
  onVideoSelected?: (video: VideoListItem) => void;
  onAnalysisLoaded?: (
    trajectory: number[][],
    turnPoints: Record<string, unknown>[],
    stats: Record<string, unknown> | undefined
  ) => void;
}

const VideoLibrary = ({ onVideoSelected, onAnalysisLoaded }: VideoLibraryProps) => {
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingAnalysis, setLoadingAnalysis] = useState<string | null>(null);

  useEffect(() => {
    loadVideosList();
  }, []);

  const loadVideosList = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getVideosList();
      if (response.success) {
        setVideos(response.videos);
      } else {
        toast.error("Не удалось загрузить список видео");
      }
    } catch (error) {
      console.error("Error loading videos:", error);
      toast.error("Ошибка при загрузке списка видео");
    } finally {
      setLoading(false);
    }
  };

  const handleVideoSelect = async (video: VideoListItem) => {
    onVideoSelected?.(video);
  };

  const handleLoadAnalysis = async (video: VideoListItem) => {
    try {
      setLoadingAnalysis(video.video_id);
      const result = await apiClient.getVideoAnalysis(video.video_id);

      if (!result.success || !result.data) {
        toast.error("Не удалось загрузить анализ видео");
        return;
      }
      const data = result.data as { trajectory?: unknown; map_trajectory?: unknown; turn_points?: unknown; map_turn_points?: unknown; processing_stats?: unknown };
      const trajectory = data.map_trajectory ?? data.trajectory;
      const turnPoints = data.map_turn_points ?? data.turn_points ?? [];
      const stats = data.processing_stats;

      if (!trajectory || !Array.isArray(trajectory) || trajectory.length === 0) {
        toast.error("Траектория пуста или отсутствует");
        return;
      }

      if (onAnalysisLoaded) {
        onAnalysisLoaded(
          trajectory as number[][],
          Array.isArray(turnPoints) ? (turnPoints as Record<string, unknown>[]) : [],
          stats as Record<string, unknown> | undefined
        );
        toast.success(`Анализ "${video.filename}" загружен`);
      }
    } catch (error) {
      console.error("Error loading analysis:", error);
      toast.error(error instanceof Error ? error.message : "Ошибка при загрузке анализа");
    } finally {
      setLoadingAnalysis(null);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    if (!dateString) return '—';
    const ts = parseFloat(dateString);
    const date = !isNaN(ts) ? new Date(ts * 1000) : new Date(dateString);
    if (isNaN(date.getTime())) return '—';
    return date.toLocaleString('ru-RU', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          Загрузка списка видео...
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Video className="h-5 w-5" />
          Библиотека видео
        </CardTitle>
      </CardHeader>
      <CardContent>
        {videos.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <FileVideo className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Нет загруженных видео</p>
            <p className="text-sm">Загрузите видео для анализа</p>
          </div>
        ) : (
          <div className="space-y-4">
            {videos.map((video) => (
              <div key={video.video_id} className="border rounded-lg p-4 hover:bg-secondary/50 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <FileVideo className="h-4 w-4 text-primary" />
                      <h3 className="font-medium truncate">{video.filename}</h3>
                      {video.stabilized && (
                        <Badge variant="secondary" className="text-xs">
                          Стабилизировано
                        </Badge>
                      )}
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-muted-foreground mb-3">
                      <div className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {formatDate(video.uploaded_at)}
                      </div>
                      <div className="flex items-center gap-1">
                        <HardDrive className="h-3 w-3" />
                        {formatFileSize(video.file_size)}
                      </div>
                      <div>Масштаб: {video.scale_factor}</div>
                      <div>FPS: {video.has_analysis ? "Проанализировано" : "Ожидает анализа"}</div>
                    </div>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleVideoSelect(video)}
                    className="gap-1"
                  >
                    <Play className="h-3 w-3" />
                    Просмотреть видео
                  </Button>

                  {video.has_analysis && (
                    <Button
                      size="sm"
                      onClick={() => handleLoadAnalysis(video)}
                      disabled={loadingAnalysis === video.video_id}
                      className="gap-1"
                    >
                      {loadingAnalysis === video.video_id ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Video className="h-3 w-3" />
                      )}
                      Загрузить анализ
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="mt-4 pt-4 border-t">
          <Button
            variant="outline"
            onClick={loadVideosList}
            className="w-full"
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Обновление...
              </>
            ) : (
              "Обновить список"
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default VideoLibrary;
