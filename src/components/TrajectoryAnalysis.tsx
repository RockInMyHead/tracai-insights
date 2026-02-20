import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Video, MapPin, Activity, Clock, Navigation, Loader2, User, X, Plus } from "lucide-react";
import { toast } from "sonner";
import { apiClient, VideoAnalysisResult } from "@/lib/api";

// Интерфейс для видео с владельцем
interface VideoWithOwner {
  id: string;
  file: File;
  ownerName: string;
  analysisResult?: any;
  isAnalyzing?: boolean;
  uploadProgress?: number;
  color: string;
  uploadedAt?: number;
}

interface Employee {
  name: string;
  color: string;
}

interface TrajectoryData {
  trajectory: number[][];
  turnPoints: any[];
  ownerName: string;
  color: string;
}

interface TrajectoryAnalysisProps {
  onTrajectoryAnalyzed?: (trajectory: number[][], turnPoints: any[], stats: any, trajectories?: TrajectoryData[]) => void;
}

const TrajectoryAnalysis = ({ onTrajectoryAnalyzed }: TrajectoryAnalysisProps) => {
  const [videos, setVideos] = useState<VideoWithOwner[]>([]);
  const [currentOwnerName, setCurrentOwnerName] = useState('');
  const [scaleFactor, setScaleFactor] = useState<number>(12.306);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [stabilizationEnabled, setStabilizationEnabled] = useState<boolean>(true);
  const [floorPlan, setFloorPlan] = useState<string | null>(null);
  const [floorPlanFile, setFloorPlanFile] = useState<File | null>(null);
  const [referencePoint, setReferencePoint] = useState(null);
  const [existingOwners, setExistingOwners] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const floorPlanInputRef = useRef<HTMLInputElement>(null);

  // Цвета для разных пользователей
  const userColors = [
    '#3b82f6', // blue
    '#ef4444', // red
    '#10b981', // emerald
    '#f59e0b', // amber
    '#8b5cf6', // violet
    '#06b6d4', // cyan
    '#84cc16', // lime
    '#f97316', // orange
    '#ec4899', // pink
    '#6b7280', // gray
  ];

  // Для обратной совместимости с отображением результатов
  const firstAnalyzedVideo = videos.find(v => v.analysisResult);
  const analysisResult = firstAnalyzedVideo ? {
    data: firstAnalyzedVideo.analysisResult
  } : null;

  // Загрузка плана из localStorage при инициализации
  useEffect(() => {
    const savedFloorPlan = localStorage.getItem('floorPlan');
    if (savedFloorPlan) {
      if (savedFloorPlan.length > 1_500_000) {
        localStorage.removeItem('floorPlan');
      } else {
        setFloorPlan(savedFloorPlan);
      }
    }
  }, []);

  // Загрузка настройки стабилизации и имен сотрудников
  useEffect(() => {
    const savedStabilization = localStorage.getItem('stabilizationEnabled');
    if (savedStabilization !== null) {
      setStabilizationEnabled(JSON.parse(savedStabilization));
    }

    const savedOwners = localStorage.getItem('trackai_owners');
    if (savedOwners) {
      try {
        setExistingOwners(JSON.parse(savedOwners));
      } catch (e) {
        console.error("Failed to load owners", e);
      }
    }
  }, []);

  const handleStabilizationToggle = () => {
    const newValue = !stabilizationEnabled;
    setStabilizationEnabled(newValue);
    localStorage.setItem('stabilizationEnabled', JSON.stringify(newValue));
  };

  const handleFloorPlanUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        toast.error("Пожалуйста, загрузите изображение плана");
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result as string;
        setFloorPlan(result);
        setFloorPlanFile(file);
        try {
          localStorage.setItem('floorPlan', result);
        } catch (storageError) {
          console.warn("Could not save floor plan to localStorage (quota exceeded)", storageError);
          // We continue anyway, it's just not saved between refreshes
        }
        toast.success(`План "${file.name}" загружен`);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFloorPlanRemove = () => {
    setFloorPlan(null);
    setFloorPlanFile(null);
    localStorage.removeItem('floorPlan');
    toast.info("План помещения удален");
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const newFiles: File[] = [];
      const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.3gp'];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const isVideoMime = file.type.startsWith("video/");
        const hasVideoExt = videoExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

        if (isVideoMime || hasVideoExt) {
          // Проверяем, нет ли уже такого файла в списке (по имени и размеру)
          const isDuplicate = selectedFiles.some(f => f.name === file.name && f.size === file.size);
          if (!isDuplicate) {
            newFiles.push(file);
          }
        } else {
          toast.error(`Файл ${file.name} не похож на видео. Если это видео, попробуйте переименовать или сменить формат.`);
        }
      }

      if (newFiles.length > 0) {
        setSelectedFiles(prev => [...prev, ...newFiles]);
        toast.info(`Добавлено ${newFiles.length} файл(ов) к выбору`);
      }
    }

    // Очищаем значение инпута, чтобы можно было выбрать те же файлы снова
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeSelectedFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const clearSelectedFiles = () => {
    setSelectedFiles([]);
  };

  const addVideoToList = () => {
    if (selectedFiles.length === 0) {
      toast.error("Сначала выберите один или несколько видео файлов");
      return;
    }

    if (!currentOwnerName.trim()) {
      toast.error("Введите имя или выберите сотрудника");
      return;
    }

    const ownerName = currentOwnerName.trim();

    // Сохраняем имя в список существующих
    if (!existingOwners.includes(ownerName)) {
      const newOwners = [...existingOwners, ownerName];
      setExistingOwners(newOwners);
      localStorage.setItem('trackai_owners', JSON.stringify(newOwners));
    }

    const timestamp = Date.now();

    // Ищем существующий цвет для этого сотрудника
    const existingOwner = videos.find(v => v.ownerName === ownerName);
    const ownerColor = existingOwner ? existingOwner.color : userColors[Array.from(new Set(videos.map(v => v.ownerName))).length % userColors.length];

    const newVideos: VideoWithOwner[] = selectedFiles.map((file, index) => ({
      id: `${timestamp}-${index}`,
      file: file,
      ownerName: ownerName,
      color: ownerColor,
      isAnalyzing: false,
      uploadedAt: timestamp
    }));

    setVideos(prev => [...prev, ...newVideos]);
    setCurrentOwnerName('');
    setSelectedFiles([]);

    // Очищаем input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    toast.success(`Добавлено ${newVideos.length} видео для сотрудника ${ownerName}`);
  };

  const removeVideo = (videoId: string) => {
    setVideos(prev => prev.filter(v => v.id !== videoId));
    toast.info("Видео удалено из списка");
  };

  const analyzeAllVideos = async () => {
    if (videos.length === 0) {
      console.log('❌ Ошибка: Нет видео для анализа');
      toast.error("Добавьте хотя бы одно видео для анализа");
      return;
    }

    console.log(`🚀 Начало пакетного анализа ${videos.length} видео`);
    console.log(`📏 Коэффициент масштаба: ${scaleFactor}`);
    console.log(`🎥 Стабилизация: ${stabilizationEnabled ? 'включена' : 'отключена'}`);

    setIsAnalyzing(true);
    setAnalysisProgress(0);

    try {
      const startTime = Date.now();

      // Use the current `videos` state for analysis
      const videosToAnalyze = [...videos];

      // Запускаем анализ всех видео параллельно (одновременно)
      const analysisPromises = videosToAnalyze.map(async (video, index) => {
        if (video.analysisResult) {
          // If already analyzed, just return it.
          // We still update progress for already analyzed videos to reflect completion.
          setAnalysisProgress(prev => prev + (100 / videosToAnalyze.length));
          return video;
        }

        // Обновляем статус: начало анализа
        setVideos(prev => prev.map(v =>
          v.id === video.id ? { ...v, isAnalyzing: true } : v
        ));
        setCurrentStep(`Анализ видео ${video.ownerName} (${video.file.name})...`);

        try {
          // Start polling for progress (будет использовать uploadedVideoId после загрузки)
          let processingId: string | null = null;
          const pollInterval = setInterval(async () => {
            if (!processingId) return;
            try {
              const status = await apiClient.getProcessingStatus(processingId);
              if (status && status.progress > 0) {
                if (status.message) {
                  setCurrentStep(`[${video.ownerName}] ${status.message} (${status.progress}%)`);
                }
              }
            } catch (e) {
              // ignore polling errors
            }
          }, 1000);

          // Шаг 1: Загружаем видео на сервер (отдельно, таймаут 2 часа)
          setCurrentStep(`Загрузка ${video.file.name} на сервер...`);
          const uploadResult = await apiClient.uploadVideo(
            video.file,
            (progress) => {
              setVideos(prev => prev.map(v =>
                v.id === video.id ? { ...v, uploadProgress: progress } : v
              ));
              setCurrentStep(`Загрузка ${video.file.name}: ${progress.toFixed(0)}%`);
            }
          );

          const uploadedVideoId = uploadResult.video_id;
          processingId = uploadedVideoId;

          // Шаг 2: Запускаем анализ уже загруженного видео
          let result = await apiClient.analyzeVideoById(
            uploadedVideoId,
            scaleFactor,
            stabilizationEnabled,
            video.file.name
          );

          // Если видео поставлено в очередь, ждем завершения через поллинг
          if (result.status === 'queued') {
            const maxAttempts = 1800; // 60 минут (для больших AVI конвертация может занять 30+ мин)
            let attempts = 0;

            while (attempts < maxAttempts) {
              const status = await apiClient.getProcessingStatus(uploadedVideoId);

              if (status.status === 'completed' && status.result) {
                result = { success: true, data: status.result, message: "Success" };
                break;
              } else if (status.status === 'error') {
                throw new Error(status.message || "Ошибка при обработке на сервере");
              }

              // Обновляем прогресс из статуса
              if (status.progress > 0) {
                setCurrentStep(`[${video.ownerName}] ${status.message || 'Обработка'} (${status.progress}%)`);
              }

              attempts++;
              await new Promise(resolve => setTimeout(resolve, 2000)); // Ждем 2 секунды перед следующим опросом
            }

            if (attempts >= maxAttempts) {
              throw new Error("Таймаут ожидания анализа (60 минут). Для больших AVI конвертация может занять 30+ мин.");
            }
          }

          clearInterval(pollInterval);

          const analyzedVideo = {
            ...video,
            analysisResult: result.data,
            isAnalyzing: false
          };

          // Обновляем состояние конкретного видео
          setVideos(prev => prev.map(v =>
            v.id === video.id ? analyzedVideo : v
          ));
          setAnalysisProgress(prev => prev + (100 / videosToAnalyze.length));

          return analyzedVideo;
        } catch (err: any) {
          console.error(`Error analyzing video ${video.id}:`, err);
          toast.error(`Ошибка при анализе ${video.file.name}: ${err.message || 'Неизвестная ошибка'}`);

          const errorVideo = { ...video, isAnalyzing: false };
          setVideos(prev => prev.map(v =>
            v.id === video.id ? errorVideo : v
          ));
          setAnalysisProgress(prev => prev + (100 / videosToAnalyze.length));
          return errorVideo;
        }
      });

      // Ждем завершения всех запросов
      const finalizedVideos = await Promise.all(analysisPromises);

      // Обновляем финальный список
      setVideos(finalizedVideos);
      setAnalysisProgress(100);
      setCurrentStep('Анализ всех видео завершен');

      const totalTime = (Date.now() - startTime) / 1000;
      console.log(`\n🎉 Пакетный анализ завершен!`);
      console.log(`📊 Статистика:`);
      console.log(`   • Обработано видео: ${videosToAnalyze.length}`);
      console.log(`   • Общее время: ${totalTime.toFixed(1)} сек`);
      console.log(`   • Среднее время на видео: ${(totalTime / finalizedVideos.length).toFixed(1)} сек`);

      const convertTrajectory = (traj: any) => {
        if (!traj) return [];

        // Если это массив массивов [[x,y,z], ...]
        if (Array.isArray(traj) && traj.length > 0) {
          if (Array.isArray(traj[0])) {
            return traj.map((point: any) => ({
              x: Number(point[0]) || 0,
              y: Number(point[1]) || 0,
              z: Number(point[2]) || 0
            }));
          }
          // Если уже объекты {x,y,z}
          if (typeof traj[0] === 'object' && 'x' in traj[0]) {
            return traj;
          }
        }
        return [];
      };

      const finalAnalyzedVideos = finalizedVideos.filter(v => v.analysisResult);

      if (finalAnalyzedVideos.length > 0 && onTrajectoryAnalyzed) {
        // Подготавливаем данные для всех траекторий
        const trajectoriesData = finalAnalyzedVideos.map(video => ({
          trajectory: convertTrajectory(video.analysisResult.trajectory),
          turnPoints: video.analysisResult.turn_points || [],
          ownerName: video.ownerName,
          color: video.color
        }));

        // Передаем результаты
        onTrajectoryAnalyzed(
          trajectoriesData[0].trajectory,
          trajectoriesData[0].turnPoints,
          finalAnalyzedVideos[0].analysisResult.processing_stats || {},
          trajectoriesData
        );
      }

      toast.success(`Анализ ${videos.length} видео завершен успешно!`);

    } catch (error) {
      console.error("Analysis error:", error);
      toast.error("Ошибка при анализе видео");
    } finally {
      setIsAnalyzing(false);
      setTimeout(() => {
        setAnalysisProgress(0);
        setCurrentStep('');
      }, 2000);
    }
  };

  const handleLoadSampleData = async () => {
    console.log('🧪 Загрузка тестовых данных для демонстрации...');

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setCurrentStep('Подготовка тестовых данных...');

    try {
      console.log('📦 Этап 1: Подготовка тестовых данных...');
      setCurrentStep('Загрузка образцов анализа...');
      await new Promise(resolve => setTimeout(resolve, 300));
      setAnalysisProgress(50);
      console.log('✅ Тестовые данные подготовлены');

      console.log('🔄 Этап 2: Обработка результатов...');
      setCurrentStep('Обработка результатов...');
      await new Promise(resolve => setTimeout(resolve, 500));
      setAnalysisProgress(90);
      console.log('✅ Результаты обработаны');

      console.log('🌐 Этап 3: Загрузка с сервера...');
      setCurrentStep('Завершение загрузки...');
      const startTime = Date.now();

      const apiUrl = import.meta.env.VITE_API_URL || '';
      console.log(`🌐 Запрос к: ${apiUrl || window.location.origin}/api/sample-data`);
      const response = await fetch(`${apiUrl}/api/sample-data`);
      if (!response.ok) {
        console.log(`❌ Ошибка загрузки тестовых данных: ${response.status} ${response.statusText}`);
        throw new Error(`Failed to load sample data: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      const endTime = Date.now();
      const loadTime = (endTime - startTime) / 1000;

      setAnalysisProgress(100);
      console.log(`✅ Тестовые данные загружены за ${loadTime.toFixed(1)} сек`);
      console.log(`📊 Содержимое:`);
      console.log(`   • Точек траектории: ${result.data.trajectory?.length || 0}`);
      console.log(`   • Поворотов: ${result.data.turn_points?.length || 0}`);

      toast.success("Тестовые данные загружены!");

      if (onTrajectoryAnalyzed) {
        // Преобразуем данные в формат множественных траекторий для TrajectoryMap
        const trajectoryData = [{
          trajectory: result.data.trajectory,
          turnPoints: result.data.turn_points,
          ownerName: 'Тестовые данные',
          color: '#10b981'
        }];

        onTrajectoryAnalyzed(
          result.data.trajectory,
          result.data.turn_points,
          result.data.processing_stats,
          trajectoryData
        );
      }
    } catch (error) {
      console.error("❌ Ошибка при загрузке тестовых данных:", error);
      toast.error("Ошибка при загрузке тестовых данных.");
    } finally {
      setTimeout(() => {
        setIsAnalyzing(false);
        setAnalysisProgress(0);
        setCurrentStep('');
        console.log('🔄 Сброс состояния загрузки тестовых данных');
      }, 500);
    }
  };

  const clearAnalysis = () => {
    setVideos([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Video className="h-5 w-5" />
          Анализ траектории движения
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Video Management */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label className="text-base font-medium">Загрузка видео для сотрудников</Label>
            <Badge variant="secondary">{videos.length} видео</Badge>
          </div>

          {/* Add new video */}
          <div className="border-2 border-dashed border-border rounded-lg p-5 bg-secondary/10 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="owner-name" className="text-sm font-semibold">Сотрудник</Label>
                <div className="flex flex-col gap-2">
                  <Input
                    id="owner-name"
                    type="text"
                    placeholder="Введите имя или выберите из списка..."
                    value={currentOwnerName}
                    onChange={(e) => setCurrentOwnerName(e.target.value)}
                    className="mt-1"
                    onKeyPress={(e) => e.key === 'Enter' && addVideoToList()}
                  />
                  {existingOwners.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {existingOwners.slice(0, 5).map(name => (
                        <Badge
                          key={name}
                          variant="outline"
                          className="cursor-pointer hover:bg-primary/20 transition-colors"
                          onClick={() => setCurrentOwnerName(name)}
                        >
                          {name}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="video-file" className="text-sm font-semibold">Видео файлы (можно несколько)</Label>
                <Input
                  ref={fileInputRef}
                  id="video-file"
                  type="file"
                  accept="video/*"
                  multiple
                  onChange={handleFileSelect}
                  className="mt-1 bg-background"
                />
                {selectedFiles.length > 0 && (
                  <div className="mt-2 p-3 bg-primary/5 rounded-lg border border-primary/20 space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-xs font-bold text-primary">Готовы к добавлению ({selectedFiles.length}):</p>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-[10px] text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={clearSelectedFiles}
                      >
                        Очистить всё
                      </Button>
                    </div>
                    <ul className="text-[10px] space-y-1.5 max-h-40 overflow-y-auto pr-1">
                      {selectedFiles.map((f, i) => (
                        <li key={`${f.name}-${i}`} className="flex items-center justify-between group bg-background/50 p-1.5 rounded">
                          <div className="flex items-center gap-2 overflow-hidden mr-2">
                            <Video className="h-3 w-3 text-primary/50 flex-shrink-0" />
                            <span className="truncate">{f.name}</span>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            <span className="text-muted-foreground">{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                            <button
                              onClick={() => removeSelectedFile(i)}
                              className="text-muted-foreground hover:text-destructive transition-colors"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            <Button
              onClick={addVideoToList}
              disabled={!currentOwnerName.trim()}
              className="w-full bg-primary/90 hover:bg-primary"
            >
              <Plus className="h-4 w-4 mr-2" />
              Добавить видео для этого сотрудника
            </Button>
          </div>

          {/* Video list grouped by owner */}
          {videos.length > 0 && (
            <div className="space-y-4 mt-6">
              <Label className="text-sm font-bold">Очередь обработки по сотрудникам:</Label>
              {Array.from(new Set(videos.map(v => v.ownerName))).map(ownerName => {
                const ownerVideos = videos.filter(v => v.ownerName === ownerName);
                const ownerColor = ownerVideos[0]?.color;

                return (
                  <div key={ownerName} className="p-4 rounded-xl border border-border/50 bg-secondary/20 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded-full border border-white" style={{ backgroundColor: ownerColor }} />
                        <span className="font-bold text-lg">{ownerName}</span>
                        <Badge variant="secondary">{ownerVideos.length} видео</Badge>
                      </div>
                    </div>

                    <div className="space-y-2">
                      {ownerVideos.map((video) => (
                        <div
                          key={video.id}
                          className="flex items-center justify-between p-2 pl-3 bg-background/50 rounded-lg border border-border/30 group"
                        >
                          <div className="flex items-center gap-3 overflow-hidden flex-1">
                            <Video className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                            <div className="flex flex-col overflow-hidden min-w-0">
                              <span className="text-sm font-medium truncate">{video.file.name}</span>
                              <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                                <span>{(video.file.size / 1024 / 1024).toFixed(1)} MB</span>
                                {video.uploadedAt && (
                                  <>
                                    <span>•</span>
                                    <span>{new Date(video.uploadedAt).toLocaleString('ru-RU', {
                                      day: '2-digit',
                                      month: '2-digit',
                                      year: 'numeric',
                                      hour: '2-digit',
                                      minute: '2-digit'
                                    })}</span>
                                  </>
                                )}
                              </div>
                            </div>
                            {video.isAnalyzing && (
                              <Badge variant="secondary" className="gap-1 animate-pulse flex-shrink-0">
                                <Loader2 className="h-3 w-3 animate-spin" />
                                {video.uploadProgress !== undefined && video.uploadProgress < 100
                                  ? `Загрузка ${video.uploadProgress.toFixed(0)}%`
                                  : 'Обработка...'}
                              </Badge>
                            )}
                            {video.analysisResult && (
                              <Badge variant="default" className="gap-1 bg-green-500/20 text-green-500 border-green-500/20 flex-shrink-0">
                                ✓ Готово
                              </Badge>
                            )}
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => removeVideo(video.id)}
                            disabled={isAnalyzing}
                            className="h-8 w-8 text-destructive opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Scale factor */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="scale-factor">Коэффициент масштаба</Label>
            <span className="text-sm font-medium text-primary bg-primary/10 px-2 py-1 rounded">
              {scaleFactor.toFixed(1)}
            </span>
          </div>

          <div className="space-y-3">
            <Slider
              id="scale-factor"
              min={1}
              max={50}
              step={0.1}
              value={[scaleFactor]}
              onValueChange={(value) => setScaleFactor(value[0])}
              className="w-full"
            />

            <div className="flex justify-between text-xs text-muted-foreground">
              <span>1.0</span>
              <span>25.0</span>
              <span>50.0</span>
            </div>
          </div>

          <div className="flex gap-2">
            <Input
              type="number"
              step="0.1"
              min="1"
              max="50"
              value={scaleFactor}
              onChange={(e) => setScaleFactor(parseFloat(e.target.value) || 1)}
              className="w-24"
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setScaleFactor(12.306)}
            >
              Сбросить
            </Button>
          </div>

          <p className="text-sm text-muted-foreground">
            Коэффициент для перевода пикселей в метры. Влияет на точность расчета расстояний.
          </p>
        </div>

        {/* Progress bar */}
        {isAnalyzing && (
          <div className="space-y-4 p-6 bg-gradient-to-r from-secondary/50 to-secondary/30 rounded-xl border border-primary/10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Loader2 className="h-5 w-5 animate-spin text-primary" />
                  <div className="absolute inset-0 h-5 w-5 rounded-full bg-primary/20 animate-ping" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold">Обработка видео</h3>
                  <p className="text-xs text-muted-foreground">ИИ анализирует траекторию движения</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-lg font-bold text-primary">{analysisProgress}%</div>
                <div className="text-xs text-muted-foreground">завершено</div>
              </div>
            </div>

            <Progress value={analysisProgress} className="w-full h-2" />

            <div className="flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4 text-primary" />
              <span className="font-medium">{currentStep}</span>
            </div>

            {/* Progress steps */}
            <div className={`grid gap-2 text-xs ${stabilizationEnabled ? 'grid-cols-4' : 'grid-cols-3'}`}>
              <div className={`text-center p-2 rounded ${analysisProgress >= 20 ? 'bg-primary/10 text-primary' : 'text-muted-foreground'}`}>
                Загрузка
              </div>
              {stabilizationEnabled && (
                <div className={`text-center p-2 rounded ${analysisProgress >= 50 ? 'bg-primary/10 text-primary' : 'text-muted-foreground'}`}>
                  Стабилизация
                </div>
              )}
              <div className={`text-center p-2 rounded ${analysisProgress >= (stabilizationEnabled ? 75 : 70) ? 'bg-primary/10 text-primary' : 'text-muted-foreground'}`}>
                SLAM анализ
              </div>
              <div className={`text-center p-2 rounded ${analysisProgress >= 100 ? 'bg-primary/10 text-primary' : 'text-muted-foreground'}`}>
                Готово
              </div>
            </div>
          </div>
        )}

        {/* Stabilization toggle */}
        <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg border">
          <div className="flex flex-col">
            <Label htmlFor="stabilization-toggle" className="text-sm font-medium">
              Программная стабилизация видео
            </Label>
            <p className="text-xs text-muted-foreground">
              Удаляет тряску камеры для более точного анализа
            </p>
          </div>
          <Switch
            id="stabilization-toggle"
            checked={stabilizationEnabled}
            onCheckedChange={handleStabilizationToggle}
          />
        </div>

        {/* Analyze buttons */}
        <div className="space-y-3">
          <Button
            onClick={analyzeAllVideos}
            disabled={videos.length === 0 || isAnalyzing}
            className="w-full"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Анализ в процессе...
              </>
            ) : (
              <>
                <Activity className="h-4 w-4 mr-2" />
                {stabilizationEnabled ? 'Анализ со стабилизацией' : 'Анализ без стабилизации'}
                {videos.length > 0 && ` (${videos.length} видео)`}
              </>
            )}
          </Button>

          <Button
            onClick={handleLoadSampleData}
            disabled={isAnalyzing}
            variant="outline"
            className="w-full"
          >
            <Video className="h-4 w-4 mr-2" />
            Загрузить тестовые данные
          </Button>
        </div>

        <div className="space-y-2 mt-3">
          <p className="text-xs text-muted-foreground text-center">
            🎥 Автоматическая стабилизация + SLAM анализ траектории
          </p>
          <p className="text-xs text-muted-foreground text-center">
            Тестовые данные позволят ознакомиться с функционалом без загрузки видео
          </p>
        </div>

        {/* Analysis results */}
        {analysisResult && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Результаты анализа</h3>

            {/* Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-secondary rounded-lg">
                <MapPin className="h-5 w-5 mx-auto mb-1 text-primary" />
                <div className="text-2xl font-bold">
                  {analysisResult?.data?.trajectory_points || 0}
                </div>
                <div className="text-xs text-muted-foreground">Точек траектории</div>
              </div>

              <div className="text-center p-3 bg-secondary rounded-lg">
                <Navigation className="h-5 w-5 mx-auto mb-1 text-primary" />
                <div className="text-2xl font-bold">
                  {analysisResult?.data?.processing_stats?.estimated_distance?.toFixed(1) || "0.0"}
                </div>
                <div className="text-xs text-muted-foreground">Расстояние (м)</div>
              </div>

              <div className="text-center p-3 bg-secondary rounded-lg">
                <Activity className="h-5 w-5 mx-auto mb-1 text-primary" />
                <div className="text-2xl font-bold">
                  {analysisResult?.data?.turn_points?.length || 0}
                </div>
                <div className="text-xs text-muted-foreground">Поворотов</div>
              </div>

              <div className="text-center p-3 bg-secondary rounded-lg">
                <Clock className="h-5 w-5 mx-auto mb-1 text-primary" />
                <div className="text-2xl font-bold">
                  {analysisResult?.data?.total_processing_time?.toFixed(1) || "0.0"}
                </div>
                <div className="text-xs text-muted-foreground">Время анализа (с)</div>
              </div>
            </div>

            {/* Video info */}
            <div className="p-4 bg-secondary rounded-lg">
              <h4 className="font-medium mb-2">Информация о видео</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Разрешение:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.video_info?.width || 0}x{analysisResult?.data?.video_info?.height || 0}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">FPS:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.video_info?.fps?.toFixed(1) || "0.0"}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Длительность:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.video_info?.duration ? formatTime(analysisResult.data.video_info.duration) : "0:00"}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Кадров:</span>
                  <span className="ml-2 font-medium">
                    {analysisResult?.data?.video_info?.frame_count || 0}
                  </span>
                </div>
              </div>
            </div>

            {/* Turn points */}
            {analysisResult?.data?.turn_points && analysisResult.data.turn_points.length > 0 && (
              <div className="p-4 bg-secondary rounded-lg">
                <h4 className="font-medium mb-2">Обнаруженные повороты</h4>
                <div className="space-y-2">
                  {analysisResult.data.turn_points.map((turn: any, index: number) => (
                    <div key={index} className="flex items-center justify-between text-sm">
                      <span>Поворот {index + 1}</span>
                      <div className="flex gap-4">
                        <span>{turn.angle_degrees?.toFixed(1) || "0.0"}°</span>
                        <span className="capitalize">{turn.turn_type === 'left' ? 'Влево' : 'Вправо'}</span>
                        <span>({turn.position?.[0]?.toFixed(1) || '0.0'}, {turn.position?.[1]?.toFixed(1) || '0.0'})</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TrajectoryAnalysis;
