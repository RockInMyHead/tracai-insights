// API client for TrackAI backend
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

async function agentFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  return globalThis.fetch(input, init);
}

function isDesktopClient(): boolean {
  if (typeof window === 'undefined') return false;
  const trackai = (window as unknown as { trackai?: { isDesktop?: boolean } }).trackai;
  const queryDesktop = new URLSearchParams(window.location.search).get('desktop') === '1';
  if (trackai?.isDesktop === true || queryDesktop) {
    window.sessionStorage.setItem('trackai_desktop_client', '1');
    return true;
  }
  return window.sessionStorage.getItem('trackai_desktop_client') === '1';
}

function clientHeaders(extra?: Record<string, string>): Record<string, string> {
  return {
    ...(extra || {}),
    ...(isDesktopClient() ? { 'X-TrackAI-Client': 'desktop' } : {}),
  };
}

export interface VideoAnalysisResult {
  success: boolean;
  status?: string;
  video_id?: string;
  data?: {
    method: string;
    trajectory: number[][];
    map_trajectory?: number[][];
    turn_points: Array<{
      frame_index: number;
      trajectory_index: number;
      angle_degrees: number;
      position: number[];
      turn_type: string;
    }>;
    map_turn_points?: Array<{
      frame_index: number;
      trajectory_index: number;
      angle_degrees: number;
      position: number[];
      turn_type: string;
    }>;
    frame_count: number;
    trajectory_points: number;
    processing_stats: {
      estimated_distance: number;
      scale_factor: number;
      fps: number;
      turns_detected: number;
      avg_matches_per_frame?: number;
      ransac_failure_rate?: number;
      gating_failure_rate?: number;
      map_matching_applied?: boolean;
      map_trajectory_points?: number;
      map_auto_scale?: number;
    };
    map_metadata?: {
      plan_width: number;
      plan_height: number;
      grid_cell: number;
      auto_scale: number;
      source: string;
    };
    r3_camera_points?: number[][];  // Все позиции камер R³ для отрисовки облака точек
    r3_raw_camera_points?: number[][];
    raw_trajectory_3d?: number[][];
    plan_trajectory?: number[][];
    r3_source_frame_indices?: Array<number | null>;
    r3_source_timestamps_seconds?: Array<number | null>;
    r3_pose_confidence?: Array<number | null>;  // Уверенность каждой позиции
    r3_pose_graph?: Record<string, unknown>;
    r3_pose_graph_candidate?: Record<string, unknown>;
    total_processing_time: number;
    video_info: {
      width: number;
      height: number;
      fps: number;
      frame_count: number;
      duration: number;
    };
  };
  message: string;
}

export type R3TrajectorySource = "raw" | "robust_candidate";

export interface TrackingOptions {
  detect_interval?: number;
  turn_vote_threshold?: number;
  use_ml_roi?: boolean;
}

export interface MapContext {
  floor_plan_data?: string | null;
  drawn_plan?: unknown[] | null;
  reference_point?: { x: number; y: number } | null;
  direction_point?: { x: number; y: number } | null;
  batch_id?: string | null;
  batch_size?: number | null;
  employee_name?: string | null;
  client_source?: string | null;
  gpu_upload_url?: string | null;
}

export interface VideoListItem {
  video_id: string;
  filename: string;
  uploaded_at: string;
  file_size: number;
  scale_factor: number;
  stabilized: boolean;
  has_analysis: boolean;
}

export interface Plan {
  id?: number;
  name: string;
  data: unknown[];
  preview_svg?: string;
  created_at?: string;
}

export interface VideoListResponse {
  success: boolean;
  videos: VideoListItem[];
}

export interface TrackingTask {
  id: string;
  employee_name?: string;
  original_filename: string;
  status: string;
  created_at: string;
  map_context?: MapContext;
}


export interface ManualTrajectoryResponse {
  success: boolean;
  video_id: string;
  exists: boolean;
  trajectory?: number[][];
  turn_points?: Record<string, unknown>[];
  updated_at?: string;
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /** Загрузить видео на сервер (без анализа). Таймаут 2 часа для больших файлов. */
  async uploadVideo(
    file: File,
    onUploadProgress?: (progress: number) => void,
    employeeName?: string,
    batchId?: string,
    batchSize?: number
  ): Promise<{ success: boolean; video_id: string; filename: string; original_filename: string; file_size: number }> {
    // Шаг 1: инициализация — получаем video_id (без загрузки файла)
    const initResp = await fetch(`${this.baseUrl}/api/init-upload`, {
      method: 'POST',
      headers: clientHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        filename: file.name,
        employee_name: employeeName || null,
        client_source: isDesktopClient() ? 'desktop' : 'web',
      }),
    });
    if (!initResp.ok) {
      const err = await initResp.text().catch(() => 'init failed');
      throw new Error(`Init upload failed: ${err.slice(0, 200)}`);
    }
    const initData = await initResp.json();
    const videoId: string = initData.video_id;

    // Шаг 2: загрузка raw-байтов файла на proxy (минуя multipart)
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.baseUrl}/api/upload-video/${videoId}`);
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');
      if (isDesktopClient()) {
        xhr.setRequestHeader('X-TrackAI-Client', 'desktop');
      }

      if (xhr.upload && onUploadProgress) {
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            onUploadProgress((event.loaded / event.total) * 100);
          }
        });
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            resolve({ ...result, video_id: videoId });
          } catch {
            reject(new Error('Ошибка ответа сервера'));
          }
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.detail || `Ошибка ${xhr.status}`));
          } catch {
            reject(new Error(`Ошибка ${xhr.status}`));
          }
        }
      };
      xhr.onerror = () => reject(new Error('Сетевая ошибка при загрузке'));
      xhr.ontimeout = () => reject(new Error('Таймаут загрузки (2 часа). Проверьте соединение.'));
      xhr.timeout = 2 * 60 * 60 * 1000;
      xhr.send(file);
    });
  }

  /** Запустить анализ уже загруженного видео по video_id */
  async analyzeVideoById(
    videoId: string,
    scaleFactor: number = 12.306,
    stabilize: boolean = true,
    originalFilename?: string,
    trackingOptions?: TrackingOptions,
    mapContext?: MapContext,
    employeeName?: string,
    analysisMethod?: 'slam' | 'r3' | 'lingbot',
    r3Options?: { frame_stride?: number; max_frames?: number; ckpt?: string; size?: number; mode?: string },
    forceReprocess: boolean = false
  ): Promise<VideoAnalysisResult> {
    const body: Record<string, unknown> = {
      video_id: videoId,
      scale_factor: scaleFactor,
      stabilize,
      original_filename: originalFilename || 'video',
      detect_interval: trackingOptions?.detect_interval ?? 5,
      turn_vote_threshold: trackingOptions?.turn_vote_threshold ?? 3,
      use_ml_roi: trackingOptions?.use_ml_roi ?? true,
      floor_plan_data: mapContext?.floor_plan_data ?? null,
      drawn_plan: mapContext?.drawn_plan ?? null,
      reference_point: mapContext?.reference_point ?? null,
      direction_point: mapContext?.direction_point ?? null,
      employee_name: employeeName,
      analysis_method: analysisMethod || 'slam',
      force_reprocess: forceReprocess,
    };
    if (analysisMethod === 'r3') {
      body.frame_stride = r3Options?.frame_stride ?? 5;
      body.max_frames = r3Options?.max_frames ?? 1500;
      body.ckpt = r3Options?.ckpt ?? 'r3_long.safetensors';
      body.size = r3Options?.size ?? 392;
      body.mode = r3Options?.mode ?? 'strided';
    } else if (analysisMethod === 'lingbot') {
      body.lingbot_fps = 10;
      body.lingbot_keyframe_interval = 6;
      body.lingbot_use_sdpa = true;
      body.lingbot_mask_sky = false;
    }

    const response = await agentFetch(`${this.baseUrl}/api/analyze-video-by-id`, {
      method: 'POST',
      headers: clientHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Ошибка ${response.status}`);
    }
    return response.json();
  }

  async analyzeVideo(
    file: File,
    scaleFactor: number = 12.306,
    stabilize: boolean = true,
    clientId?: string,
    onUploadProgress?: (progress: number) => void,
    trackingOptions?: TrackingOptions,
    mapContext?: MapContext,
    employeeName?: string
  ): Promise<VideoAnalysisResult> {
    console.log(`🔗 API: Отправка запроса на анализ видео`);
    console.log(`   📄 Файл: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`   📏 Масштаб: ${scaleFactor}`);
    console.log(`   🎥 Стабилизация: ${stabilize}`);
    if (clientId) console.log(`   🆔 Client ID: ${clientId}`);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('scale_factor', scaleFactor.toString());
    formData.append('stabilize', stabilize.toString());
    formData.append('detect_interval', String(trackingOptions?.detect_interval ?? 5));
    formData.append('turn_vote_threshold', String(trackingOptions?.turn_vote_threshold ?? 3));
    formData.append('use_ml_roi', String(trackingOptions?.use_ml_roi ?? true));
    if (employeeName) {
      formData.append('employee_name', employeeName);
    }

    if (mapContext?.floor_plan_data) {
      formData.append('floor_plan_data', mapContext.floor_plan_data);
    }
    if (mapContext?.drawn_plan) {
      formData.append('drawn_plan', JSON.stringify(mapContext.drawn_plan));
    }
    if (mapContext?.batch_id) {
      formData.append('batch_id', String(mapContext.batch_id));
    }
    if (mapContext?.batch_size !== undefined) {
      formData.append('batch_size', String(mapContext.batch_size));
    }
    if (mapContext?.reference_point) {
      formData.append('reference_point', JSON.stringify(mapContext.reference_point));
    }
    if (mapContext?.direction_point) {
      formData.append('direction_point', JSON.stringify(mapContext.direction_point));
    }
    if (clientId) {
      formData.append('client_id', clientId);
    }

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const startTime = Date.now();

      xhr.open('POST', `${this.baseUrl}/api/analyze-video`);
      if (isDesktopClient()) {
        xhr.setRequestHeader('X-TrackAI-Client', 'desktop');
      }

      if (xhr.upload && onUploadProgress) {
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100;
            onUploadProgress(percentComplete);
          }
        });
      }

      xhr.onload = () => {
        const endTime = Date.now();
        const responseTime = (endTime - startTime) / 1000;
        console.log(`📡 API: Получен ответ от сервера (${xhr.status}) за ${responseTime.toFixed(1)} сек`);

        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            if (result.status === 'queued') {
              console.log(`📡 API: Видео поставлено в очередь для обработки (ID: ${result.video_id})`);
            }
            resolve(result);
          } catch (e) {
            reject(new Error("Не удалось распарсить ответ сервера"));
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.detail || `Ошибка сервера (${xhr.status})`));
          } catch (e) {
            reject(new Error(`Ошибка сервера (${xhr.status})`));
          }
        }
      };

      xhr.onerror = () => {
        reject(new Error("Сетевая ошибка при загрузке видео"));
      };

      xhr.onabort = () => {
        reject(new Error("Загрузка видео отменена"));
      };

      // Таймаут 30 минут
      xhr.timeout = 1800000;
      xhr.ontimeout = () => {
        reject(new Error("Превышено время ожидания загрузки видео (30 мин)"));
      };

      console.log(`🌐 Отправка POST запроса на ${this.baseUrl}/api/analyze-video...`);
      xhr.send(formData);
    });
  }

  async healthCheck(): Promise<{ status: string; service: string }> {
    const response = await agentFetch(`${this.baseUrl}/api/health`);
    if (!response.ok) {
      throw new Error('Backend is not available');
    }
    return response.json();
  }

  /** Список загруженных на сервер видео (для выбора перед анализом) */
  async getUploadedVideosList(): Promise<VideoListResponse> {
    const response = await agentFetch(`${this.baseUrl}/api/uploaded-videos`);
    if (!response.ok) {
      throw new Error('Failed to fetch uploaded videos list');
    }
    return response.json();
  }

  async getAdminTasks(): Promise<TrackingTask[]> {
    const response = await agentFetch(`${this.baseUrl}/api/admin/tasks`);
    if (!response.ok) {
      throw new Error('Failed to fetch admin tasks');
    }
    return response.json();
  }

  async getAdminTask(id: string): Promise<TrackingTask> {
    const response = await agentFetch(`${this.baseUrl}/api/admin/tasks/${id}`);
    if (!response.ok) {
      throw new Error('Failed to fetch admin task');
    }
    return response.json();
  }

  async registerExistingVideoTask(
    videoId: string,
    employeeName?: string
  ): Promise<{ success: boolean; video_id: string; status: string }> {
    const response = await agentFetch(`${this.baseUrl}/api/admin/tasks/${videoId}/register-existing`, {
      method: 'POST',
      headers: clientHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        employee_name: employeeName || null,
        client_source: isDesktopClient() ? 'desktop' : 'web',
      }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Не удалось зарегистрировать видео в админке');
    }
    return response.json();
  }

  async deleteAdminTask(id: string): Promise<{ success: boolean; id: string }> {
    const response = await agentFetch(`${this.baseUrl}/api/admin/tasks/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      let detail = 'Не удалось удалить задачу';
      try {
        const err = await response.json();
        if (err?.detail) detail = typeof err.detail === 'string' ? err.detail : detail;
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    return response.json();
  }

  async clearAdminDatabase(): Promise<{ success: boolean; message?: string }> {
    const response = await agentFetch(`${this.baseUrl}/api/admin/clear-database`, {
      method: 'POST',
    });
    if (!response.ok) {
      let detail = 'Не удалось очистить базу';
      try {
        const err = await response.json();
        if (err?.detail) detail = typeof err.detail === 'string' ? err.detail : detail;
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    return response.json();
  }

  async updateTaskContext(taskId: string, context: {
    floor_plan_data?: string | null;
    drawn_plan?: unknown[] | null;
    reference_point?: { x: number; y: number } | null;
    direction_point?: { x: number; y: number } | null;
    employee_name?: string;
  }): Promise<{ success: boolean }> {
    const response = await agentFetch(`${this.baseUrl}/api/admin/tasks/${taskId}/context`, {
      method: 'POST',
      headers: clientHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(context),
    });
    if (!response.ok) {
      // Silently fail — context sync is best-effort
      console.warn(`Failed to update task context for ${taskId}`);
      return { success: false };
    }
    return response.json();
  }

  async getVideosList(): Promise<VideoListResponse> {
    const response = await agentFetch(`${this.baseUrl}/api/videos`);
    if (!response.ok) {
      throw new Error('Failed to fetch videos list');
    }
    return response.json();
  }

  getUploadedVideoUrl(videoId: string): string {
    return `${this.baseUrl}/api/uploaded-video/${videoId}/stream`;
  }

  getUploadedVideoPreviewUrl(videoId: string): string {
    return `${this.baseUrl}/api/uploaded-video/${videoId}/preview.mp4`;
  }

  async getManualTrajectory(videoId: string): Promise<ManualTrajectoryResponse> {
    const response = await agentFetch(`${this.baseUrl}/api/manual-trajectory/${videoId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch manual trajectory');
    }
    return response.json();
  }

  async saveManualTrajectory(
    videoId: string,
    trajectory: number[][],
    turnPoints: Record<string, unknown>[] = []
  ): Promise<{ success: boolean; video_id: string; updated_at: string; trajectory_points: number }> {
    const response = await agentFetch(`${this.baseUrl}/api/manual-trajectory/${videoId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        trajectory,
        turn_points: turnPoints,
      }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to save manual trajectory");
    }
    return response.json();
  }

  async getVideoAnalysis(videoId: string): Promise<VideoAnalysisResult> {
    const response = await agentFetch(`${this.baseUrl}/api/video/${videoId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch video analysis');
    }
    return response.json();
  }

  async getProcessingStatus(videoId: string): Promise<{
    status: string;
    progress: number;
    message: string;
    result?: VideoAnalysisResult["data"];
  }> {
    const response = await agentFetch(`${this.baseUrl}/api/status/${videoId}`);
    if (!response.ok) {
      // If status endpoint returns 404 or other error, return default unknown status
      return { status: "unknown", progress: 0, message: "Status not available" };
    }
    return response.json();
  }

  getVideoDownloadUrl(videoId: string): string {
    return `${this.baseUrl}/api/video/${videoId}/download`;
  }

  async getPlans(): Promise<Plan[]> {
    const response = await agentFetch(`${this.baseUrl}/api/plans`);
    if (!response.ok) {
      throw new Error('Failed to fetch plans');
    }
    return response.json();
  }

  async savePlan(plan: Plan): Promise<{ id: number; name: string; status: string }> {
    const response = await agentFetch(`${this.baseUrl}/api/plans`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(plan),
    });
    if (!response.ok) {
      throw new Error('Failed to save plan');
    }
    return response.json();
  }

  async deletePlan(id: number): Promise<{ status: string; id: number }> {
    const response = await agentFetch(`${this.baseUrl}/api/plans/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to delete plan');
    }
    return response.json();
  }

  async convertPdfToImage(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ success: boolean; png: string; filename: string }> {
    const formData = new FormData();
    formData.append('file', file);
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.baseUrl}/api/convert-pdf`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          onProgress?.(Math.round((e.loaded / e.total) * 50));
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new Error('Ошибка ответа сервера'));
          }
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.detail || `Ошибка ${xhr.status}`));
          } catch {
            reject(new Error(`Ошибка ${xhr.status}`));
          }
        }
      };
      xhr.onerror = () => reject(new Error('Сетевая ошибка'));
      xhr.ontimeout = () => reject(new Error('Таймаут загрузки PDF'));
      xhr.timeout = 5 * 60 * 1000; // 5 мин
      xhr.send(formData);
    });
  }

  async convertDwgToImage(
    file: File,
    onProgress?: (progress: number, message: string) => void
  ): Promise<{ success: boolean; png: string; filename: string }> {
    const formData = new FormData();
    formData.append('file', file);
    const job_id = await new Promise<string>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.baseUrl}/api/convert-dwg`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 20);
          onProgress?.(pct, 'загрузка');
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            if (data.job_id) resolve(data.job_id);
            else reject(new Error('Сервер не вернул job_id'));
          } catch {
            reject(new Error('Ошибка ответа сервера'));
          }
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.detail || `Ошибка ${xhr.status}`));
          } catch {
            reject(new Error(`Ошибка ${xhr.status}`));
          }
        }
      };
      xhr.onerror = () => reject(new Error('Сетевая ошибка'));
      xhr.ontimeout = () => reject(new Error('Таймаут загрузки. Файл 170 MB — экспортируйте план в PNG в AutoCAD (File → Export → PNG).'));
      xhr.timeout = 30 * 60 * 1000;
      xhr.send(formData);
    });
    const start = Date.now();
    const timeout = 30 * 60 * 1000;
    while (true) {
      if (Date.now() - start > timeout) {
        throw new Error('Превышено время ожидания. Файл 170 MB — экспортируйте план в PNG в AutoCAD (File → Export → PNG).');
      }
      const statusRes = await agentFetch(`${this.baseUrl}/api/convert-dwg-status/${job_id}`);
      const status = await statusRes.json();
      const serverProgress = status.progress ?? 0;
      onProgress?.(20 + Math.round(serverProgress * 0.8), status.message ?? '');
      if (status.status === 'done') {
        return { success: true, png: status.png, filename: status.filename };
      }
      if (status.status === 'error') {
        throw new Error(status.error || status.message || 'Ошибка конвертации DWG');
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  /** Subscribe to real-time R³ streaming events via SSE */
  subscribeR3Stream(
    videoId: string,
    callbacks: {
      onFrameProcessed?: (data: {
        num_processed: number;
        new_trajectory_points?: number[][];
        new_poses?: unknown[];
      }) => void;
      onVideoInfo?: (data: { frames: number; fps: number; width: number; height: number }) => void;
      onComplete?: (data: Record<string, unknown>) => void;
      onError?: (error: string) => void;
      onStatus?: (data: Record<string, unknown>) => void;
    }
  ): () => void {
    const url = `${this.baseUrl}/api/r3-stream/${videoId}`;
    const eventSource = new EventSource(url);

    eventSource.addEventListener('frame_processed', (event) => {
      try {
        const data = JSON.parse(event.data);
        callbacks.onFrameProcessed?.(data);
      } catch { /* ignore parse errors */ }
    });

    eventSource.addEventListener('video_info', (event) => {
      try {
        const data = JSON.parse(event.data);
        callbacks.onVideoInfo?.(data);
      } catch { /* ignore */ }
    });

    eventSource.addEventListener('complete', (event) => {
      try {
        const data = JSON.parse(event.data);
        callbacks.onComplete?.(data);
      } catch { /* ignore */ }
    });

    eventSource.addEventListener('error', (event) => {
      try {
        const data = event.data ? JSON.parse(event.data) : {};
        callbacks.onError?.(data.message || 'SSE connection error');
      } catch {
        callbacks.onError?.('SSE connection error');
      }
    });

    // Generic handler for any other events
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        callbacks.onStatus?.(data);
      } catch { /* ignore */ }
    };

    // Support extra events
    eventSource.addEventListener('connected', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'connected' }); } catch {}
    });
    eventSource.addEventListener('receiving', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'receiving' }); } catch {}
    });
    eventSource.addEventListener('video_received', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'video_received' }); } catch {}
    });
    eventSource.addEventListener('processing_started', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'r3_start' }); } catch {}
    });
    eventSource.addEventListener('r3_start', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'r3_start' }); } catch {}
    });
    eventSource.addEventListener('replay', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'replay' }); } catch {}
    });
    eventSource.addEventListener('pointcloud_status', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'pointcloud_status' }); } catch {}
    });
    eventSource.addEventListener('r3_segment_start', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'r3_segment_start' }); } catch {}
    });
    eventSource.addEventListener('r3_segment_complete', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'r3_segment_complete' }); } catch {}
    });
    eventSource.addEventListener('r3_segmented_complete', (event) => {
      try { callbacks.onStatus?.({ ...JSON.parse(event.data), event_type: 'r3_segmented_complete' }); } catch {}
    });

    // Return unsubscribe function
    return () => {
      eventSource.close();
    };
  }

  /** Статус фонового построения production point cloud. */
  async getR3PointCloudStatus(videoId: string): Promise<{
    video_id: string;
    status: "not_started" | "queued" | "processing" | "completed" | "error" | "cancelled";
    stage: string;
    progress: number;
    message: string;
    points?: number;
    source_points?: number;
    frames_used?: number;
    elapsed_seconds?: number;
    error?: string;
  }> {
    const resp = await agentFetch(`${this.baseUrl}/api/r3-pointcloud-status/${videoId}`);
    if (!resp.ok) {
      throw new Error(`Point cloud status fetch failed (HTTP ${resp.status})`);
    }
    return resp.json();
  }

  /** Получить полное облако точек R³ через отдельный API (не через SSE). */
  async getR3PointCloud(videoId: string, maxPoints: number = 100000, minConf: number = 1.0): Promise<{
    success: boolean;
    video_id: string;
    num_points: number;
    num_points_total: number;
    points: number[][];
  }> {
    const resp = await agentFetch(
      `${this.baseUrl}/api/r3-pointcloud/${videoId}?max_points=${maxPoints}&min_conf=${minConf}`,
    );
    if (!resp.ok) {
      throw new Error(`Point cloud fetch failed (HTTP ${resp.status})`);
    }
    return resp.json();
  }

  /** Получить серверно отфильтрованное облако R³. */
  async getR3PointCloudFiltered(videoId: string, params: {
    maxPoints?: number;
    minConf?: number;
    frameStart?: number;
    frameEnd?: number;
    samplingStrategy?: "confidence_top" | "random" | "voxel" | "per_frame_uniform";
    includeTrajectory?: boolean;
    includeCameras?: boolean;
  } = {}): Promise<{
    success: boolean;
    video_id: string;
    points: number[][];
    /** Plan-space path for the floor map. This remains the compatibility trajectory. */
    trajectory?: number[][];
    plan_trajectory?: number[][];
    /** Cleaned c2w translations for Three.js only; never use these as map X/Y. */
    raw_trajectory_3d?: number[][];
    turn_points?: Array<{
      frame_index: number;
      r3_frame_index?: number;
      source_frame_index?: number | null;
      timestamp_seconds?: number | null;
      trajectory_index: number;
      angle_degrees: number;
      position: number[];
      turn_type: string;
      confidence?: number | null;
    }>;
    source_frame_indices?: Array<number | null>;
    source_timestamps_seconds?: Array<number | null>;
    cameras?: unknown[];
    stats?: {
      source_points: number;
      filtered_points: number;
      returned_points: number;
      min_conf: number;
      frame_start: number | null;
      frame_end: number | null;
      sampling_strategy: string;
      trajectory_quality?: {
        quality?: string;
        raw_points?: number;
        cleaned_points?: number;
        raw_step_median?: number;
        raw_step_p90?: number;
        raw_step_p99?: number;
        step_limit?: number;
        clipped_steps?: number;
        outlier_points?: number;
        smoothed_points?: number;
        cleaned_distance?: number;
        projection?: Record<string, unknown>;
      } | null;
    };
    diagnostics?: {
      pointcloud_file: string;
      pointcloud_shape: number[];
      has_conf: boolean;
      has_frame_idx: boolean;
      run_params?: Record<string, unknown>;
      stale_run?: boolean;
    };
  }> {
    const query = new URLSearchParams();
    query.set("max_points", String(params.maxPoints ?? 100000));
    query.set("min_conf", String(params.minConf ?? 1.4));
    query.set("sampling_strategy", params.samplingStrategy ?? "random");
    query.set("include_trajectory", String(params.includeTrajectory ?? false));
    query.set("include_cameras", String(params.includeCameras ?? false));
    if (typeof params.frameStart === "number") query.set("frame_start", String(params.frameStart));
    if (typeof params.frameEnd === "number") query.set("frame_end", String(params.frameEnd));

    const resp = await agentFetch(
      `${this.baseUrl}/api/r3-pointcloud-filtered/${videoId}?${query.toString()}`,
    );
    if (!resp.ok) {
      throw new Error(`Filtered point cloud fetch failed (HTTP ${resp.status})`);
    }
    return resp.json();
  }

  /** Rebuild only the current R3 trajectory; avoids loading the point cloud. */
  async getR3Trajectory(
    videoId: string,
    trajectorySource: R3TrajectorySource = "raw",
  ): Promise<{
    success: boolean;
    video_id: string;
    method: string;
    trajectory: number[][];
    plan_trajectory: number[][];
    raw_plan_trajectory?: number[][];
    raw_trajectory_3d?: number[][];
    turn_points?: Array<{
      frame_index: number;
      r3_frame_index?: number;
      source_frame_index?: number | null;
      timestamp_seconds?: number | null;
      trajectory_index: number;
      angle_degrees: number;
      position: number[];
      turn_type: string;
      confidence?: number | null;
    }>;
    source_frame_indices?: Array<number | null>;
    source_timestamps_seconds?: Array<number | null>;
    trajectory_quality?: Record<string, unknown>;
    trajectory_source_requested?: R3TrajectorySource;
    trajectory_source?: R3TrajectorySource;
    trajectory_source_fallback_reason?: string | null;
    trajectory_source_selection?: Record<string, unknown>;
    run_params?: Record<string, unknown>;
    fallback_summary?: Record<string, unknown>;
    pose_graph?: Record<string, unknown>;
    pose_graph_candidate?: Record<string, unknown>;
  }> {
    const query = new URLSearchParams({ trajectory_source: trajectorySource });
    const resp = await agentFetch(
      `${this.baseUrl}/api/r3-trajectory/${videoId}?${query.toString()}`,
    );
    if (!resp.ok) {
      throw new Error(`R3 trajectory fetch failed (HTTP ${resp.status})`);
    }
    return resp.json();
  }

  /** Диагностика R³ output: файлы, pointcloud shape, confidence percentiles. */
  async getR3Diagnostics(videoId: string): Promise<{
    success: boolean;
    video_id: string;
    output_exists: boolean;
    files?: Record<string, number>;
    pointcloud?: {
      exists: boolean;
      file?: string;
      shape?: number[];
      has_conf?: boolean;
      has_frame_idx?: boolean;
      rgb_min?: number[];
      rgb_max?: number[];
    };
    conf_stats?: {
      percentiles?: Record<string, number>;
      counts_by_threshold?: Record<string, number>;
    };
    run_params?: Record<string, unknown>;
  }> {
    const resp = await agentFetch(`${this.baseUrl}/api/r3-diagnostics/${videoId}`);
    if (!resp.ok) {
      throw new Error(`R3 diagnostics fetch failed (HTTP ${resp.status})`);
    }
    return resp.json();
  }
}

export const apiClient = new ApiClient();
