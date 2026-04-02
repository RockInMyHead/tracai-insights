// API client for TrackAI backend
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// #region agent log
async function agentFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const urlStr =
    typeof input === 'string'
      ? input
      : input instanceof URL
        ? input.href
        : (input as Request).url;
  const res = await globalThis.fetch(input, init);
  fetch('http://127.0.0.1:7343/ingest/767aed2a-4a75-4bf7-922d-0437d34eb3ef', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Debug-Session-Id': '64890b' },
    body: JSON.stringify({
      sessionId: '64890b',
      runId: 'run1',
      hypothesisId: 'H3',
      location: 'api.ts:agentFetch',
      message: 'fetch_response',
      data: { url: urlStr.slice(0, 400), status: res.status, ok: res.ok },
      timestamp: Date.now(),
    }),
  }).catch(() => {});
  return res;
}
function agentLogXhr(hypothesisId: string, path: string, status: number, ok: boolean): void {
  fetch('http://127.0.0.1:7343/ingest/767aed2a-4a75-4bf7-922d-0437d34eb3ef', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Debug-Session-Id': '64890b' },
    body: JSON.stringify({
      sessionId: '64890b',
      runId: 'run1',
      hypothesisId,
      location: 'api.ts:xhr',
      message: 'xhr_response',
      data: { path: path.slice(0, 400), status, ok },
      timestamp: Date.now(),
    }),
  }).catch(() => {});
}
// #endregion

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
    employeeName?: string
  ): Promise<{ success: boolean; video_id: string; filename: string; original_filename: string; file_size: number }> {
    const formData = new FormData();
    formData.append('file', file);
    if (employeeName) {
      formData.append('employee_name', employeeName);
    }

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.baseUrl}/api/upload-video`);

      if (xhr.upload && onUploadProgress) {
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            onUploadProgress((event.loaded / event.total) * 100);
          }
        });
      }

      xhr.onload = () => {
        // #region agent log
        agentLogXhr('H4', `${this.baseUrl}/api/upload-video`, xhr.status, xhr.status >= 200 && xhr.status < 300);
        // #endregion
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
      xhr.onerror = () => reject(new Error('Сетевая ошибка при загрузке'));
      xhr.ontimeout = () => reject(new Error('Таймаут загрузки (2 часа). Проверьте соединение.'));
      xhr.timeout = 2 * 60 * 60 * 1000; // 2 часа
      xhr.send(formData);
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
    employeeName?: string
  ): Promise<VideoAnalysisResult> {
    const response = await agentFetch(`${this.baseUrl}/api/analyze-video-by-id`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
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
        employee_name: employeeName
      }),
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
        // #region agent log
        agentLogXhr('H4', `${this.baseUrl}/api/analyze-video`, xhr.status, xhr.status >= 200 && xhr.status < 300);
        // #endregion

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

  async updateTaskContext(taskId: string, context: {
    floor_plan_data?: string | null;
    drawn_plan?: unknown[] | null;
    reference_point?: { x: number; y: number } | null;
    direction_point?: { x: number; y: number } | null;
    employee_name?: string;
  }): Promise<{ success: boolean }> {
    const response = await agentFetch(`${this.baseUrl}/api/admin/tasks/${taskId}/context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
}

export const apiClient = new ApiClient();
