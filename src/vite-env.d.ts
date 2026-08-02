/// <reference types="vite/client" />

interface TrackAICameraImportBridge {
  getSettings: () => Promise<{ enabled: boolean; ownerName: string }>;
  setSettings: (settings: Partial<{ enabled: boolean; ownerName: string }>) => Promise<{ enabled: boolean; ownerName: string }>;
  scanNow: (options?: { forceImport?: boolean }) => Promise<unknown>;
  getStatus: () => Promise<unknown>;
  onStatus: (callback: (status: unknown) => void) => () => void;
  onProgress: (callback: (progress: unknown) => void) => () => void;
  onComplete: (callback: (videos: unknown[]) => void) => () => void;
  onError: (callback: (error: { message: string }) => void) => () => void;
}

interface TrackAIWindowBridge {
  isDesktop?: boolean;
  version?: string;
  serverUrl?: string;
  openExternal?: (url: string) => void;
  copyToClipboard?: (text: string) => void;
  readFromClipboard?: () => string;
  cameraImport?: TrackAICameraImportBridge;
  processing?: {
    resolveMode: () => Promise<{
      mode: 'online' | 'local_gpu' | 'local_cpu';
      label: string;
      reason?: string | null;
      localGpuReason?: string;
      localGpuDetails?: {
        reason?: string;
        minimumDriver?: string;
        driverUrl?: string;
        gpus?: Array<{
          name?: string;
          memoryMb?: number;
          driver?: string;
        }>;
      };
      gpu?: {
        name?: string;
        memoryMb?: number;
        driver?: string;
      };
      minimumDriver?: string;
    }>;
  };
  logs?: {
    download: () => Promise<{
      ok: boolean;
      canceled: boolean;
      filePath?: string;
      fileName?: string;
      fileCount?: number;
      archiveBytes?: number;
    }>;
  };
  localCpu?: {
    process: (video: unknown) => Promise<unknown>;
    history: () => Promise<unknown[]>;
    analysis: (videoId: string) => Promise<unknown>;
    onProgress: (callback: (progress: unknown) => void) => () => void;
  };
  localGpu?: {
    process: (video: unknown) => Promise<unknown>;
    history: () => Promise<unknown[]>;
    analysis: (videoId: string) => Promise<unknown>;
  };
}

interface Window {
  trackai?: TrackAIWindowBridge;
}
