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
  processing?: { resolveMode: () => Promise<{ mode: 'online' | 'local'; label: string }> };
  localCpu?: {
    process: (video: unknown) => Promise<unknown>;
    history: () => Promise<unknown[]>;
    analysis: (videoId: string) => Promise<unknown>;
  };
}

interface Window {
  trackai?: TrackAIWindowBridge;
}
