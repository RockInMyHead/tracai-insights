export interface CameraImportVolume {
  name: string;
  rootPath: string;
  fileCount: number;
  totalBytes: number;
  freeBytes: number | null;
  totalVolumeBytes: number | null;
}

export interface CameraImportPendingFile {
  path: string;
  name: string;
  size: number;
  mtimeMs: number;
  fingerprint: string;
  volumeName?: string;
}

export interface CameraImportStatus {
  enabled: boolean;
  scanning: boolean;
  importing: boolean;
  volumes: CameraImportVolume[];
  pendingFiles: CameraImportPendingFile[];
  lastError: string | null;
}

export interface CameraImportProgress {
  index: number;
  total: number;
  fileName: string;
  filePath: string;
  percent: number;
  phase: 'uploading';
}

export interface CameraImportedVideo {
  video_id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  ownerName: string;
  sourcePath: string;
  volumeName: string | null;
  localPath?: string;
}

export interface CameraImportSettings {
  enabled: boolean;
  ownerName: string;
}

export interface TrackAICameraImportAPI {
  getSettings: () => Promise<CameraImportSettings>;
  setSettings: (settings: Partial<CameraImportSettings>) => Promise<CameraImportSettings>;
  scanNow: (options?: { forceImport?: boolean }) => Promise<CameraImportStatus>;
  getStatus: () => Promise<CameraImportStatus>;
  onStatus: (callback: (status: CameraImportStatus) => void) => () => void;
  onProgress: (callback: (progress: CameraImportProgress) => void) => () => void;
  onComplete: (callback: (videos: CameraImportedVideo[]) => void) => () => void;
  onError: (callback: (error: { message: string }) => void) => () => void;
}

export const CAMERA_IMPORT_OWNER_KEY = 'trackai_camera_import_owner';
export const CAMERA_IMPORT_ENABLED_KEY = 'trackai_camera_import_enabled';

export function getCameraImportAPI(): TrackAICameraImportAPI | null {
  if (typeof window === 'undefined') return null;
  const trackai = (window as unknown as { trackai?: { cameraImport?: TrackAICameraImportAPI } }).trackai;
  return trackai?.cameraImport ?? null;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}
