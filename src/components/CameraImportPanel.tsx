import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { HardDriveDownload, Loader2, Usb } from "lucide-react";
import { toast } from "sonner";
import {
  CAMERA_IMPORT_ENABLED_KEY,
  CAMERA_IMPORT_OWNER_KEY,
  CameraImportProgress,
  CameraImportStatus,
  CameraImportedVideo,
  formatBytes,
  getCameraImportAPI,
} from "@/lib/cameraImport";

interface CameraImportPanelProps {
  existingOwners: string[];
  onVideosImported: (videos: CameraImportedVideo[]) => void;
}

const CameraImportPanel = ({ existingOwners, onVideosImported }: CameraImportPanelProps) => {
  const cameraImport = getCameraImportAPI();
  const [enabled, setEnabled] = useState(true);
  const [ownerName, setOwnerName] = useState("");
  const [status, setStatus] = useState<CameraImportStatus | null>(null);
  const [progress, setProgress] = useState<CameraImportProgress | null>(null);

  useEffect(() => {
    if (!cameraImport) return;

    const savedOwner = localStorage.getItem(CAMERA_IMPORT_OWNER_KEY) || "";
    const savedEnabled = localStorage.getItem(CAMERA_IMPORT_ENABLED_KEY);
    const initialEnabled = savedEnabled == null ? true : savedEnabled === "1";
    setOwnerName(savedOwner);
    setEnabled(initialEnabled);

    void cameraImport.setSettings({
      enabled: initialEnabled,
      ownerName: savedOwner,
    });
    void cameraImport.getStatus().then(setStatus).catch(() => {});

    const unsubStatus = cameraImport.onStatus(setStatus);
    const unsubProgress = cameraImport.onProgress(setProgress);
    const unsubComplete = cameraImport.onComplete((videos) => {
      setProgress(null);
      if (videos.length > 0) {
        onVideosImported(videos);
        toast.success(`С камеры загружено ${videos.length} видео в TrackAI`);
      }
    });
    const unsubError = cameraImport.onError((error) => {
      toast.error(error.message || "Ошибка автоимпорта с камеры");
    });

    return () => {
      unsubStatus();
      unsubProgress();
      unsubComplete();
      unsubError();
    };
  }, [cameraImport, onVideosImported]);

  if (!cameraImport) {
    return null;
  }

  const connectedVolume = status?.volumes?.[0] ?? null;
  const pendingCount = status?.pendingFiles?.length ?? 0;
  const isBusy = Boolean(status?.scanning || status?.importing);

  const persistSettings = async (next: { enabled?: boolean; ownerName?: string }) => {
    const nextEnabled = next.enabled ?? enabled;
    const nextOwner = next.ownerName ?? ownerName;
    if (typeof next.enabled === "boolean") {
      setEnabled(next.enabled);
      localStorage.setItem(CAMERA_IMPORT_ENABLED_KEY, next.enabled ? "1" : "0");
    }
    if (typeof next.ownerName === "string") {
      setOwnerName(next.ownerName);
      localStorage.setItem(CAMERA_IMPORT_OWNER_KEY, next.ownerName);
    }
    await cameraImport.setSettings({
      enabled: nextEnabled,
      ownerName: nextOwner,
    });
    if (nextOwner.trim()) {
      await cameraImport.scanNow({ forceImport: true });
    }
  };

  const handleManualImport = async () => {
    if (!ownerName.trim()) {
      toast.error("Укажите сотрудника для импорта с камеры");
      return;
    }
    await persistSettings({ ownerName: ownerName.trim() });
    await cameraImport.scanNow({ forceImport: true });
  };

  return (
    <div className="rounded-lg border border-primary/20 bg-primary/5 p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Usb className="h-4 w-4 text-primary" />
            Автоимпорт с экшен-камеры
          </div>
          <p className="text-xs text-muted-foreground">
            Подключите камеру по USB — TrackAI автоматически загрузит все видео с карты на сервер.
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Label htmlFor="camera-auto-import" className="text-xs text-muted-foreground">
            Авто
          </Label>
          <Switch
            id="camera-auto-import"
            checked={enabled}
            onCheckedChange={(checked) => {
              void persistSettings({ enabled: checked });
            }}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-2">
        <Input
          value={ownerName}
          onChange={(e) => setOwnerName(e.target.value)}
          onBlur={() => void persistSettings({ ownerName: ownerName.trim() })}
          placeholder="Сотрудник для видео с камеры"
        />
        <Button
          type="button"
          variant="secondary"
          className="gap-2"
          disabled={isBusy || pendingCount === 0}
          onClick={() => void handleManualImport()}
        >
          {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <HardDriveDownload className="h-4 w-4" />}
          Загрузить {pendingCount > 0 ? `(${pendingCount})` : ""}
        </Button>
      </div>

      {existingOwners.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {existingOwners.slice(0, 5).map((name) => (
            <Badge
              key={name}
              variant="outline"
              className="cursor-pointer hover:bg-primary/20"
              onClick={() => void persistSettings({ ownerName: name })}
            >
              {name}
            </Badge>
          ))}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 text-xs">
        {connectedVolume ? (
          <>
            <Badge variant="secondary">Камера: {connectedVolume.name}</Badge>
            <Badge variant="outline">{connectedVolume.fileCount} видео</Badge>
            <Badge variant="outline">{formatBytes(connectedVolume.totalBytes)}</Badge>
          </>
        ) : (
          <Badge variant="outline">Камера не подключена</Badge>
        )}
        {pendingCount > 0 && enabled && ownerName.trim() && (
          <Badge className="bg-primary/90">Ожидает загрузки: {pendingCount}</Badge>
        )}
        {!ownerName.trim() && connectedVolume && (
          <span className="text-amber-600">Укажите сотрудника для автозагрузки</span>
        )}
      </div>

      {progress && (
        <div className="text-xs text-muted-foreground">
          Загрузка {progress.index + 1}/{progress.total}: {progress.fileName} — {progress.percent.toFixed(0)}%
        </div>
      )}

      {status?.lastError && (
        <div className="text-xs text-destructive">{status.lastError}</div>
      )}
    </div>
  );
};

export default CameraImportPanel;
