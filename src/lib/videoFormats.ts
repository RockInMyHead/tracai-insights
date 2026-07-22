export const VIDEO_EXTENSIONS = [
  ".mp4",
  ".avi",
  ".mov",
  ".mkv",
  ".webm",
  ".flv",
  ".wmv",
  ".3gp",
  ".mts",
  ".m2ts",
] as const;

export function isVideoFileName(name: string): boolean {
  const lower = name.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export function isVideoFile(file: Pick<File, "name" | "type">): boolean {
  if (file.type.startsWith("video/")) return true;
  return isVideoFileName(file.name);
}
