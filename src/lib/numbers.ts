/** Безопасное приведение к числу (для ESLint: без `Number(x) ?? 0`). */
export function finiteNum(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}
