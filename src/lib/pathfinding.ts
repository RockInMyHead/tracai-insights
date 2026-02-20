/**
 * Pathfinding по плану помещения — траектория не проходит сквозь стены.
 * Использует A* на occupancy grid, построенном из изображения плана.
 */

export interface Point {
  x: number;
  y: number;
}

const GRID_CELL = 8; // пикселей на ячейку (меньше = точнее, но медленнее)
const LUMINANCE_THRESHOLD = 0.6; // выше = проходимо (белый/светлый), ниже = стена
const SUBSAMPLE = 5; // брать каждую N-ю точку как waypoint для pathfinding

function luminance(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

/** Строит occupancy grid из изображения плана. 0 = проходимо, 1 = стена */
export async function buildOccupancyGrid(
  imageUrl: string,
  width: number,
  height: number
): Promise<{ grid: Uint8Array; cols: number; rows: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    if (!imageUrl.startsWith("data:")) img.crossOrigin = "anonymous";
    img.onload = () => {
      const cols = Math.ceil(width / GRID_CELL);
      const rows = Math.ceil(height / GRID_CELL);
      const grid = new Uint8Array(cols * rows);

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas context"));
        return;
      }
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);
      const imageData = ctx.getImageData(0, 0, width, height);
      const data = imageData.data;

      for (let gy = 0; gy < rows; gy++) {
        for (let gx = 0; gx < cols; gx++) {
          let sum = 0;
          let count = 0;
          for (let py = gy * GRID_CELL; py < Math.min((gy + 1) * GRID_CELL, height); py++) {
            for (let px = gx * GRID_CELL; px < Math.min((gx + 1) * GRID_CELL, width); px++) {
              const i = (py * width + px) * 4;
              sum += luminance(data[i], data[i + 1], data[i + 2]);
              count++;
            }
          }
          const avg = count > 0 ? sum / count : 1;
          grid[gy * cols + gx] = avg >= LUMINANCE_THRESHOLD ? 0 : 1;
        }
      }
      resolve({ grid, cols, rows });
    };
    img.onerror = () => reject(new Error("Failed to load floor plan image"));
    img.src = imageUrl;
  });
}

function toGrid(x: number, y: number, cols: number, rows: number): { gx: number; gy: number } {
  return {
    gx: Math.max(0, Math.min(cols - 1, Math.floor(x / GRID_CELL))),
    gy: Math.max(0, Math.min(rows - 1, Math.floor(y / GRID_CELL))),
  };
}

function toWorld(gx: number, gy: number): Point {
  return {
    x: gx * GRID_CELL + GRID_CELL / 2,
    y: gy * GRID_CELL + GRID_CELL / 2,
  };
}

/** A* pathfinding */
function astar(
  grid: Uint8Array,
  cols: number,
  rows: number,
  start: { gx: number; gy: number },
  end: { gx: number; gy: number }
): Point[] {
  const key = (gx: number, gy: number) => gy * cols + gx;
  const isWall = (gx: number, gy: number) => {
    if (gx < 0 || gx >= cols || gy < 0 || gy >= rows) return true;
    return grid[key(gx, gy)] === 1;
  };

  if (isWall(start.gx, start.gy)) {
    // Старт в стене — ищем ближайшую свободную
    for (let r = 1; r <= 3; r++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const nx = start.gx + dx;
          const ny = start.gy + dy;
          if (!isWall(nx, ny)) {
            start = { gx: nx, gy: ny };
            break;
          }
        }
      }
    }
  }
  if (isWall(end.gx, end.gy)) {
    for (let r = 1; r <= 3; r++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const nx = end.gx + dx;
          const ny = end.gy + dy;
          if (!isWall(nx, ny)) {
            end = { gx: nx, gy: ny };
            break;
          }
        }
      }
    }
  }

  const open = new Map<string, { gx: number; gy: number; f: number }>();
  const cameFrom = new Map<string, { gx: number; gy: number }>();
  const gScore = new Map<string, number>();
  const k = (gx: number, gy: number) => `${gx},${gy}`;

  open.set(k(start.gx, start.gy), { ...start, f: 0 });
  gScore.set(k(start.gx, start.gy), 0);

  const neighbors = [
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-1, -1], [-1, 1], [1, -1], [1, 1],
  ];

  let iterations = 0;
  const maxIter = cols * rows * 2;

  while (open.size > 0 && iterations++ < maxIter) {
    let current = { gx: -1, gy: -1, f: Infinity };
    for (const [, v] of open) {
      if (v.f < current.f) current = v;
    }
    const ck = k(current.gx, current.gy);
    open.delete(ck);

    if (current.gx === end.gx && current.gy === end.gy) {
      const path: Point[] = [];
      let cur: { gx: number; gy: number } | undefined = end;
      while (cur) {
        path.unshift(toWorld(cur.gx, cur.gy));
        cur = cameFrom.get(k(cur.gx, cur.gy));
      }
      return path;
    }

    for (const [dx, dy] of neighbors) {
      const nx = current.gx + dx;
      const ny = current.gy + dy;
      if (isWall(nx, ny)) continue;

      const nk = k(nx, ny);
      const dist = Math.sqrt(dx * dx + dy * dy);
      const tentative = (gScore.get(ck) ?? Infinity) + dist;

      if (tentative < (gScore.get(nk) ?? Infinity)) {
        cameFrom.set(nk, { gx: current.gx, gy: current.gy });
        gScore.set(nk, tentative);
        const h = Math.sqrt((end.gx - nx) ** 2 + (end.gy - ny) ** 2);
        open.set(nk, { gx: nx, gy: ny, f: tentative + h });
      }
    }
  }

  return [toWorld(start.gx, start.gy), toWorld(end.gx, end.gy)];
}

/** Корректирует траекторию с учётом стен на плане */
export async function correctPathWithFloorPlan(
  floorPlanUrl: string,
  points: Point[],
  viewBoxWidth: number,
  viewBoxHeight: number
): Promise<Point[]> {
  if (points.length < 2) return points;

  const { grid, cols, rows } = await buildOccupancyGrid(floorPlanUrl, viewBoxWidth, viewBoxHeight);

  const waypoints: Point[] = [];
  for (let i = 0; i < points.length; i += SUBSAMPLE) {
    waypoints.push(points[i]);
  }
  if (waypoints[waypoints.length - 1]?.x !== points[points.length - 1]?.x ||
      waypoints[waypoints.length - 1]?.y !== points[points.length - 1]?.y) {
    waypoints.push(points[points.length - 1]);
  }

  const result: Point[] = [];
  for (let i = 0; i < waypoints.length - 1; i++) {
    const a = toGrid(waypoints[i].x, waypoints[i].y, cols, rows);
    const b = toGrid(waypoints[i + 1].x, waypoints[i + 1].y, cols, rows);

    const seg = astar(grid, cols, rows, a, b);
    const skipLast = i < waypoints.length - 2;
    for (let j = 0; j < seg.length - (skipLast ? 1 : 0); j++) {
      result.push(seg[j]);
    }
  }
  return result;
}
