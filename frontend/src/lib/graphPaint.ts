import type { GraphNode } from "@/hooks/useGraphSocket";

export const PLATFORM_COLOR: Record<string, string> = {
  reddit: "#f97316",
  hn: "#eab308",
  github: "#a855f7",
};

const DEFAULT_COLOR = "#6b7280";
const BASE_RADIUS = 4;
const MAX_EXTRA_RADIUS = 12;

export function degreeToRadius(degree: number): number {
  // logarithmic scale so high-degree nodes don't dominate
  return BASE_RADIUS + Math.min(Math.log1p(degree) * 2.5, MAX_EXTRA_RADIUS);
}

export function makeNodeCanvasObject(anomalousIds: Set<string>) {
  return function nodeCanvasObject(
    node: GraphNode & { x?: number; y?: number },
    ctx: CanvasRenderingContext2D,
    _globalScale: number
  ) {
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    const r = degreeToRadius(node.degree);
    const color = PLATFORM_COLOR[node.platform?.toLowerCase()] ?? DEFAULT_COLOR;
    const isAnomaly = anomalousIds.has(node.id);

    // glow
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = isAnomaly ? 18 : 10;

    // fill
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // red anomaly outline
    if (isAnomaly) {
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.restore();
  };
}
