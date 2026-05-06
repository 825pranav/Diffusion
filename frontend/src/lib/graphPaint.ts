import type { GraphNode } from "@/hooks/useGraphSocket";
import { PLATFORM_COLOR } from "@/lib/colors";
export { PLATFORM_COLOR };

const DEFAULT_COLOR = "#3a5045"; // outline / muted
const BASE_RADIUS = 4;
const MAX_EXTRA_RADIUS = 12;

export function degreeToRadius(degree: number): number {
  return BASE_RADIUS + Math.min(Math.log1p(degree) * 2.5, MAX_EXTRA_RADIUS);
}

export function makeNodeCanvasObject(anomalousIds: Set<string>) {
  return function nodeCanvasObject(
    node: GraphNode & { x?: number; y?: number },
    ctx: CanvasRenderingContext2D,
    globalScale: number
  ) {
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    const r = degreeToRadius(node.degree);
    const color = PLATFORM_COLOR[node.platform?.toLowerCase()] ?? DEFAULT_COLOR;
    const isAnomaly = anomalousIds.has(node.id);

    ctx.save();

    // pulsing ring for anomalous nodes
    if (isAnomaly) {
      const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 400);
      const ringR = r + 4 + pulse * 6;
      ctx.beginPath();
      ctx.arc(x, y, ringR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,77,77,${0.3 + pulse * 0.5})`; // error
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // second outer ring
      const ring2R = r + 10 + pulse * 10;
      ctx.beginPath();
      ctx.arc(x, y, ring2R, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,77,77,${0.1 + pulse * 0.2})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // glow
    ctx.shadowColor = isAnomaly ? "#ff4d4d" : color;
    ctx.shadowBlur = isAnomaly ? 20 : 8;

    // fill
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = isAnomaly ? "#ff4d4d" : color;
    ctx.fill();

    ctx.shadowBlur = 0;

    // label — show when zoomed in enough or node is large/anomalous
    const label = node.label ?? node.id;
    const showLabel = globalScale > 1.5 || r > 8 || isAnomaly;

    if (showLabel) {
      const fontSize = Math.max(8, Math.min(11, r * 1.1)) / globalScale;
      ctx.font = `${fontSize}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";

      // truncate long labels
      const maxChars = 24;
      const displayLabel = label.length > maxChars ? label.slice(0, maxChars) + "…" : label;

      // text background
      const textWidth = ctx.measureText(displayLabel).width;
      const padding = 2 / globalScale;
      ctx.fillStyle = "rgba(10,13,15,0.85)"; // background at 85%
      ctx.fillRect(
        x - textWidth / 2 - padding,
        y + r + 3 / globalScale,
        textWidth + padding * 2,
        fontSize + padding * 2
      );

      ctx.fillStyle = isAnomaly ? "#ff4d4d" : "#c8e8e0"; // error or body text
      ctx.fillText(displayLabel, x, y + r + 3 / globalScale + padding);
    }

    ctx.restore();
  };
}
