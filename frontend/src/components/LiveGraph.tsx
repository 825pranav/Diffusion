"use client";

import dynamic from "next/dynamic";
import { useMemo, useEffect, useRef } from "react";
import { useGraphSocket, GraphNode, GraphEdge } from "@/hooks/useGraphSocket";
import { useAnomalies } from "@/hooks/useAnomalies";
import { makeNodeCanvasObject, PLATFORM_COLOR } from "@/lib/graphPaint";
import { useConnection } from "@/components/ConnectionProvider";

// react-force-graph-2d uses browser canvas APIs — no SSR
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const EDGE_COLOR = "#2a2a35";

const LEGEND = [
  { label: "Reddit", color: PLATFORM_COLOR.reddit },
  { label: "HN", color: PLATFORM_COLOR.hn },
  { label: "GitHub", color: PLATFORM_COLOR.github },
  { label: "Anomaly", color: "#ef4444", outline: true },
];

export default function LiveGraph() {
  const { nodes, edges, connected } = useGraphSocket();
  const anomalousIds = useAnomalies();
  const { setConnected } = useConnection();
  const graphRef = useRef<any>(null);

  useEffect(() => {
    setConnected(connected);
  }, [connected, setConnected]);

  // re-render every frame so anomaly pulse animation runs
  useEffect(() => {
    let raf: number;
    function tick() {
      graphRef.current?.refresh?.();
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const graphData = useMemo(() => {
    const nodeMap = nodes;
    const nodeArray: (GraphNode & { x?: number; y?: number })[] = Array.from(nodeMap.values());
    // only include edges where both ends exist
    const edgeArray = edges
      .filter((e: GraphEdge) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e: GraphEdge) => ({ source: e.source, target: e.target }));
    return { nodes: nodeArray, links: edgeArray };
  }, [nodes, edges]);

  // nodeCanvasObject recreated when anomalousIds changes — stable ref otherwise
  const anomalousRef = useRef(anomalousIds);
  anomalousRef.current = anomalousIds;
  const nodeCanvasObject = useMemo(
    () => makeNodeCanvasObject(anomalousRef.current),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [anomalousIds]
  );

  return (
    <div className="relative w-full h-full bg-surface">
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeId="id"
        nodeCanvasObject={nodeCanvasObject as never}
        nodeCanvasObjectMode={() => "replace"}
        linkColor={() => EDGE_COLOR}
        linkWidth={1}
        backgroundColor="#0d0d0f"
        width={undefined}
        height={undefined}
        nodeLabel={(node: any) => node.label ?? node.id}
      />

      {/* legend */}
      <div className="absolute bottom-4 left-4 flex flex-col gap-1.5 bg-surface-1/80 border border-surface-3 rounded-lg px-3 py-2 backdrop-blur-sm">
        {LEGEND.map(({ label, color, outline }) => (
          <div key={label} className="flex items-center gap-2 text-xs text-zinc-400">
            <span
              className="w-3 h-3 rounded-full shrink-0"
              style={{
                backgroundColor: outline ? "transparent" : color,
                border: outline ? `2px solid ${color}` : "none",
                boxShadow: `0 0 6px ${color}88`,
              }}
            />
            {label}
          </div>
        ))}
      </div>

      {/* connection badge */}
      <div className="absolute top-3 right-4 flex items-center gap-1.5 text-xs text-zinc-500">
        <span
          className="w-2 h-2 rounded-full"
          style={{
            backgroundColor: connected ? "#22c55e" : "#52525b",
            boxShadow: connected ? "0 0 6px #22c55e" : "none",
          }}
        />
        {connected ? "live" : "reconnecting…"}
      </div>
    </div>
  );
}
