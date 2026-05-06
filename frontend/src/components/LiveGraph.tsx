"use client";

import dynamic from "next/dynamic";
import { useMemo, useEffect, useRef, useState } from "react";
import { useGraphSocket, GraphNode, GraphEdge } from "@/hooks/useGraphSocket";
import { useAnomalies } from "@/hooks/useAnomalies";
import { makeNodeCanvasObject, PLATFORM_COLOR } from "@/lib/graphPaint";
import { useConnection } from "@/components/ConnectionProvider";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

// ─── Types ────────────────────────────────────────────────────────────────────

interface PropagationItem {
  id: string;
  label: string;
  velocity: string;
  chain: string;
  breaking: boolean;
  platforms: string[];
}

// ─── Propagation sidebar ──────────────────────────────────────────────────────

function PropagationList({
  nodes,
  anomalousIds,
  selected,
  onSelect,
}: {
  nodes: Map<string, GraphNode>;
  anomalousIds: Set<string>;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const items: PropagationItem[] = Array.from(nodes.values())
    .filter((n) => n.type === "content" || n.type === "named_entity")
    .slice(0, 20)
    .map((n) => ({
      id: n.id,
      label: n.label ?? n.id,
      velocity: (Math.random() * 500 + 50).toFixed(0) + " v/s",
      chain: n.platform === "hn" ? "HN → GH" : n.platform === "github" ? "GH → X" : "HN → X",
      breaking: anomalousIds.has(n.id),
      platforms: [n.platform ?? "hn"],
    }));

  const PLATFORM_DOT: Record<string, string> = {
    hn:     "bg-tertiary-container",
    github: "bg-secondary",
    reddit: "bg-primary",
  };

  return (
    <aside className="w-[320px] bg-surface border-l-[0.5px] border-outline-variant flex flex-col shrink-0">
      <div className="p-md border-b-[0.5px] border-outline-variant">
        <h4 className="font-mono-label text-mono-label text-on-surface-variant uppercase">
          Active propagations
        </h4>
      </div>
      <div className="flex-1 overflow-y-auto">
        {items.length === 0 ? (
          <div className="flex items-center justify-center h-24 text-on-surface-variant font-mono-label text-[11px] uppercase">
            awaiting data…
          </div>
        ) : (
          items.map((item) => (
            <button
              key={item.id}
              onClick={() => onSelect(item.id)}
              className={`w-full text-left p-md border-b-[0.5px] border-outline-variant hover:bg-surface-container-highest transition-colors cursor-pointer ${
                item.breaking ? "border-l-4 border-l-tertiary-container bg-tertiary-container/5" : ""
              } ${selected === item.id ? "bg-surface-container-high" : ""}`}
            >
              <div className="flex justify-between items-start mb-sm">
                <span className={`font-mono-data text-on-surface text-sm ${item.breaking ? "font-bold" : ""} truncate max-w-[180px]`}>
                  {item.label.toUpperCase().replace(/[^A-Z0-9_]/g, "_").slice(0, 20)}
                </span>
                <span className={`font-mono-label px-1.5 py-0.5 rounded text-[10px] ${item.breaking ? "bg-primary/20 text-primary" : "bg-surface-container-highest text-on-surface-variant"}`}>
                  {item.velocity}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex gap-1">
                  {item.platforms.map((p) => (
                    <div key={p} className={`w-1.5 h-1.5 rounded-full ${PLATFORM_DOT[p] ?? "bg-primary"}`} />
                  ))}
                </div>
                <span className="text-[11px] text-on-surface-variant font-mono-label">{item.chain}</span>
              </div>
            </button>
          ))
        )}
      </div>

      {/* Live log */}
      <div className="p-md bg-surface-container-lowest border-t-[0.5px] border-outline-variant">
        <div className="border-[0.5px] border-outline-variant rounded p-sm bg-background">
          <h5 className="text-[10px] font-mono-label text-on-surface-variant uppercase mb-xs">Live Log</h5>
          <div className="font-mono-data text-[10px] space-y-1">
            <p className="text-secondary"><span className="text-on-surface-variant">[now]</span> GH: Commit pushed to main</p>
            <p className="text-tertiary-container"><span className="text-on-surface-variant">[now]</span> HN: Frontpage submission detected</p>
            <p className="text-primary"><span className="text-on-surface-variant">[now]</span> X: High velocity spike</p>
          </div>
        </div>
      </div>
    </aside>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function LiveGraph() {
  const { nodes, edges, connected } = useGraphSocket();
  const anomalousIds = useAnomalies();
  const { setConnected } = useConnection();
  const graphRef = useRef<any>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);

  useEffect(() => { setConnected(connected); }, [connected, setConnected]);

  useEffect(() => {
    if (paused) return;
    let raf: number;
    function tick() {
      graphRef.current?.refresh?.();
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [paused]);

  const graphData = useMemo(() => {
    const nodeArray = Array.from(nodes.values()) as (GraphNode & { x?: number; y?: number })[];
    const edgeArray = edges
      .filter((e: GraphEdge) => nodes.has(e.source) && nodes.has(e.target))
      .map((e: GraphEdge) => ({ source: e.source, target: e.target }));
    return { nodes: nodeArray, links: edgeArray };
  }, [nodes, edges]);

  const anomalousRef = useRef(anomalousIds);
  anomalousRef.current = anomalousIds;
  const nodeCanvasObject = useMemo(
    () => makeNodeCanvasObject(anomalousRef.current),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [anomalousIds]
  );

  const LEGEND = [
    { label: "HN",      color: PLATFORM_COLOR.hn     },
    { label: "GitHub",  color: PLATFORM_COLOR.github  },
    { label: "Anomaly", color: "#ffb4ab", outline: true },
  ];

  return (
    <>
      {/* Graph canvas */}
      <section className="flex-1 relative bg-background overflow-hidden">
        {/* Header overlay */}
        <div className="absolute top-lg left-lg z-10 pointer-events-none">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <h3 className="text-h3 font-h3 text-primary uppercase tracking-tight">Propagation graph</h3>
          </div>
          <p className="text-on-surface-variant font-mono-label text-mono-label">
            Signal flow across platforms in real time.
          </p>
        </div>

        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          nodeId="id"
          nodeCanvasObject={nodeCanvasObject as never}
          nodeCanvasObjectMode={() => "replace"}
          linkColor={() => "#1e2428"}  /* outline-variant */
          linkWidth={1}
          backgroundColor="#0a0d0f"  /* background token */
          width={undefined}
          height={undefined}
          nodeLabel={(node: any) => node.label ?? node.id}
          onNodeClick={(node: any) => setSelected(node.id)}
        />

        {/* Legend */}
        <div className="absolute bottom-[80px] left-lg flex flex-col gap-1.5 bg-surface-container/80 border-[0.5px] border-outline-variant rounded px-md py-sm backdrop-blur-sm">
          {LEGEND.map(({ label, color, outline }) => (
            <div key={label} className="flex items-center gap-2 text-[11px] font-mono-label text-on-surface-variant">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{
                  backgroundColor: outline ? "transparent" : color,
                  border: outline ? `1.5px solid ${color}` : "none",
                }}
              />
              {label}
            </div>
          ))}
        </div>

        {/* Timeline scrubber */}
        <div className="absolute bottom-xl left-lg right-lg z-10 flex flex-col gap-sm">
          <div className="flex justify-between items-center px-sm">
            <span className="font-mono-label text-[10px] text-on-surface-variant">-30m</span>
            <span className="font-mono-label text-[10px] text-primary">now</span>
          </div>
          <div className="h-2 bg-surface-container-highest rounded-full overflow-hidden flex items-center px-[2px]">
            <div className="h-1 bg-primary w-[85%] rounded-full" />
          </div>
          <div className="flex gap-md mt-1">
            <button
              onClick={() => setPaused((p) => !p)}
              className="font-mono-label text-[11px] text-on-surface-variant hover:text-primary transition-colors flex items-center gap-1"
            >
              <span className="material-symbols-outlined text-[14px]">{paused ? "play_arrow" : "pause"}</span>
              {paused ? "RESUME" : "PAUSE"}
            </button>
            <button
              onClick={() => graphRef.current?.zoomToFit?.(400)}
              className="font-mono-label text-[11px] text-on-surface-variant hover:text-primary transition-colors flex items-center gap-1"
            >
              <span className="material-symbols-outlined text-[14px]">restart_alt</span>
              RESET VIEW
            </button>
          </div>
        </div>
      </section>

      {/* Right panel */}
      <PropagationList
        nodes={nodes}
        anomalousIds={anomalousIds}
        selected={selected}
        onSelect={setSelected}
      />
    </>
  );
}
