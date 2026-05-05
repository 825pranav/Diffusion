"use client";

import { useState, useEffect } from "react";
import GridBackground from "@/components/GridBackground";
import AnomalySelector from "@/components/AnomalySelector";
import AgentLog from "@/components/AgentLog";
import { useAgentStream } from "@/hooks/useAgentStream";
import { useConnection } from "@/components/ConnectionProvider";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function AgentPage() {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const { events, streaming, done, start, reset } = useAgentStream();
  const { setConnected } = useConnection();

  // reflect SSE connection state into the global top-bar indicator
  useEffect(() => {
    setConnected(streaming);
  }, [streaming, setConnected]);

  // auto-select the newest uninvestigated anomaly on first load
  useEffect(() => {
    async function autoSelect() {
      try {
        const res = await fetch(
          `${API_URL}/anomalies?investigated=false&limit=1`,
        );
        if (!res.ok) return;
        const data: { id: number }[] = await res.json();
        if (data.length > 0) setSelectedId(data[0].id);
      } catch {
        // backend unreachable — leave selector empty
      }
    }
    autoSelect();
  }, []);

  function handleSelect(id: number) {
    if (id === selectedId && streaming) return;
    setSelectedId(id);
    reset();
  }

  function handleInvestigate() {
    if (selectedId === null) return;
    start(selectedId);
  }

  return (
    <div className="relative w-full h-full flex overflow-hidden">
      <GridBackground />

      {/* sidebar — anomaly list */}
      <aside className="relative z-10 w-72 shrink-0 flex flex-col gap-4 border-r border-surface-3 bg-surface-1/80 backdrop-blur-sm p-4 overflow-hidden">
        <div className="flex items-center justify-between shrink-0">
          <span className="text-[10px] tracking-widest uppercase text-zinc-600">
            uninvestigated
          </span>
          <span className="text-[10px] text-zinc-700">15s poll</span>
        </div>
        <AnomalySelector selectedId={selectedId} onSelect={handleSelect} />
      </aside>

      {/* main — log + controls */}
      <div className="relative z-10 flex flex-col flex-1 min-w-0 overflow-hidden">
        {/* controls strip */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-surface-3 bg-surface-1/60 backdrop-blur-sm shrink-0">
          <button
            onClick={handleInvestigate}
            disabled={selectedId === null || streaming}
            className={`px-3 py-1.5 rounded text-xs tracking-widest uppercase transition-all duration-150 ${
              selectedId !== null && !streaming
                ? "bg-neon-purple/10 border border-neon-purple/40 text-neon-purple hover:bg-neon-purple/20"
                : "border border-surface-3 text-zinc-700 cursor-not-allowed"
            }`}
          >
            {streaming ? "investigating…" : "investigate"}
          </button>

          {(events.length > 0 || done) && (
            <button
              onClick={reset}
              disabled={streaming}
              className="px-3 py-1.5 rounded text-xs tracking-widest uppercase border border-surface-3 text-zinc-600 hover:text-zinc-400 transition-colors disabled:cursor-not-allowed"
            >
              clear
            </button>
          )}

          {selectedId !== null && (
            <span className="text-[11px] text-zinc-700 ml-auto">
              anomaly #{selectedId}
            </span>
          )}
        </div>

        {/* log */}
        <div className="flex-1 overflow-hidden">
          <AgentLog events={events} streaming={streaming} done={done} />
        </div>
      </div>
    </div>
  );
}
