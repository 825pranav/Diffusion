"use client";

import { useState, useEffect, useRef } from "react";
import { useAgentStream, AgentEvent } from "@/hooks/useAgentStream";
import { useConnection } from "@/components/ConnectionProvider";
import { PLATFORM_COLOR, CLASSIFICATION_COLOR } from "@/lib/colors";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const POLL_MS = 15_000;

interface Anomaly {
  id: number;
  node_id: string;
  platform: string;
  z_score: number;
  velocity: number;
  detected_at: string;
  investigated: boolean;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ts(ms: number) {
  return new Date(ms).toISOString().replace("T", " ").substring(0, 23);
}

function relativeAge(iso: string): string {
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h`;
}

// ─── Terminal log line ────────────────────────────────────────────────────────

function LogLine({ ev }: { ev: AgentEvent }) {
  const time = <span style={{ color: "#3a5045" }}>[{ts(ev.ts)}]</span>;

  switch (ev.type) {
    case "started":
      return (
        <div className="flex gap-sm py-0.5 bg-primary-container border-l-2 border-primary -ml-md pl-md">
          {time}
          <span className="text-primary">ALERT:</span>
          <span className="text-on-surface-variant">
            Investigating{" "}
            <span className="text-primary">{ev.node_id}</span>
            {" — "}
            <span className="text-on-surface-variant">{ev.platform}</span>
            <span className="text-outline"> z={ev.z_score?.toFixed(2)} vel={ev.velocity?.toFixed(1)}</span>
          </span>
        </div>
      );
    case "tool_call":
      return (
        <div className="flex gap-sm py-0.5">
          {time}
          <span className="text-secondary">INFO:</span>
          <span className="text-on-surface-variant">
            → <span className="text-secondary">{ev.tool}</span>
            {ev.input !== undefined && (
              <span className="text-outline ml-2">{JSON.stringify(ev.input)}</span>
            )}
          </span>
        </div>
      );
    case "tool_result":
      return (
        <div className="flex gap-sm py-0.5">
          {time}
          <span className="text-secondary">DATA:</span>
          <span className="text-on-surface-variant break-all">
            {typeof ev.output === "string" ? ev.output : JSON.stringify(ev.output)}
          </span>
        </div>
      );
    case "completed": {
      const color = CLASSIFICATION_COLOR[ev.classification ?? "uncertain"] ?? "#ffd7b5";
      return (
        <div className="flex gap-sm py-0.5 bg-primary-container border-l-2 border-primary -ml-md pl-md">
          {time}
          <span className="text-primary">RESULT:</span>
          <span style={{ color }}>{ev.classification}</span>
          <span className="text-on-surface-variant">
            confidence {((ev.confidence ?? 0) * 100).toFixed(0)}%
          </span>
          {ev.needs_review && (
            <span className="text-error border border-error/40 px-xs font-mono-label text-[10px] uppercase">
              review
            </span>
          )}
        </div>
      );
    }
    case "timeout":
      return (
        <div className="flex gap-sm py-0.5">
          {time}
          <span className="text-error">ERR:</span>
          <span className="text-on-surface-variant">stream timed out</span>
        </div>
      );
    default:
      return null;
  }
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function AgentPage() {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [paused, setPaused] = useState(false);
  const { events, streaming, done, start, reset } = useAgentStream();
  const { setConnected } = useConnection();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { setConnected(streaming); }, [streaming, setConnected]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  // Poll anomalies
  useEffect(() => {
    let cancelled = false;
    async function fetch_() {
      try {
        const res = await fetch(`${API_URL}/anomalies?investigated=false&limit=20`);
        if (!res.ok) return;
        const data: Anomaly[] = await res.json();
        if (!cancelled) {
          setAnomalies(data);
          if (data.length > 0 && selectedId === null) setSelectedId(data[0].id);
        }
      } catch { /* backend unreachable */ }
    }
    fetch_();
    const id = setInterval(fetch_, POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleInvestigate(id: number) {
    setSelectedId(id);
    reset();
    start(id);
  }

  const selectedAnomaly = anomalies.find((a) => a.id === selectedId);

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Terminal panel — 60% */}
      <section className="h-[60%] w-full bg-surface-container-lowest border-t-2 border-primary relative flex flex-col overflow-hidden">
        {/* Terminal header */}
        <div className="flex justify-between items-center px-md py-sm border-b-[0.5px] border-outline-variant/30 bg-surface-container-lowest shrink-0">
          <div className="flex items-center gap-sm">
            <span className="text-primary-fixed-dim font-mono-data text-[11px] uppercase tracking-widest">
              Live Agent Stream
            </span>
            {streaming && (
              <div className="flex gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              </div>
            )}
            {selectedAnomaly && (
              <span className="text-on-surface-variant font-mono-data text-[11px]">
                — anomaly #{selectedId} / {selectedAnomaly.node_id}
              </span>
            )}
          </div>
          <div className="flex items-center gap-sm">
            {(events.length > 0 || done) && (
              <button
                onClick={reset}
                disabled={streaming}
                className="font-mono-label text-[10px] uppercase border border-outline-variant px-sm py-0.5 rounded-sm text-on-surface-variant hover:text-primary transition-colors disabled:opacity-40"
              >
                clear
              </button>
            )}
            <button
              onClick={() => setPaused((p) => !p)}
              className="font-mono-label text-[10px] uppercase border border-outline-variant px-sm py-0.5 rounded-sm text-on-surface-variant hover:text-primary transition-colors"
            >
              {paused ? "resume stream" : "pause stream"}
            </button>
          </div>
        </div>

        {/* Log area */}
        <div className="flex-1 p-md font-mono-data text-[12px] overflow-y-auto leading-relaxed">
          {events.length === 0 && !streaming ? (
            <div className="flex items-center gap-sm mt-1">
              <span className="text-outline">[ready]</span>
              <span className="text-primary">&gt;</span>
              <span className="terminal-cursor" />
            </div>
          ) : (
            <>
              {events.map((ev, i) => <LogLine key={i} ev={ev} />)}
              {streaming && (
                <div className="flex items-center gap-sm mt-1">
                  <span className="text-outline">[{ts(Date.now())}]</span>
                  <span className="text-primary">&gt;</span>
                  <span className="terminal-cursor" />
                </div>
              )}
              {done && (
                <div className="flex gap-sm py-0.5 mt-1">
                  <span className="text-outline">[done]</span>
                  <span className="text-primary">— investigation complete —</span>
                </div>
              )}
            </>
          )}
          <div ref={bottomRef} />
        </div>
      </section>

      {/* Triage queue — 40% */}
      <section className="h-[40%] bg-surface-container-low border-t-[0.5px] border-outline-variant flex flex-col">
        <div className="px-lg py-md flex items-center justify-between border-b-[0.5px] border-outline-variant/30 shrink-0">
          <div className="flex items-center gap-md">
            <h2 className="font-h3 text-h3 text-on-surface">Anomaly triage</h2>
            <span className="bg-tertiary-container/20 text-tertiary-fixed-dim font-mono-label px-2 py-0.5 rounded-full border border-tertiary-container/30 text-[10px]">
              {anomalies.length} UNRESOLVED
            </span>
          </div>
        </div>

        <div className="flex-1 overflow-auto">
          <table className="w-full text-left font-body-sm">
            <thead className="sticky top-0 bg-surface-container-low border-b-[0.5px] border-outline-variant/30 z-10">
              <tr className="font-mono-label text-on-surface-variant uppercase tracking-tighter text-[10px]">
                <th className="px-lg py-sm font-medium">Status</th>
                <th className="px-md py-sm font-medium">Topic / Vector</th>
                <th className="px-md py-sm font-medium">Platform</th>
                <th className="px-md py-sm font-medium">Velocity Δ</th>
                <th className="px-md py-sm font-medium text-right">Age</th>
                <th className="px-lg py-sm font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/20">
              {anomalies.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-lg py-xl text-center font-mono-label text-[11px] text-on-surface-variant uppercase">
                    no uninvestigated anomalies
                  </td>
                </tr>
              ) : (
                anomalies.map((a) => {
                  const color = PLATFORM_COLOR[a.platform] ?? "#6ef6c7";
                  const isSelected = a.id === selectedId;
                  return (
                    <tr
                      key={a.id}
                      className={`hover:bg-surface-container-highest transition-colors ${isSelected ? "bg-tertiary-container/5 border-l-2 border-tertiary-container/80" : ""}`}
                    >
                      <td className="px-lg py-md">
                        <div
                          className={`w-2 h-2 rounded-full ${isSelected && streaming ? "animate-pulse" : ""}`}
                          style={{ backgroundColor: isSelected ? "#6ef6c7" : color }}
                        />
                      </td>
                      <td className="px-md py-md">
                        <div className="font-semibold text-on-surface text-sm">{a.node_id}</div>
                        <div className="text-[11px] text-on-surface-variant font-mono-data opacity-60">
                          z={a.z_score.toFixed(2)}
                        </div>
                      </td>
                      <td className="px-md py-md">
                        <span
                          className="font-mono-label text-[10px] px-sm py-0.5 rounded uppercase"
                          style={{ color, backgroundColor: `${color}18`, border: `1px solid ${color}44` }}
                        >
                          {a.platform}
                        </span>
                      </td>
                      <td className="px-md py-md font-mono-data text-primary">
                        +{a.velocity.toFixed(1)}
                      </td>
                      <td className="px-md py-md text-right text-on-surface-variant font-mono-data text-[12px]">
                        {relativeAge(a.detected_at)}
                      </td>
                      <td className="px-lg py-md text-right space-x-md">
                        <button
                          onClick={() => handleInvestigate(a.id)}
                          disabled={streaming}
                          className="text-primary hover:underline font-mono-label text-[11px] uppercase tracking-wider disabled:opacity-40"
                        >
                          Investigate
                        </button>
                        <button className="text-on-surface-variant hover:text-on-surface font-mono-label text-[11px] uppercase tracking-wider">
                          Dismiss
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
