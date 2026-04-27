"use client";

import { useEffect, useState } from "react";

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

const PLATFORM_COLOR: Record<string, string> = {
  reddit: "#f97316",
  hn: "#eab308",
  github: "#a855f7",
};

interface Props {
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export default function AnomalySelector({ selectedId, onSelect }: Props) {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function fetch_() {
      try {
        const res = await fetch(
          `${API_URL}/anomalies?investigated=false&limit=20`,
        );
        if (!res.ok) return;
        const data: Anomaly[] = await res.json();
        if (!cancelled) {
          setAnomalies(data);
          setLoading(false);
        }
      } catch {
        // backend unreachable — keep previous list
      }
    }

    fetch_();
    const id = setInterval(fetch_, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col gap-2 w-full">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-10 rounded-lg bg-surface-2 animate-pulse border border-surface-3"
            style={{ animationDelay: `${i * 100}ms` }}
          />
        ))}
      </div>
    );
  }

  if (anomalies.length === 0) {
    return (
      <p className="text-xs text-zinc-600 tracking-widest uppercase py-4">
        no uninvestigated anomalies
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-1.5 w-full overflow-y-auto max-h-full">
      {anomalies.map((a) => {
        const color = PLATFORM_COLOR[a.platform] ?? "#3b82f6";
        const active = a.id === selectedId;
        return (
          <button
            key={a.id}
            onClick={() => onSelect(a.id)}
            className={`w-full text-left px-3 py-2.5 rounded-lg border text-xs transition-all duration-150 ${
              active
                ? "border-neon-purple bg-surface-2"
                : "border-surface-3 bg-surface-1 hover:bg-surface-2 hover:border-surface-3"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span
                className="truncate font-medium"
                style={{ color: active ? "#a855f7" : "#a1a1aa" }}
              >
                {a.node_id}
              </span>
              <span
                className="shrink-0 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider"
                style={{
                  color,
                  backgroundColor: `${color}18`,
                  border: `1px solid ${color}44`,
                }}
              >
                {a.platform}
              </span>
            </div>
            <div className="flex items-center gap-3 mt-1 text-[11px] text-zinc-600">
              <span>z {a.z_score.toFixed(2)}</span>
              <span>vel {a.velocity.toFixed(1)}</span>
              <span className="ml-auto">
                {new Date(a.detected_at).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
