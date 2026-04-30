"use client";

import { useEffect, useRef } from "react";
import { AgentEvent } from "@/hooks/useAgentStream";
import { CLASSIFICATION_COLOR } from "@/lib/colors";

interface Props {
  events: AgentEvent[];
  streaming: boolean;
  done: boolean;
}

function timestamp(ms: number) {
  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function EventRow({ ev }: { ev: AgentEvent }) {
  switch (ev.type) {
    case "started":
      return (
        <div className="flex gap-3 items-start">
          <span className="text-zinc-600 shrink-0">{timestamp(ev.ts)}</span>
          <span className="text-neon-purple">▶</span>
          <span className="text-zinc-300">
            investigating{" "}
            <span className="text-neon-purple">{ev.node_id}</span>
            {" — "}
            <span className="text-zinc-500">{ev.platform}</span>
            <span className="text-zinc-600">
              {" "}z={ev.z_score?.toFixed(2)} vel={ev.velocity?.toFixed(1)}
            </span>
          </span>
        </div>
      );

    case "tool_call":
      return (
        <div className="flex gap-3 items-start">
          <span className="text-zinc-600 shrink-0">{timestamp(ev.ts)}</span>
          <span className="text-neon-yellow">⟶</span>
          <span>
            <span className="text-neon-yellow">{ev.tool}</span>
            {ev.input !== undefined && (
              <span className="text-zinc-600 ml-2">
                {JSON.stringify(ev.input)}
              </span>
            )}
          </span>
        </div>
      );

    case "tool_result":
      return (
        <div className="flex gap-3 items-start">
          <span className="text-zinc-600 shrink-0">{timestamp(ev.ts)}</span>
          <span className="text-neon-blue">⟵</span>
          <span className="text-zinc-500 break-all">
            {typeof ev.output === "string"
              ? ev.output
              : JSON.stringify(ev.output)}
          </span>
        </div>
      );

    case "completed": {
      const color =
        CLASSIFICATION_COLOR[ev.classification ?? "uncertain"] ?? "#eab308";
      return (
        <div className="flex gap-3 items-start">
          <span className="text-zinc-600 shrink-0">{timestamp(ev.ts)}</span>
          <span style={{ color }}>■</span>
          <span>
            <span style={{ color }}>{ev.classification}</span>
            <span className="text-zinc-500 ml-2">
              confidence {((ev.confidence ?? 0) * 100).toFixed(0)}%
            </span>
            {ev.needs_review && (
              <span className="ml-2 text-[10px] uppercase tracking-wider text-neon-red border border-neon-red/40 px-1.5 py-0.5 rounded">
                review
              </span>
            )}
          </span>
        </div>
      );
    }

    case "timeout":
      return (
        <div className="flex gap-3 items-start">
          <span className="text-zinc-600 shrink-0">{timestamp(ev.ts)}</span>
          <span className="text-zinc-600">✕</span>
          <span className="text-zinc-600">stream timed out</span>
        </div>
      );

    default:
      return null;
  }
}

export default function AgentLog({ events, streaming, done }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  if (events.length === 0 && !streaming) {
    return (
      <div className="flex items-center justify-center h-full text-xs text-zinc-700 tracking-widest uppercase">
        select an anomaly to begin
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5 h-full overflow-y-auto px-4 py-4 text-xs font-mono">
      {events.map((ev, i) => (
        <EventRow key={i} ev={ev} />
      ))}

      {streaming && (
        <div className="flex gap-3 items-center">
          <span className="text-zinc-600">{timestamp(Date.now())}</span>
          <span className="flex gap-0.5">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="w-1 h-1 rounded-full bg-neon-purple animate-bounce"
                style={{ animationDelay: `${i * 150}ms` }}
              />
            ))}
          </span>
        </div>
      )}

      {done && events.length > 0 && (
        <div className="text-zinc-700 text-[11px] tracking-widest uppercase pt-1">
          — investigation complete —
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
