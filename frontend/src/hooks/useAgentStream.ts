"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type AgentEventType =
  | "started"
  | "tool_call"
  | "tool_result"
  | "completed"
  | "timeout";

export interface AgentEvent {
  type: AgentEventType;
  ts: number; // client-side timestamp (ms)
  // started
  node_id?: string;
  platform?: string;
  z_score?: number;
  velocity?: number;
  // tool_call
  tool?: string;
  input?: unknown;
  // tool_result
  output?: unknown;
  // completed
  classification?: string;
  confidence?: number;
  needs_review?: boolean;
}

export interface UseAgentStreamResult {
  events: AgentEvent[];
  streaming: boolean;
  done: boolean;
  start: (anomalyId: number) => void;
  reset: () => void;
}

export function useAgentStream(): UseAgentStreamResult {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [done, setDone] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  const reset = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    setEvents([]);
    setStreaming(false);
    setDone(false);
  }, []);

  const start = useCallback(
    (anomalyId: number) => {
      reset();

      const es = new EventSource(`${API_URL}/sse/agent/${anomalyId}`);
      esRef.current = es;
      setStreaming(true);

      es.onmessage = (ev) => {
        try {
          const raw = JSON.parse(ev.data);
          setEvents((prev) => [...prev, { ...raw, ts: Date.now() }]);
        } catch {
          // malformed frame — skip
        }
      };

      es.addEventListener("done", () => {
        setStreaming(false);
        setDone(true);
        es.close();
      });

      es.addEventListener("timeout", () => {
        setStreaming(false);
        setDone(true);
        es.close();
      });

      es.onerror = () => {
        setStreaming(false);
        es.close();
      };
    },
    [reset],
  );

  // cleanup on unmount
  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  return { events, streaming, done, start, reset };
}
