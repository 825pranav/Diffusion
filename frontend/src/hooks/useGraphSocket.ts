"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export interface GraphNode {
  id: string;
  platform: string;
  degree: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  platform: string;
  ts: string;
}

interface WsMessage {
  source_id: string;
  target_id: string;
  platform: string;
  ts: string;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/graph";
const BACKOFF_MAX = 30_000;

export function useGraphSocket() {
  const [nodes, setNodes] = useState<Map<string, GraphNode>>(new Map());
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [connected, setConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(1_000);
  const deadRef = useRef(false);

  const connect = useCallback(() => {
    if (deadRef.current) return;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      backoffRef.current = 1_000;
    };

    ws.onmessage = (ev) => {
      try {
        const msg: WsMessage = JSON.parse(ev.data);
        const { source_id, target_id, platform, ts } = msg;

        setNodes((prev) => {
          const next = new Map(prev);
          const src = next.get(source_id) ?? { id: source_id, platform, degree: 0 };
          const tgt = next.get(target_id) ?? { id: target_id, platform, degree: 0 };
          next.set(source_id, { ...src, degree: src.degree + 1 });
          next.set(target_id, { ...tgt, degree: tgt.degree + 1 });
          return next;
        });

        setEdges((prev) => [...prev, { source: source_id, target: target_id, platform, ts }]);
      } catch {
        // malformed frame — skip
      }
    };

    ws.onclose = () => {
      setConnected(false);
      if (!deadRef.current) {
        const delay = backoffRef.current;
        backoffRef.current = Math.min(delay * 2, BACKOFF_MAX);
        setTimeout(connect, delay);
      }
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    deadRef.current = false;
    connect();
    return () => {
      deadRef.current = true;
      wsRef.current?.close();
    };
  }, [connect]);

  return { nodes, edges, connected };
}
