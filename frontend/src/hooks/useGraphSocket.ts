"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export interface GraphNode {
  id: string;
  platform: string;
  degree: number;
  label?: string;
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
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
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

  // Bootstrap with existing nodes + edges from REST API on mount
  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/nodes?limit=100&since_minutes=1440`).then((r) => r.json()),
      fetch(`${API_URL}/edges?limit=200&since_minutes=1440`).then((r) => r.json()),
    ])
      .then(([nodeData, edgeData]: [{ id: string; platform: string; in_degree: number; label?: string }[], { source_id: string; target_id: string; platform: string; ts: string }[]]) => {
        setNodes((prev) => {
          const next = new Map(prev);
          // add known nodes
          for (const n of nodeData) {
            next.set(n.id, { id: n.id, platform: n.platform, degree: n.in_degree, label: n.label });
          }
          // add any nodes referenced in edges but missing from node list
          for (const e of edgeData) {
            if (!next.has(e.source_id)) next.set(e.source_id, { id: e.source_id, platform: e.platform, degree: 0 });
            if (!next.has(e.target_id)) next.set(e.target_id, { id: e.target_id, platform: e.platform, degree: 0 });
          }
          return next;
        });
        setEdges(edgeData.map((e) => ({ source: e.source_id, target: e.target_id, platform: e.platform, ts: e.ts })));
      })
      .catch(() => {});
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
