"use client";

import { useEffect, useRef, useState, useCallback } from "react";

// ─── Types (mirrored from TrendingDashboard) ──────────────────────────────────

type Platform = "hn" | "gh" | "rd";

interface PropagationStep {
  platform: Platform;
  minutesAgo: number;
}

export interface TrendingTopic {
  id: string;
  name: string;
  url?: string;
  platforms: Platform[];
  velocity: number;
  velocityDelta: number;
  trajectory: "rising" | "plateau" | "cooling";
  propagation: PropagationStep[];
  sparkline: number[];
  breaking: boolean;
}

export interface TrendingStats {
  trendingTopics: number;
  trendingDelta: number;
  fastestVelocity: number;
  fastestName: string;
  crossPlatform: number;
  breakingNow: number;
}

type SortKey = "velocity" | "spread" | "newest";
type TimeWindow = "30m" | "2h" | "today";
type PlatformFilter = "all" | Platform;

// ─── Config ───────────────────────────────────────────────────────────────────

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/graph";
const BACKOFF_MAX = 30_000;

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useTrending(
  timeWindow: TimeWindow,
  platformFilter: PlatformFilter,
  sortKey: SortKey
) {
  const [topics, setTopics] = useState<TrendingTopic[]>([]);
  const [stats, setStats] = useState<TrendingStats>({
    trendingTopics: 0,
    trendingDelta: 0,
    fastestVelocity: 0,
    fastestName: "—",
    crossPlatform: 0,
    breakingNow: 0,
  });
  const [wsLive, setWsLive] = useState(false);
  const [loading, setLoading] = useState(true);

  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(1_000);
  const deadRef = useRef(false);

  // ── Fetch ──────────────────────────────────────────────────────────────────

  const fetchTrending = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        window: timeWindow,
        platform: platformFilter,
        sort: sortKey,
      });
      const res = await fetch(`${API_URL}/trending?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setTopics(data.topics ?? []);
      setStats(data.stats ?? {});
    } catch {
      // keep stale data on error
    } finally {
      setLoading(false);
    }
  }, [timeWindow, platformFilter, sortKey]);

  useEffect(() => {
    fetchTrending();
  }, [fetchTrending]);

  // ── WebSocket ──────────────────────────────────────────────────────────────

  const connect = useCallback(() => {
    if (deadRef.current) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsLive(true);
      backoffRef.current = 1_000;
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string);
        const { type, payload } = msg as { type: string; payload: TrendingTopic };

        if (type === "graph_delta" && payload?.id) {
          setTopics((prev) => {
            const idx = prev.findIndex((t) => t.id === payload.id);
            if (idx === -1) return [payload, ...prev];
            const next = [...prev];
            next[idx] = payload;
            return next;
          });
        }

        if (type === "anomaly" && payload?.id) {
          setTopics((prev) =>
            prev.map((t) => (t.id === payload.id ? { ...t, breaking: true } : t))
          );
        }
      } catch {
        // malformed frame — skip
      }
    };

    ws.onclose = () => {
      setWsLive(false);
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

  return { topics, stats, wsLive, loading };
}
