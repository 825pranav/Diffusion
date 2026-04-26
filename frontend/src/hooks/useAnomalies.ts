"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const POLL_INTERVAL = 30_000;

export function useAnomalies(): Set<string> {
  const [anomalousIds, setAnomalousIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;

    async function fetch_() {
      try {
        const res = await fetch(`${API_URL}/anomalies`);
        if (!res.ok) return;
        const data: { node_id: string }[] = await res.json();
        if (!cancelled) {
          setAnomalousIds(new Set(data.map((a) => a.node_id)));
        }
      } catch {
        // backend unreachable — keep previous set
      }
    }

    fetch_();
    const id = setInterval(fetch_, POLL_INTERVAL);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return anomalousIds;
}
