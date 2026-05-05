"use client";

import { useEffect, useState, useCallback } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const POLL_MS = 30_000;

export interface CaseFileSummary {
  id: number;
  anomaly_event_id: number | null;
  trend: string;
  label: string;
  platform_origin: string;
  detected_at: string;
  classification: string;
  confidence: number;
  needs_review: boolean;
  created_at: string;
  z_score: number | null;
  velocity: number | null;
}

export interface SimilarCase {
  trend: string;
  similarity: number;
  outcome: string;
}

export interface RagasScores {
  retrieval_relevance?: number;
  reasoning_consistency?: number;
  confidence_calibration?: number;
}

export interface CaseFileDetail extends CaseFileSummary {
  signals: string[];
  similar_past_cases: SimilarCase[];
  ragas_scores: RagasScores;
  agent_reasoning_steps: number;
  reasoning_steps_detail: string[];
}

export interface CaseFilters {
  needsReview?: boolean;
  classification?: string;
}

export function useCaseFiles(filters: CaseFilters = {}) {
  const [cases, setCases] = useState<CaseFileSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const fetch_ = useCallback(async () => {
    const params = new URLSearchParams({ limit: "50" });
    if (filters.needsReview !== undefined)
      params.set("needs_review", String(filters.needsReview));
    if (filters.classification)
      params.set("classification", filters.classification);

    try {
      const res = await fetch(`${API_URL}/case-files?${params}`);
      if (!res.ok) return;
      const data: CaseFileSummary[] = await res.json();
      setCases(data);
      setLoading(false);
    } catch {
      // backend unreachable — keep previous state
    }
  }, [filters.needsReview, filters.classification]);

  useEffect(() => {
    setLoading(true);
    fetch_();
    const id = setInterval(fetch_, POLL_MS);
    return () => clearInterval(id);
  }, [fetch_]);

  return { cases, loading, refresh: fetch_ };
}

export async function fetchCaseDetail(id: number): Promise<CaseFileDetail | null> {
  try {
    const res = await fetch(`${API_URL}/case-files/${id}`);
    if (!res.ok) return null;
    const data = await res.json();
    // signals and similar_past_cases may come back as JSON strings from asyncpg
    if (typeof data.signals === "string") data.signals = JSON.parse(data.signals);
    if (typeof data.similar_past_cases === "string") data.similar_past_cases = JSON.parse(data.similar_past_cases);
    if (typeof data.ragas_scores === "string") data.ragas_scores = JSON.parse(data.ragas_scores);
    if (typeof data.reasoning_steps_detail === "string") data.reasoning_steps_detail = JSON.parse(data.reasoning_steps_detail);
    if (!Array.isArray(data.reasoning_steps_detail)) data.reasoning_steps_detail = [];
    return data;
  } catch {
    return null;
  }
}
