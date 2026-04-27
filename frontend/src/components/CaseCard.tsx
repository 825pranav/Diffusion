"use client";

import { useState } from "react";
import {
  CaseFileSummary,
  CaseFileDetail,
  fetchCaseDetail,
} from "@/hooks/useCaseFiles";

const CLASSIFICATION_COLOR: Record<string, string> = {
  organic: "#22c55e",
  coordinated_amplification: "#ef4444",
  uncertain: "#eab308",
};

const PLATFORM_COLOR: Record<string, string> = {
  reddit: "#f97316",
  hn: "#eab308",
  github: "#a855f7",
};

function relativeTime(iso: string): string {
  const delta = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(delta / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function RagasBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <span className="text-zinc-600 w-36 shrink-0">{label}</span>
      <div className="flex-1 h-1 bg-surface-3 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${pct}%`,
            backgroundColor: value >= 0.8 ? "#22c55e" : value >= 0.6 ? "#eab308" : "#ef4444",
          }}
        />
      </div>
      <span className="text-zinc-500 w-8 text-right">{pct}</span>
    </div>
  );
}

interface Props {
  summary: CaseFileSummary;
}

export default function CaseCard({ summary }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [detail, setDetail] = useState<CaseFileDetail | null>(null);
  const [fetching, setFetching] = useState(false);

  const classColor = CLASSIFICATION_COLOR[summary.classification] ?? "#eab308";
  const platColor = PLATFORM_COLOR[summary.platform_origin] ?? "#3b82f6";
  const confPct = Math.round(summary.confidence * 100);

  async function toggle() {
    if (expanded) {
      setExpanded(false);
      return;
    }
    setExpanded(true);
    if (!detail) {
      setFetching(true);
      const d = await fetchCaseDetail(summary.id);
      setDetail(d);
      setFetching(false);
    }
  }

  return (
    <div
      className={`border rounded-lg transition-all duration-150 ${
        expanded ? "border-surface-3 bg-surface-2" : "border-surface-3 bg-surface-1 hover:bg-surface-2"
      }`}
    >
      {/* header row */}
      <button
        onClick={toggle}
        className="w-full text-left px-4 py-3 flex flex-col gap-2"
      >
        {/* top line */}
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs text-zinc-300 truncate flex-1 font-medium">
            {summary.trend}
          </span>

          {summary.needs_review && (
            <span className="shrink-0 text-[10px] uppercase tracking-wider text-neon-red border border-neon-red/40 px-1.5 py-0.5 rounded">
              review
            </span>
          )}

          <span
            className="shrink-0 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider"
            style={{
              color: platColor,
              backgroundColor: `${platColor}18`,
              border: `1px solid ${platColor}44`,
            }}
          >
            {summary.platform_origin}
          </span>
        </div>

        {/* bottom line */}
        <div className="flex items-center gap-3">
          {/* classification */}
          <span className="text-[11px] font-medium" style={{ color: classColor }}>
            {summary.classification.replace("_", " ")}
          </span>

          {/* confidence bar */}
          <div className="flex items-center gap-1.5 flex-1">
            <div className="flex-1 h-0.5 bg-surface-3 rounded-full overflow-hidden max-w-24">
              <div
                className="h-full rounded-full"
                style={{ width: `${confPct}%`, backgroundColor: classColor }}
              />
            </div>
            <span className="text-[11px] text-zinc-600">{confPct}%</span>
          </div>

          <span className="text-[11px] text-zinc-700 ml-auto shrink-0">
            {relativeTime(summary.created_at)}
          </span>

          <span className="text-zinc-700 text-[11px] shrink-0">
            {expanded ? "▲" : "▼"}
          </span>
        </div>
      </button>

      {/* expanded detail */}
      {expanded && (
        <div className="px-4 pb-4 flex flex-col gap-4 border-t border-surface-3 pt-3">
          {fetching && (
            <div className="flex gap-0.5 py-1">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-1 h-1 rounded-full bg-zinc-600 animate-bounce"
                  style={{ animationDelay: `${i * 150}ms` }}
                />
              ))}
            </div>
          )}

          {detail && (
            <>
              {/* signals */}
              {detail.signals.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">
                    signals
                  </span>
                  {detail.signals.map((s, i) => (
                    <div key={i} className="flex gap-2 text-xs text-zinc-400">
                      <span className="text-zinc-700 shrink-0">—</span>
                      {s}
                    </div>
                  ))}
                </div>
              )}

              {/* similar past cases */}
              {detail.similar_past_cases.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">
                    similar past cases
                  </span>
                  {detail.similar_past_cases.map((c, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 text-[11px]"
                    >
                      <span className="text-zinc-500 truncate flex-1">{c.trend}</span>
                      <span className="text-zinc-700 shrink-0">
                        {Math.round(c.similarity * 100)}% match
                      </span>
                      <span
                        className="shrink-0 text-[10px] px-1.5 py-0.5 rounded"
                        style={{
                          color: CLASSIFICATION_COLOR[c.outcome] ?? "#eab308",
                          backgroundColor: `${CLASSIFICATION_COLOR[c.outcome] ?? "#eab308"}18`,
                        }}
                      >
                        {c.outcome.replace("_", " ")}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* ragas scores */}
              {Object.keys(detail.ragas_scores).length > 0 && (
                <div className="flex flex-col gap-2">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">
                    ragas evaluation
                  </span>
                  <div className="flex flex-col gap-1.5 text-[11px] font-mono">
                    {detail.ragas_scores.retrieval_relevance !== undefined && (
                      <RagasBar
                        label="retrieval relevance"
                        value={detail.ragas_scores.retrieval_relevance}
                      />
                    )}
                    {detail.ragas_scores.reasoning_consistency !== undefined && (
                      <RagasBar
                        label="reasoning consistency"
                        value={detail.ragas_scores.reasoning_consistency}
                      />
                    )}
                    {detail.ragas_scores.confidence_calibration !== undefined && (
                      <RagasBar
                        label="confidence calibration"
                        value={detail.ragas_scores.confidence_calibration}
                      />
                    )}
                  </div>
                </div>
              )}

              <div className="text-[11px] text-zinc-700">
                {detail.agent_reasoning_steps} reasoning step
                {detail.agent_reasoning_steps !== 1 ? "s" : ""}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
