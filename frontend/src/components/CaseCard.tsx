"use client";

import { useState } from "react";
import {
  CaseFileSummary,
  CaseFileDetail,
  fetchCaseDetail,
} from "@/hooks/useCaseFiles";
import { CLASSIFICATION_COLOR, PLATFORM_COLOR } from "@/lib/colors";

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
  const color = value >= 0.75 ? "#22c55e" : value >= 0.5 ? "#eab308" : "#ef4444";
  return (
    <div className="flex items-center gap-2">
      <span className="text-zinc-600 w-40 shrink-0 text-[11px]">{label}</span>
      <div className="flex-1 h-1.5 bg-surface-3 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-[11px] w-8 text-right" style={{ color }}>{pct}</span>
    </div>
  );
}

function StatPill({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col items-center px-3 py-1.5 rounded bg-surface-3/60 border border-surface-3">
      <span className="text-[9px] uppercase tracking-widest text-zinc-600">{label}</span>
      <span className="text-sm font-mono font-semibold" style={{ color: color ?? "#e4e4e7" }}>{value}</span>
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

  // Use the real label if available, fall back to trend ID
  const displayName = summary.label && summary.label !== summary.trend
    ? summary.label
    : summary.trend;
  const isRealLabel = summary.label && summary.label !== summary.trend;

  async function toggle() {
    if (expanded) { setExpanded(false); return; }
    setExpanded(true);
    if (!detail) {
      setFetching(true);
      const d = await fetchCaseDetail(summary.id);
      setDetail(d);
      setFetching(false);
    }
  }

  return (
    <div className={`border rounded-lg transition-all duration-150 ${
      expanded ? "border-surface-3 bg-surface-2" : "border-surface-3 bg-surface-1 hover:bg-surface-2"
    }`}>
      {/* header */}
      <button onClick={toggle} className="w-full text-left px-4 py-3 flex flex-col gap-2">

        {/* title row */}
        <div className="flex items-start gap-2 min-w-0">
          <div className="flex flex-col flex-1 min-w-0">
            <span className="text-sm text-zinc-200 font-medium leading-snug line-clamp-2">
              {displayName}
            </span>
            {isRealLabel && (
              <span className="text-[10px] text-zinc-600 font-mono mt-0.5 truncate">
                {summary.trend}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5 shrink-0 mt-0.5">
            {summary.needs_review && (
              <span className="text-[10px] uppercase tracking-wider text-neon-red border border-neon-red/40 px-1.5 py-0.5 rounded">
                review
              </span>
            )}
            <span
              className="px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider"
              style={{ color: platColor, backgroundColor: `${platColor}18`, border: `1px solid ${platColor}44` }}
            >
              {summary.platform_origin}
            </span>
          </div>
        </div>

        {/* stats row */}
        <div className="flex items-center gap-3">
          {/* classification */}
          <span className="text-[11px] font-semibold tracking-wide" style={{ color: classColor }}>
            {summary.classification.replace(/_/g, " ")}
          </span>

          {/* confidence bar */}
          <div className="flex items-center gap-1.5 flex-1">
            <div className="flex-1 h-1 bg-surface-3 rounded-full overflow-hidden max-w-20">
              <div className="h-full rounded-full" style={{ width: `${confPct}%`, backgroundColor: classColor }} />
            </div>
            <span className="text-[11px]" style={{ color: classColor }}>{confPct}%</span>
          </div>

          {/* trigger stats */}
          {summary.z_score != null && (
            <span className="text-[11px] text-zinc-600 font-mono">
              z={summary.z_score.toFixed(1)}
            </span>
          )}
          {summary.velocity != null && (
            <span className="text-[11px] text-zinc-600 font-mono">
              v={summary.velocity.toFixed(0)}
            </span>
          )}

          <span className="text-[11px] text-zinc-700 shrink-0">{relativeTime(summary.created_at)}</span>
          <span className="text-zinc-600 text-[11px]">{expanded ? "▲" : "▼"}</span>
        </div>
      </button>

      {/* expanded */}
      {expanded && (
        <div className="px-4 pb-4 flex flex-col gap-4 border-t border-surface-3 pt-3">
          {fetching && (
            <div className="flex gap-0.5 py-1">
              {[0, 1, 2].map((i) => (
                <span key={i} className="w-1 h-1 rounded-full bg-zinc-600 animate-bounce" style={{ animationDelay: `${i * 150}ms` }} />
              ))}
            </div>
          )}

          {detail && (
            <>
              {/* trigger stats */}
              {(detail.z_score != null || detail.velocity != null) && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">anomaly trigger</span>
                  <div className="flex gap-2">
                    {detail.z_score != null && (
                      <StatPill label="z-score" value={detail.z_score.toFixed(2)} color="#f97316" />
                    )}
                    {detail.velocity != null && (
                      <StatPill label="velocity" value={detail.velocity.toFixed(1)} color="#a78bfa" />
                    )}
                    <StatPill
                      label="confidence"
                      value={`${Math.round(detail.confidence * 100)}%`}
                      color={classColor}
                    />
                  </div>
                </div>
              )}

              {/* signals */}
              {detail.signals.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">signals</span>
                  {detail.signals.map((s, i) => (
                    <div key={i} className="flex gap-2 text-xs text-zinc-400">
                      <span className="text-zinc-600 shrink-0">—</span>
                      {s}
                    </div>
                  ))}
                </div>
              )}

              {/* reasoning steps */}
              {detail.reasoning_steps_detail.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">
                    agent reasoning ({detail.reasoning_steps_detail.length} steps)
                  </span>
                  {detail.reasoning_steps_detail.map((step, i) => (
                    <div key={i} className="flex gap-2 text-xs text-zinc-500">
                      <span className="text-zinc-700 shrink-0 font-mono">{i + 1}.</span>
                      {step}
                    </div>
                  ))}
                </div>
              )}

              {/* similar past cases */}
              {detail.similar_past_cases.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">similar past cases</span>
                  {detail.similar_past_cases.map((c, i) => (
                    <div key={i} className="flex items-center gap-2 text-[11px]">
                      <span className="text-zinc-500 truncate flex-1">{c.trend}</span>
                      <span className="text-zinc-700 shrink-0">{Math.round(c.similarity * 100)}% match</span>
                      <span
                        className="shrink-0 text-[10px] px-1.5 py-0.5 rounded"
                        style={{ color: CLASSIFICATION_COLOR[c.outcome] ?? "#eab308", backgroundColor: `${CLASSIFICATION_COLOR[c.outcome] ?? "#eab308"}18` }}
                      >
                        {c.outcome.replace(/_/g, " ")}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* ragas evaluation */}
              {Object.keys(detail.ragas_scores).length > 0 && (
                <div className="flex flex-col gap-2">
                  <span className="text-[10px] uppercase tracking-widest text-zinc-600">ragas evaluation</span>
                  <div className="flex flex-col gap-2">
                    {detail.ragas_scores.retrieval_relevance !== undefined && (
                      <RagasBar label="retrieval relevance" value={detail.ragas_scores.retrieval_relevance} />
                    )}
                    {detail.ragas_scores.reasoning_consistency !== undefined && (
                      <RagasBar label="reasoning consistency" value={detail.ragas_scores.reasoning_consistency} />
                    )}
                    {detail.ragas_scores.confidence_calibration !== undefined && (
                      <RagasBar label="confidence calibration" value={detail.ragas_scores.confidence_calibration} />
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
