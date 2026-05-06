"use client";

import { useState } from "react";
import { useCaseFiles, CaseFileSummary, CaseFileDetail, fetchCaseDetail } from "@/hooks/useCaseFiles";
import { CLASSIFICATION_COLOR, PLATFORM_COLOR } from "@/lib/colors";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function relativeTime(iso: string): string {
  const delta = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(delta / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function caseId(id: number) {
  return `CASE-${String(id).padStart(4, "0")}`;
}

function statusBadge(classification: string, needsReview: boolean) {
  if (needsReview)
    return { label: "REVIEW",  bg: "bg-error-container/20",     border: "border-error-container/40",     text: "text-error"            };
  if (classification === "organic")
    return { label: "RESOLVED", bg: "bg-secondary/10",           border: "border-secondary/20",           text: "text-secondary"        };
  if (classification === "coordinated_amplification")
    return { label: "OPEN",     bg: "bg-primary/10",             border: "border-primary/20",             text: "text-primary"          };
  return   { label: "MONITORING",bg: "bg-tertiary-container/10", border: "border-tertiary-container/20",  text: "text-tertiary-container"};
}

// ─── Case list item ───────────────────────────────────────────────────────────

function CaseListItem({
  summary,
  active,
  onClick,
}: {
  summary: CaseFileSummary;
  active: boolean;
  onClick: () => void;
}) {
  const { label, bg, border, text } = statusBadge(summary.classification, summary.needs_review);
  const displayName = summary.label && summary.label !== summary.trend ? summary.label : summary.trend;

  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-md border-b-[0.5px] border-outline-variant hover:bg-surface-container transition-colors ${
        active ? "bg-primary-container border-l-4 border-l-primary" : ""
      }`}
    >
      <div className="flex justify-between items-start mb-xs">
        <span className={`font-mono-data text-sm tracking-tighter ${active ? "text-primary" : "text-outline"}`}>
          {caseId(summary.id)}
        </span>
        <span className={`px-xs py-[2px] ${bg} border ${border} ${text} font-mono-label text-[10px] uppercase`}>
          {label}
        </span>
      </div>
      <h3 className="font-h3 text-sm font-semibold mb-xs text-on-surface truncate">{displayName}</h3>
      <p className="font-body-sm text-xs text-on-surface-variant line-clamp-2">
        {summary.classification.replace(/_/g, " ")} — confidence {Math.round(summary.confidence * 100)}%
      </p>
      <div className="mt-sm flex items-center gap-sm text-[10px] font-mono-label text-outline">
        <span>{relativeTime(summary.created_at)}</span>
        <span>•</span>
        <span>{summary.platform_origin}</span>
      </div>
    </button>
  );
}

// ─── Signal row ───────────────────────────────────────────────────────────────

function SignalRow({ signal, ts }: { signal: string; ts: string }) {
  return (
    <div className="grid grid-cols-12 items-center py-md border-b-[0.5px] border-outline-variant/50 hover:bg-surface-container-low transition-colors group">
      <div className="col-span-1">
        <span className="px-1.5 py-0.5 bg-secondary-container/20 text-secondary border border-secondary-container/40 rounded-sm font-mono-label text-[9px] uppercase">
          SIG
        </span>
      </div>
      <div className="col-span-2 font-mono-data text-xs text-outline group-hover:text-on-surface-variant">{ts}</div>
      <div className="col-span-7 font-body-sm text-sm text-on-surface-variant">{signal}</div>
      <div className="col-span-2 text-right font-mono-data text-xs text-primary">+Δv</div>
    </div>
  );
}

// ─── Case detail panel ────────────────────────────────────────────────────────

function CaseDetail({ summary }: { summary: CaseFileSummary }) {
  const [detail, setDetail] = useState<CaseFileDetail | null>(null);
  const [loading, setLoading] = useState(false);

  // fetch detail on mount
  useState(() => {
    setLoading(true);
    fetchCaseDetail(summary.id).then((d) => {
      setDetail(d);
      setLoading(false);
    });
  });

  const classColor = CLASSIFICATION_COLOR[summary.classification] ?? "#ffd7b5";
  const displayName = summary.label && summary.label !== summary.trend ? summary.label : summary.trend;
  const confPct = Math.round(summary.confidence * 100);

  return (
    <section className="flex-1 flex flex-col bg-surface-container-lowest overflow-y-auto">
      {/* Detail header */}
      <div className="p-xl pb-lg flex flex-col gap-sm border-b-[0.5px] border-outline-variant">
        <div className="flex items-center gap-md">
          <span className="font-mono-data text-h2 text-primary tracking-tighter">{caseId(summary.id)}</span>
          <div className="h-4 w-[0.5px] bg-outline-variant" />
          <span className="font-mono-label uppercase text-on-surface-variant">
            Confidence: {confPct}%
          </span>
        </div>
        <h1 className="font-h1 text-[28px] text-on-surface leading-tight">{displayName}</h1>
      </div>

      {/* AI summary */}
      <div className="mx-xl mt-xl mb-lg bg-surface-container-lowest border-l-4 border-primary p-lg">
        <div className="flex items-center gap-sm mb-md text-primary">
          <span className="material-symbols-outlined text-[20px]">psychology</span>
          <span className="font-mono-label uppercase tracking-widest text-[11px]">AI Intelligence Synthesis</span>
        </div>
        <p className="font-body-lg text-on-surface-variant leading-relaxed italic opacity-90">
          {loading
            ? "Synthesizing investigation data…"
            : detail?.signals?.length
            ? detail.signals.join(". ")
            : `${summary.classification.replace(/_/g, " ")} pattern detected with ${confPct}% confidence. Review signals below.`}
        </p>
      </div>

      {/* Evidence / signals */}
      <div className="px-xl flex-1">
        <div className="flex items-center justify-between mb-md border-b border-outline-variant pb-sm">
          <h3 className="font-mono-label uppercase tracking-widest text-on-surface-variant">
            Evidence Logs &amp; Signal Rows
          </h3>
          <span className="font-mono-label text-primary">
            {detail?.signals?.length ?? 0} SIGNALS
          </span>
        </div>

        {loading ? (
          <div className="space-y-2 py-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-10 bg-surface-container rounded animate-pulse" />
            ))}
          </div>
        ) : detail?.signals?.length ? (
          detail.signals.map((s, i) => (
            <SignalRow key={i} signal={s} ts={new Date(summary.created_at).toISOString().substring(11, 23)} />
          ))
        ) : (
          <div className="py-8 text-center font-mono-label text-[11px] text-on-surface-variant uppercase">
            no signals recorded
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div className="p-xl border-t border-outline-variant mt-auto">
        <div className="flex items-center justify-between">
          {/* Sparkline replay */}
          <div className="flex flex-col gap-1">
            <span className="font-mono-label text-on-surface-variant text-[10px] uppercase mb-1">Propagation Replay</span>
            <div className="w-64 h-12 relative overflow-hidden bg-surface-container border border-outline-variant rounded-sm">
              <div className="absolute inset-0 flex items-end px-1 gap-[2px]">
                {[30, 45, 40, 60, 80, 75, 95, 100, 60, 30, 20, 15].map((h, i) => (
                  <div key={i} className="w-1 rounded-sm" style={{ height: `${h}%`, backgroundColor: `rgba(110,246,199,${h / 100})` }} />
                ))}
              </div>
              <div className="absolute top-0 bottom-0 left-[70%] w-0.5 bg-white/30 z-10" />
            </div>
          </div>

          <div className="flex gap-md">
            <button className="px-lg py-sm border border-tertiary-fixed-dim text-tertiary-fixed-dim font-mono-label uppercase tracking-widest hover:bg-tertiary-container/10 transition-colors text-[12px]">
              Flag Anomaly
            </button>
            <button className="px-lg py-sm border border-primary text-primary font-mono-label uppercase tracking-widest hover:bg-primary/10 transition-colors text-[12px]">
              Mark Resolved
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const FILTERS = [
  { value: undefined,                  label: "all"         },
  { value: "organic",                  label: "organic"     },
  { value: "coordinated_amplification",label: "coordinated" },
  { value: "uncertain",                label: "uncertain"   },
];

export default function CasesPage() {
  const [needsReview, setNeedsReview] = useState<boolean | undefined>(undefined);
  const [classification, setClassification] = useState<string | undefined>(undefined);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const { cases, loading } = useCaseFiles({ needsReview, classification });
  const selectedCase = cases.find((c) => c.id === selectedId) ?? null;

  return (
    <div className="w-full h-full flex bg-background">
      {/* Left: case list */}
      <section className="w-[380px] border-r-[0.5px] border-outline-variant flex flex-col shrink-0">
        {/* Filter header */}
        <div className="p-md border-b-[0.5px] border-outline-variant flex justify-between items-center bg-surface-container-lowest">
          <h2 className="font-mono-label uppercase text-on-surface-variant tracking-widest">
            Active Investigations
          </h2>
          <span className="material-symbols-outlined text-[20px] text-primary">filter_list</span>
        </div>

        {/* Filter chips */}
        <div className="flex flex-wrap gap-xs p-md border-b-[0.5px] border-outline-variant">
          <button
            onClick={() => setNeedsReview(undefined)}
            className={`font-mono-label text-[10px] px-sm py-xs rounded border uppercase transition-colors ${
              needsReview === undefined ? "border-primary/40 bg-primary/10 text-primary" : "border-outline-variant text-on-surface-variant hover:border-outline"
            }`}
          >
            all
          </button>
          <button
            onClick={() => setNeedsReview(true)}
            className={`font-mono-label text-[10px] px-sm py-xs rounded border uppercase transition-colors ${
              needsReview === true ? "border-error/40 bg-error/10 text-error" : "border-outline-variant text-on-surface-variant hover:border-outline"
            }`}
          >
            needs review
          </button>
          {FILTERS.filter((f) => f.value !== undefined).map(({ value, label }) => (
            <button
              key={label}
              onClick={() => setClassification(value)}
              className={`font-mono-label text-[10px] px-sm py-xs rounded border uppercase transition-colors ${
                classification === value ? "border-primary/40 bg-primary/10 text-primary" : "border-outline-variant text-on-surface-variant hover:border-outline"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex flex-col">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="p-md border-b-[0.5px] border-outline-variant animate-pulse">
                  <div className="h-3 bg-surface-container-highest rounded w-1/3 mb-sm" />
                  <div className="h-4 bg-surface-container-highest rounded w-3/4 mb-xs" />
                  <div className="h-3 bg-surface-container-highest rounded w-full" />
                </div>
              ))}
            </div>
          ) : cases.length === 0 ? (
            <div className="flex items-center justify-center h-32 font-mono-label text-[11px] text-on-surface-variant uppercase">
              no case files
            </div>
          ) : (
            cases.map((c) => (
              <CaseListItem
                key={c.id}
                summary={c}
                active={c.id === selectedId}
                onClick={() => setSelectedId(c.id)}
              />
            ))
          )}
        </div>
      </section>

      {/* Right: detail */}
      {selectedCase ? (
        <CaseDetail key={selectedCase.id} summary={selectedCase} />
      ) : (
        <div className="flex-1 flex items-center justify-center bg-surface-container-lowest">
          <div className="text-center">
            <span className="material-symbols-outlined text-[48px] text-outline-variant block mb-md">manage_search</span>
            <p className="font-mono-label text-[11px] text-on-surface-variant uppercase">
              Select a case to view details
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
