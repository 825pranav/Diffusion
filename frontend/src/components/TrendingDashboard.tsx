"use client";

import { useTrending, TrendingTopic } from "@/hooks/useTrending";

// ─── Platform metadata ────────────────────────────────────────────────────────

type Platform = "hn" | "gh" | "rd";

const PLATFORM_META: Record<Platform, { label: string; bg: string; text: string }> = {
  hn: { label: "HN",     bg: "bg-tertiary-container/20", text: "text-tertiary-container" },
  gh: { label: "GitHub", bg: "bg-secondary/10",          text: "text-secondary"          },
  rd: { label: "Reddit", bg: "bg-primary/10",            text: "text-primary"            },
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function PlatformChip({ platform }: { platform: Platform }) {
  const m = PLATFORM_META[platform] ?? PLATFORM_META.hn;
  return (
    <span className={`font-mono-label text-mono-label px-2 py-1 rounded ${m.bg} ${m.text} border border-current/20`}>
      {m.label}
    </span>
  );
}

function MiniSparkline({ values }: { values: number[] }) {
  if (!values.length) return null;
  const max = Math.max(...values, 1);
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * 100},${20 - (v / max) * 18}`)
    .join(" ");
  return (
    <svg className="w-full h-8" viewBox="0 0 100 20" preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke="#6ef6c7" strokeWidth="1.5" />
    </svg>
  );
}

function HeroSparkline({ values }: { values: number[] }) {
  if (!values.length) return null;
  const max = Math.max(...values, 1);
  const w = 400;
  const h = 100;
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * w} ${h - (v / max) * (h - 8)}`)
    .join(" L ");
  return (
    <svg className="absolute inset-0 w-full h-full p-gutter" preserveAspectRatio="none" viewBox={`0 0 ${w} ${h}`}>
      <path d={`M ${pts}`} fill="none" stroke="#4dd9ac" strokeWidth="2" /* primary */ />
      {values.length > 0 && (
        <circle
          cx={w}
          cy={h - (values[values.length - 1] / max) * (h - 8)}
          r="4"
          className="fill-primary animate-pulse"
        />
      )}
    </svg>
  );
}

function PropagationTimeline({ topic }: { topic: TrendingTopic }) {
  const steps = topic.propagation ?? [];
  if (!steps.length) return null;
  return (
    <div className="mt-lg flex items-center justify-between px-xs">
      {steps.map((step, i) => {
        const m = PLATFORM_META[step.platform as Platform] ?? PLATFORM_META.hn;
        const isLast = i === steps.length - 1;
        return (
          <div key={i} className="flex items-center flex-1">
            <div className="flex flex-col items-center gap-1">
              <div className={`rounded-full ${isLast ? "w-3 h-3 ring-4 ring-primary/20 bg-primary" : "w-1.5 h-1.5 bg-outline-variant"}`} />
              <span className={`font-mono-label text-mono-label ${isLast ? "text-primary" : "text-on-surface-variant"}`}>
                {step.minutesAgo === 0 ? "NOW " : `${step.minutesAgo}m `}{m.label}
              </span>
            </div>
            {!isLast && <div className="flex-1 h-[1px] bg-outline-variant mx-2" />}
          </div>
        );
      })}
    </div>
  );
}

function HeroCard({ topic }: { topic: TrendingTopic }) {
  return (
    <div className="col-span-12 lg:col-span-8 bg-surface-container rounded-xl border-[0.5px] border-outline-variant p-lg relative overflow-hidden">
      <div className="flex justify-between items-start mb-md">
        <div className="flex-1 min-w-0 pr-4">
          {topic.breaking && (
            <span className="inline-block bg-tertiary-container text-on-tertiary-container font-mono-label text-mono-label px-sm py-1 rounded-sm uppercase tracking-widest mb-sm">
              BREAKING
            </span>
          )}
          <h2 className="font-h2 text-h2 text-on-surface mb-xs truncate">{topic.name}</h2>
          <p className="font-body-sm text-body-sm text-on-surface-variant max-w-xl line-clamp-2">
            Propagation detected across platforms. Velocity index rising.
          </p>
        </div>
        <div className="flex items-center gap-xs shrink-0">
          {(topic.platforms as Platform[]).map((p, i) => (
            <div key={p} className="flex items-center gap-xs">
              <PlatformChip platform={p} />
              {i < topic.platforms.length - 1 && (
                <span className="material-symbols-outlined text-outline-variant text-[12px]">arrow_forward</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Sparkline area */}
      <div className="h-[200px] w-full bg-surface-container-lowest border-[0.5px] border-outline-variant rounded relative overflow-hidden mt-md">
        <HeroSparkline values={topic.sparkline} />
        <div className="absolute bottom-4 left-4 right-4 flex justify-between pointer-events-none">
          <div className="flex flex-col">
            <span className="font-mono-label text-mono-label text-on-surface-variant uppercase">Velocity Index</span>
            <span className="font-h1 text-h1 text-primary">{(topic.velocity * 15).toFixed(0)}</span>
          </div>
          <div className="flex flex-col items-end">
            <span className="font-mono-label text-mono-label text-on-surface-variant uppercase">Mentions / Min</span>
            <span className="font-h1 text-h1 text-tertiary">{topic.velocity.toFixed(0)}</span>
          </div>
        </div>
      </div>

      <PropagationTimeline topic={topic} />
    </div>
  );
}

function SignalHealthCard({ wsLive }: { wsLive: boolean }) {
  return (
    <div className="bg-surface-container rounded-xl border-l-4 border-primary border-[0.5px] border-outline-variant p-lg">
      <h3 className="font-h3 text-h3 mb-lg flex items-center gap-sm">
        <span className="material-symbols-outlined text-primary">monitor_heart</span>
        Signal Health
      </h3>
      <div className="space-y-xl">
        {[
          { label: "HN ingestion",  value: 98, color: "bg-primary" },
          { label: "GH events",     value: 94, color: "bg-primary" },
          { label: "WS latency",    value: 65, color: "bg-tertiary" },
        ].map(({ label, value, color }) => (
          <div key={label}>
            <div className="flex justify-between mb-2">
              <span className="font-mono-label text-mono-label uppercase text-on-surface-variant">{label}</span>
              <span className={`font-mono-data text-mono-data ${color === "bg-primary" ? "text-primary" : "text-tertiary"}`}>
                {value}%
              </span>
            </div>
            <div className="h-1 w-full bg-surface-container-lowest rounded-full overflow-hidden">
              <div className={`h-full ${color}`} style={{ width: `${value}%` }} />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-xl pt-lg border-t-[0.5px] border-outline-variant">
        <button className="w-full py-sm border border-outline-variant rounded font-mono-label text-mono-label uppercase hover:bg-surface-container-highest transition-colors flex items-center justify-center gap-sm text-on-surface-variant">
          <span className="material-symbols-outlined text-[16px]">terminal</span>
          Show system logs
        </button>
      </div>
    </div>
  );
}

function LiveStreamCard({ wsLive }: { wsLive: boolean }) {
  return (
    <div className="bg-surface-container-lowest rounded-xl border-[0.5px] border-outline-variant p-md flex-1">
      <div className="flex items-center gap-2 mb-md border-b-[0.5px] border-outline-variant pb-2">
        <div className={`w-2 h-2 rounded-full ${wsLive ? "bg-primary animate-pulse" : "bg-error"}`} />
        <span className="font-mono-label text-mono-label text-on-surface-variant uppercase">Live Stream</span>
      </div>
      <div className="font-mono-data text-[11px] space-y-1 h-[120px] overflow-y-auto">
        <p className="text-secondary">[INFO] Connection established to wss://events.main</p>
        <p className="text-on-surface-variant">[DATA] Payload received: vector_id: 9402</p>
        <p className="text-tertiary">[WARN] Latency spike detected in EU-West-1</p>
        <p className="text-primary">[SYNC] HN-Scraper aligned with latest thread</p>
        <p className="text-on-surface-variant">[DATA] Topic cluster: AI, Security</p>
        {!wsLive && <p className="text-error">[ERR] WebSocket: reconnecting...</p>}
      </div>
    </div>
  );
}

function MiniTrendCard({ topic, rank }: { topic: TrendingTopic; rank: number }) {
  const platform = (topic.platforms[0] ?? "hn") as Platform;
  const m = PLATFORM_META[platform] ?? PLATFORM_META.hn;
  return (
    <div className="col-span-12 md:col-span-4 bg-surface-container rounded-xl border-[0.5px] border-outline-variant p-md hover:bg-surface-container-high transition-all cursor-pointer">
      <div className="flex justify-between items-center mb-sm">
        <div className="flex items-center gap-sm">
          <span className="font-mono-label text-mono-label text-on-surface-variant">
            {String(rank).padStart(2, "0")}
          </span>
          {topic.breaking && (
            <span className="font-mono-label text-mono-label px-2 py-0.5 rounded bg-error-container text-on-error-container uppercase text-[10px]">
              Break
            </span>
          )}
        </div>
        <span className={`font-mono-label text-mono-label px-2 py-0.5 rounded ${m.bg} ${m.text} border border-current/20 uppercase text-[10px]`}>
          {m.label}
        </span>
      </div>
      <h4 className="font-h3 text-h3 text-on-surface mb-md truncate">{topic.name}</h4>
      <MiniSparkline values={topic.sparkline} />
    </div>
  );
}

function SkeletonHero() {
  return (
    <div className="col-span-12 lg:col-span-8 bg-surface-container rounded-xl border-[0.5px] border-outline-variant p-lg animate-pulse">
      <div className="h-6 bg-surface-container-highest rounded w-1/3 mb-sm" />
      <div className="h-8 bg-surface-container-highest rounded w-2/3 mb-xs" />
      <div className="h-4 bg-surface-container-highest rounded w-full mb-md" />
      <div className="h-[200px] bg-surface-container-lowest rounded border-[0.5px] border-outline-variant" />
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function TrendingDashboard() {
  const { topics, wsLive, loading } = useTrending("30m", "all", "velocity");

  const hero = topics[0] ?? null;
  const miniCards = topics.slice(1, 4);

  return (
    <div className="h-full overflow-y-auto p-margin bg-surface-dim">
      {/* Title bar */}
      <div className="flex items-center gap-sm mb-lg">
        <div className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
        </div>
        <h1 className="font-h1 text-h1">Global velocity</h1>
      </div>

      {/* Dashboard grid */}
      <div className="grid grid-cols-12 gap-gutter">
        {/* Hero card */}
        {loading || !hero ? (
          <SkeletonHero />
        ) : (
          <HeroCard topic={hero} />
        )}

        {/* Right panel */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-gutter">
          <SignalHealthCard wsLive={wsLive} />
          <LiveStreamCard wsLive={wsLive} />
        </div>

        {/* Mini trend cards */}
        {loading
          ? [0, 1, 2].map((i) => (
              <div
                key={i}
                className="col-span-12 md:col-span-4 bg-surface-container rounded-xl border-[0.5px] border-outline-variant p-md animate-pulse h-28"
              />
            ))
          : miniCards.map((t, i) => (
              <MiniTrendCard key={t.id} topic={t} rank={i + 2} />
            ))}
      </div>
    </div>
  );
}
