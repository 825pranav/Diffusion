"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useTrending } from "@/hooks/useTrending";

// ─── Types ────────────────────────────────────────────────────────────────────

type Platform = "hn" | "gh" | "rd";

interface PropagationStep {
  platform: Platform;
  minutesAgo: number;
}

interface TrendingTopic {
  id: string;
  name: string;
  url?: string;
  platforms: Platform[];
  velocity: number; // mentions/min
  velocityDelta: number; // % change vs 5 min ago
  trajectory: "rising" | "plateau" | "cooling";
  propagation: PropagationStep[];
  sparkline: number[]; // 12 values, 0–1 normalised
  breaking: boolean;
}

type SortKey = "velocity" | "spread" | "newest";
type TimeWindow = "30m" | "2h" | "today";
type PlatformFilter = "all" | Platform;

// ─── Constants ────────────────────────────────────────────────────────────────

const PLATFORM_META: Record<Platform, { label: string; color: string; bg: string; dot: string }> = {
  hn: { label: "HN",     color: "text-red-700",    bg: "bg-red-50",     dot: "bg-red-400" },
  gh: { label: "GitHub", color: "text-violet-700",  bg: "bg-violet-50",  dot: "bg-violet-500" },
  rd: { label: "Reddit", color: "text-emerald-700", bg: "bg-emerald-50", dot: "bg-emerald-500" },
};


// ─── Sub-components ───────────────────────────────────────────────────────────

function Sparkline({ values, lit }: { values: number[]; lit: boolean }) {
  const max = Math.max(...values);
  return (
    <div className="flex items-end gap-[3px] h-7 mt-2.5">
      {values.map((v, i) => {
        const h = Math.round((v / max) * 100);
        const active = lit && i >= values.length - 5;
        return (
          <div
            key={i}
            className={`w-1.5 rounded-sm transition-all duration-300 ${
              active ? "bg-violet-400" : "bg-neutral-200 dark:bg-neutral-700"
            }`}
            style={{ height: `${h}%` }}
          />
        );
      })}
    </div>
  );
}

function PlatformTag({ platform }: { platform: Platform }) {
  const m = PLATFORM_META[platform];
  return (
    <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${m.bg} ${m.color}`}>
      {m.label}
    </span>
  );
}

function PropagationChain({ steps }: { steps: PropagationStep[] }) {
  return (
    <div className="flex items-center gap-1 mt-2.5 flex-wrap">
      {steps.map((step, i) => {
        const m = PLATFORM_META[step.platform];
        return (
          <div key={step.platform} className="flex items-center gap-1">
            <span className={`text-[11px] font-semibold ${m.color}`}>{m.label}</span>
            <span className="text-[11px] text-neutral-400">{step.minutesAgo}m ago</span>
            {i < steps.length - 1 && (
              <span className="text-[11px] text-neutral-300 dark:text-neutral-600 mx-0.5">→</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function TrajectoryBadge({ t, delta }: { t: TrendingTopic["trajectory"]; delta: number }) {
  if (t === "rising")
    return <span className="text-[13px] font-medium text-emerald-500">↑ {delta}%</span>;
  if (t === "cooling")
    return <span className="text-[13px] font-medium text-neutral-400">↓ {Math.abs(delta)}%</span>;
  return <span className="text-[13px] font-medium text-neutral-400">→ plateau</span>;
}

function TrendCard({ topic, rank }: { topic: TrendingTopic; rank: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: rank * 0.04 }}
      className={`
        border rounded-xl px-4 py-3.5 mb-2.5 bg-white dark:bg-neutral-900
        hover:border-neutral-300 dark:hover:border-neutral-600 transition-colors cursor-pointer
        ${topic.breaking
          ? "border-l-2 border-l-violet-400 border-t border-r border-b border-neutral-200 dark:border-neutral-700"
          : "border border-neutral-200 dark:border-neutral-800"
        }
      `}
    >
      <div className="flex items-start gap-3">
        {/* Rank */}
        <span className="text-lg font-medium text-neutral-300 dark:text-neutral-600 min-w-[20px] mt-0.5">
          {rank}
        </span>

        {/* Body */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start gap-2 flex-wrap">
            {topic.url ? (
              <a
                href={topic.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[14px] font-medium text-neutral-900 dark:text-neutral-100 leading-snug hover:text-violet-600 dark:hover:text-violet-400 hover:underline transition-colors"
                onClick={(e) => e.stopPropagation()}
              >
                {topic.name}
              </a>
            ) : (
              <span className="text-[14px] font-medium text-neutral-900 dark:text-neutral-100 leading-snug">
                {topic.name}
              </span>
            )}
            {topic.breaking && (
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-violet-50 text-violet-600 dark:bg-violet-950 dark:text-violet-300 shrink-0">
                breaking
              </span>
            )}
          </div>

          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            {topic.platforms.map((p) => (
              <PlatformTag key={p} platform={p} />
            ))}
            <span className="text-[12px] text-neutral-500">
              <span className="font-medium text-neutral-700 dark:text-neutral-300">
                {topic.velocity.toFixed(1)}
              </span>{" "}
              mentions/min
            </span>
          </div>

          <PropagationChain steps={topic.propagation} />
          <Sparkline values={topic.sparkline} lit={topic.trajectory === "rising"} />
        </div>

        {/* Right column */}
        <div className="flex flex-col items-end gap-1 shrink-0">
          <span className="text-xl font-medium text-neutral-900 dark:text-neutral-100 leading-none">
            {topic.velocity.toFixed(1)}
          </span>
          <span className="text-[11px] text-neutral-400">mentions/min</span>
          <TrajectoryBadge t={topic.trajectory} delta={topic.velocityDelta} />
        </div>
      </div>
    </motion.div>
  );
}

function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub: string;
}) {
  return (
    <div className="bg-neutral-50 dark:bg-neutral-800/60 rounded-lg px-3.5 py-3 flex-1 min-w-0">
      <p className="text-[11px] text-neutral-400 uppercase tracking-wide mb-1">{label}</p>
      <p className="text-base font-medium text-neutral-900 dark:text-neutral-100">{value}</p>
      <p className="text-[11px] text-neutral-400 mt-0.5">{sub}</p>
    </div>
  );
}

// ─── Main dashboard ───────────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div className="border border-neutral-200 dark:border-neutral-800 rounded-xl px-4 py-3.5 mb-2.5 bg-white dark:bg-neutral-900 animate-pulse">
      <div className="flex items-start gap-3">
        <div className="w-5 h-4 bg-neutral-200 dark:bg-neutral-700 rounded mt-0.5 shrink-0" />
        <div className="flex-1 space-y-2">
          <div className="h-3.5 bg-neutral-200 dark:bg-neutral-700 rounded w-3/4" />
          <div className="flex gap-2">
            <div className="h-3 bg-neutral-200 dark:bg-neutral-700 rounded-full w-10" />
            <div className="h-3 bg-neutral-200 dark:bg-neutral-700 rounded-full w-10" />
            <div className="h-3 bg-neutral-200 dark:bg-neutral-700 rounded w-20" />
          </div>
          <div className="h-2 bg-neutral-200 dark:bg-neutral-700 rounded w-1/2" />
          <div className="flex items-end gap-[3px] h-7 mt-2.5">
            {Array.from({ length: 12 }).map((_, i) => (
              <div
                key={i}
                className="w-1.5 rounded-sm bg-neutral-200 dark:bg-neutral-700"
                style={{ height: `${20 + Math.random() * 60}%` }}
              />
            ))}
          </div>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <div className="h-5 w-10 bg-neutral-200 dark:bg-neutral-700 rounded" />
          <div className="h-2.5 w-16 bg-neutral-200 dark:bg-neutral-700 rounded" />
        </div>
      </div>
    </div>
  );
}

export default function TrendingDashboard() {
  const [sortKey, setSortKey] = useState<SortKey>("velocity");
  const [timeWindow, setTimeWindow] = useState<TimeWindow>("30m");
  const [platformFilter, setPlatformFilter] = useState<PlatformFilter>("all");
  const [activeTab, setActiveTab] = useState<"trending" | "propagation" | "investigations">(
    "trending"
  );

  const { topics, stats, wsLive, loading } = useTrending(timeWindow, platformFilter, sortKey);

  // Client-side sort is a secondary sort on top of server sort (server already sorts by the key,
  // but we re-sort here so filter changes are instant without a round-trip)
  const sorted = [...topics].sort((a, b) => {
    if (sortKey === "spread") return b.platforms.length - a.platforms.length;
    if (sortKey === "newest") return 0; // server order is newest
    return b.velocity - a.velocity;
  });

  return (
    <div className="flex flex-col h-screen bg-white dark:bg-neutral-950 font-sans">
      {/* ── Top bar ── */}
      <header className="flex items-center gap-4 px-5 h-12 border-b border-neutral-200 dark:border-neutral-800 shrink-0">
        <span className="text-[15px] font-medium tracking-tight">
          diff<span className="text-violet-500">usion</span>
        </span>

        <nav className="flex gap-1 ml-4">
          {(["trending", "propagation", "investigations"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`text-[13px] px-3 py-1 rounded-md transition-colors capitalize ${
                activeTab === tab
                  ? "bg-neutral-100 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 font-medium"
                  : "text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
              }`}
            >
              {tab}
            </button>
          ))}
        </nav>

        {/* WS badge */}
        <div className="ml-auto flex items-center gap-1.5 text-[12px] text-neutral-500">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              wsLive ? "bg-emerald-400" : "bg-amber-400 animate-pulse"
            }`}
          />
          {wsLive ? "live" : "reconnecting…"}
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* ── Sidebar ── */}
        <aside className="w-52 shrink-0 border-r border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900/50 p-3 flex flex-col gap-1 overflow-y-auto">
          <p className="text-[11px] text-neutral-400 uppercase tracking-wide px-2 pt-1 pb-0.5">
            Platforms
          </p>
          {(
            [
              { key: "all", label: "All sources", dot: "bg-neutral-400" },
              { key: "hn",  label: "Hacker News", dot: PLATFORM_META.hn.dot },
              { key: "gh",  label: "GitHub",      dot: PLATFORM_META.gh.dot },
              { key: "rd",  label: "Reddit",       dot: PLATFORM_META.rd.dot },
            ] as const
          ).map(({ key, label, dot }) => (
            <button
              key={key}
              onClick={() => setPlatformFilter(key)}
              className={`flex items-center gap-2 text-[13px] px-2.5 py-1.5 rounded-lg text-left transition-colors w-full ${
                platformFilter === key
                  ? "bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 font-medium shadow-sm"
                  : "text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
              }`}
            >
              <span className={`w-2 h-2 rounded-full shrink-0 ${dot}`} />
              {label}
            </button>
          ))}

          <p className="text-[11px] text-neutral-400 uppercase tracking-wide px-2 pt-3 pb-0.5">
            Timeframe
          </p>
          {(
            [
              { key: "30m",   label: "Last 30 min" },
              { key: "2h",    label: "Last 2 hours" },
              { key: "today", label: "Today" },
            ] as const
          ).map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setTimeWindow(key)}
              className={`text-[13px] px-2.5 py-1.5 rounded-lg text-left transition-colors w-full ${
                timeWindow === key
                  ? "bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 font-medium shadow-sm"
                  : "text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
              }`}
            >
              {label}
            </button>
          ))}

          <p className="text-[11px] text-neutral-400 uppercase tracking-wide px-2 pt-3 pb-0.5">
            Topics
          </p>
          {["AI / ML", "Security", "Infrastructure"].map((t) => (
            <button
              key={t}
              className="text-[13px] px-2.5 py-1.5 rounded-lg text-left text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors w-full"
            >
              {t}
            </button>
          ))}
        </aside>

        {/* ── Main ── */}
        <main className="flex-1 overflow-y-auto px-5 py-5">
          {/* Stat cards */}
          <div className="flex gap-2.5 mb-5">
            <StatCard
              label="trending topics"
              value={stats.trendingTopics}
              sub={`+${stats.trendingDelta} vs prev window`}
            />
            <StatCard
              label="fastest velocity"
              value={`${stats.fastestVelocity}/min`}
              sub={stats.fastestName}
            />
            <StatCard
              label="cross-platform"
              value={`${stats.crossPlatform} topics`}
              sub="on 2+ platforms"
            />
            <StatCard
              label="breaking now"
              value={stats.breakingNow}
              sub="spiked in <15 min"
            />
          </div>

          {/* Section header */}
          <div className="flex items-center justify-between mb-3.5">
            <span className="text-[14px] font-medium text-neutral-900 dark:text-neutral-100">
              Trending now
            </span>
            <div className="flex gap-1">
              {(["velocity", "spread", "newest"] as const).map((key) => (
                <button
                  key={key}
                  onClick={() => setSortKey(key)}
                  className={`text-[12px] px-2.5 py-1 rounded-md border transition-colors capitalize ${
                    sortKey === key
                      ? "border-neutral-300 dark:border-neutral-600 bg-neutral-100 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100"
                      : "border-neutral-200 dark:border-neutral-700 text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
                  }`}
                >
                  {key}
                </button>
              ))}
            </div>
          </div>

          {/* Cards */}
          {loading ? (
            Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} />)
          ) : sorted.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-neutral-400">
              <span className="text-sm">No trending topics in this window yet.</span>
              <span className="text-xs mt-1 text-neutral-300 dark:text-neutral-600">
                Data arrives as producers ingest from HN, GitHub, and Reddit.
              </span>
            </div>
          ) : (
            <AnimatePresence mode="popLayout">
              {sorted.map((topic, i) => (
                <TrendCard key={topic.id} topic={topic} rank={i + 1} />
              ))}
            </AnimatePresence>
          )}
        </main>
      </div>
    </div>
  );
}
