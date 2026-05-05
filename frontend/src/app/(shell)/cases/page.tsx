"use client";

import { useState } from "react";
import GridBackground from "@/components/GridBackground";
import CaseCard from "@/components/CaseCard";
import { useCaseFiles, CaseFilters } from "@/hooks/useCaseFiles";

const CLASSIFICATIONS = [
  { value: undefined, label: "all" },
  { value: "organic", label: "organic" },
  { value: "coordinated_amplification", label: "coordinated" },
  { value: "uncertain", label: "uncertain" },
];

export default function CasesPage() {
  const [needsReview, setNeedsReview] = useState<boolean | undefined>(undefined);
  const [classification, setClassification] = useState<string | undefined>(undefined);

  const filters: CaseFilters = { needsReview, classification };
  const { cases, loading } = useCaseFiles(filters);

  return (
    <div className="relative w-full h-full flex overflow-hidden">
      <GridBackground />

      {/* filter sidebar */}
      <aside className="relative z-10 w-52 shrink-0 flex flex-col gap-5 border-r border-surface-3 bg-surface-1/80 backdrop-blur-sm p-4">
        <div>
          <span className="text-[10px] uppercase tracking-widest text-zinc-600 block mb-2">
            queue
          </span>
          <div className="flex flex-col gap-1">
            <FilterBtn
              active={needsReview === undefined}
              onClick={() => setNeedsReview(undefined)}
            >
              all cases
            </FilterBtn>
            <FilterBtn
              active={needsReview === true}
              onClick={() => setNeedsReview(true)}
              accent="text-neon-red"
            >
              needs review
            </FilterBtn>
          </div>
        </div>

        <div>
          <span className="text-[10px] uppercase tracking-widest text-zinc-600 block mb-2">
            classification
          </span>
          <div className="flex flex-col gap-1">
            {CLASSIFICATIONS.map(({ value, label }) => (
              <FilterBtn
                key={label}
                active={classification === value}
                onClick={() => setClassification(value)}
              >
                {label}
              </FilterBtn>
            ))}
          </div>
        </div>

        <div className="mt-auto text-[10px] text-zinc-700 tracking-wider">
          30s poll
        </div>
      </aside>

      {/* main feed */}
      <div className="relative z-10 flex flex-col flex-1 min-w-0 overflow-hidden">
        {loading ? (
          <div className="flex flex-col gap-3 p-4 overflow-y-auto h-full">
            {[1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="h-16 rounded-lg bg-surface-2/60 animate-pulse border border-surface-3"
                style={{ animationDelay: `${i * 80}ms` }}
              />
            ))}
          </div>
        ) : cases.length === 0 ? (
          <div className="flex items-center justify-center h-full text-xs text-zinc-700 tracking-widest uppercase">
            no case files
          </div>
        ) : (
          <div className="flex flex-col gap-2 p-4 overflow-y-auto h-full">
            {cases.map((c) => (
              <CaseCard key={c.id} summary={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function FilterBtn({
  active,
  onClick,
  children,
  accent = "text-neon-purple",
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
  accent?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-1.5 rounded text-xs tracking-wider transition-all duration-100 ${
        active
          ? `bg-surface-3 ${accent}`
          : "text-zinc-500 hover:text-zinc-300 hover:bg-surface-2"
      }`}
    >
      {children}
    </button>
  );
}
