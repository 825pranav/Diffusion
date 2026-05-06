"use client";

export default function BottomBar() {
  return (
    <footer className="fixed bottom-0 right-0 left-[148px] h-8 bg-surface-container-lowest border-t-[0.5px] border-outline-variant flex items-center justify-between gap-xl px-lg z-50">
      <div className="flex items-center gap-xl">
        <div className="flex items-center gap-xs font-mono-data text-mono-data text-tertiary-fixed-dim">
          <span className="material-symbols-outlined text-[14px]">speed</span>
          <span>Latency: 12ms</span>
        </div>
        <div className="flex items-center gap-xs font-mono-data text-mono-data text-on-surface-variant">
          <span className="material-symbols-outlined text-[14px]">schedule</span>
          <span>Last Event: 2s ago</span>
        </div>
        <div className="flex items-center gap-xs font-mono-data text-mono-data text-on-surface-variant">
          <span className="material-symbols-outlined text-[14px]">hub</span>
          <span>Active Nodes: 1,204</span>
        </div>
      </div>
      <div className="flex items-center gap-xs">
        <span className="w-1.5 h-1.5 rounded-full bg-primary" />
        <span className="font-mono-label text-[10px] text-primary uppercase">System Optimal</span>
      </div>
    </footer>
  );
}
