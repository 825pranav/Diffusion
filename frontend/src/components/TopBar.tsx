"use client";

import { usePathname } from "next/navigation";
import { useConnection } from "@/components/ConnectionProvider";

const titles: Record<string, string> = {
  "/graph": "live propagation graph",
  "/agent": "agent thought stream",
  "/cases": "case file feed",
};

export default function TopBar() {
  const pathname = usePathname();
  const { connected } = useConnection();
  const title = Object.entries(titles).find(([k]) => pathname.startsWith(k))?.[1] ?? "diffusion";

  return (
    <header className="flex items-center justify-between h-14 px-6 border-b border-surface-3 bg-surface-1 shrink-0">
      <span className="text-sm text-zinc-400 tracking-widest uppercase">{title}</span>
      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <span
          className={`w-2 h-2 rounded-full transition-all duration-300 ${
            connected ? "bg-neon-green shadow-neon-green animate-pulse" : "bg-zinc-600"
          }`}
        />
        {connected ? "connected" : "disconnected"}
      </div>
    </header>
  );
}
