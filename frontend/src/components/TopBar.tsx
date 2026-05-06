"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useConnection } from "@/components/ConnectionProvider";

const TABS = [
  { href: "/",      label: "Velocity"   },
  { href: "/graph", label: "Graph"      },
  { href: "/cases", label: "Case Files" },
  { href: "/agent", label: "Console"    },
];

export default function TopBar() {
  const pathname = usePathname();
  const { connected } = useConnection();

  return (
    <header className="fixed top-0 right-0 left-[148px] h-[56px] border-b-[0.5px] border-outline-variant bg-surface-dim flex justify-between items-center px-lg z-40">
      <div className="flex items-center gap-xl">
        <span className="font-h3 text-h3 font-bold text-primary">Diffusion</span>
        <nav className="flex gap-lg">
          {TABS.map(({ href, label }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`font-mono-label text-mono-label uppercase tracking-wider transition-colors ${
                  active
                    ? "text-primary border-b-2 border-primary pb-1"
                    : "text-on-surface-variant hover:text-primary"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="flex items-center gap-md">
        <button className="text-on-surface-variant hover:text-primary transition-colors active:scale-95">
          <span className="material-symbols-outlined">settings</span>
        </button>
        {/* WS status dot */}
        <div className="flex items-center gap-xs">
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-primary" : "bg-tertiary-container animate-pulse"}`} />
          <span className="font-mono-label text-[10px] text-on-surface-variant uppercase">
            {connected ? "live" : "reconnecting"}
          </span>
        </div>
      </div>
    </header>
  );
}
