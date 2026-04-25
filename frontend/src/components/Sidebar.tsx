"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/graph", label: "Graph", icon: "⬡" },
  { href: "/agent", label: "Agent", icon: "◈" },
  { href: "/cases", label: "Cases", icon: "◻" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex flex-col w-16 border-r border-surface-3 bg-surface-1 shrink-0">
      <div className="flex items-center justify-center h-14 border-b border-surface-3">
        <span className="text-neon-purple text-lg font-bold tracking-widest">D</span>
      </div>
      <nav className="flex flex-col items-center gap-1 pt-4">
        {links.map(({ href, label, icon }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              title={label}
              className={`
                flex flex-col items-center justify-center w-12 h-12 rounded-lg text-xs gap-1 transition-all duration-150
                ${active
                  ? "bg-surface-3 text-neon-purple shadow-neon-purple"
                  : "text-zinc-500 hover:text-zinc-300 hover:bg-surface-2"}
              `}
            >
              <span className="text-base leading-none">{icon}</span>
              <span className="leading-none">{label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
