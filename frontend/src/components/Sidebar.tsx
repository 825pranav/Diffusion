"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/",       icon: "dashboard",              label: "Dashboard"    },
  { href: "/graph",  icon: "query_stats",             label: "Propagation"  },
  { href: "/cases",  icon: "psychology",              label: "Intelligence" },
  { href: "/agent",  icon: "settings_input_component",label: "Operations"   },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="h-screen w-[148px] fixed left-0 top-0 border-r-[0.5px] border-outline-variant bg-surface-dim flex flex-col py-lg px-md z-50">
      {/* Logo */}
      <div className="mb-xl">
        <h2 className="font-h2 text-h2 font-bold text-primary">Diffusion</h2>
        <p className="font-body-sm text-[11px] text-on-surface-variant leading-tight">Propagation Suite</p>
      </div>

      {/* Nav */}
      <nav className="flex-1 flex flex-col gap-xs">
        {NAV.map(({ href, icon, label }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex flex-col items-center justify-center py-sm rounded transition-colors ${
                active
                  ? "text-primary border-r-2 border-primary bg-primary/10"
                  : "text-on-surface-variant hover:bg-surface-container-highest"
              }`}
            >
              <span className="material-symbols-outlined mb-xs">{icon}</span>
              <span className="font-body-sm text-[10px]">{label}</span>
            </Link>
          );
        })}
      </nav>

      {/* CTA */}
      <button className="mt-xl bg-primary text-on-primary font-mono-label text-mono-label py-sm rounded hover:brightness-110 transition-all uppercase">
        + New feature
      </button>

      {/* Footer links */}
      <div className="mt-auto pt-lg border-t-[0.5px] border-outline-variant flex flex-col gap-sm">
        <a href="#" className="flex items-center gap-sm text-on-surface-variant hover:text-primary transition-colors">
          <span className="material-symbols-outlined text-[18px]">settings_heart</span>
          <span className="font-mono-label text-[10px]">Health</span>
        </a>
        <a href="#" className="flex items-center gap-sm text-on-surface-variant hover:text-primary transition-colors">
          <span className="material-symbols-outlined text-[18px]">help</span>
          <span className="font-mono-label text-[10px]">Support</span>
        </a>
      </div>
    </aside>
  );
}
