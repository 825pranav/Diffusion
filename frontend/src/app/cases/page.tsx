import GridBackground from "@/components/GridBackground";

export default function CasesPage() {
  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <GridBackground />
      <div className="relative z-10 flex flex-col items-center gap-3 text-zinc-600">
        <div className="space-y-3 w-72">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-14 rounded-lg bg-zinc-800/50 animate-pulse border border-zinc-800" />
          ))}
        </div>
        <span className="text-xs tracking-widest uppercase">cases — day 7</span>
      </div>
    </div>
  );
}
