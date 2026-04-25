import GridBackground from "@/components/GridBackground";

export default function AgentPage() {
  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <GridBackground />
      <div className="relative z-10 flex flex-col items-center gap-3 text-zinc-600">
        <div className="space-y-2 w-64">
          {[80, 60, 90, 50].map((w, i) => (
            <div
              key={i}
              className="h-2 rounded bg-zinc-800 animate-pulse"
              style={{ width: `${w}%`, animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
        <span className="text-xs tracking-widest uppercase">agent — day 5</span>
      </div>
    </div>
  );
}
