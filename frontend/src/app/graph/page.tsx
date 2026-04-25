import GridBackground from "@/components/GridBackground";

export default function GraphPage() {
  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <GridBackground />
      <div className="relative z-10 flex flex-col items-center gap-3 text-zinc-600">
        <div className="w-32 h-32 rounded-full border border-zinc-800 animate-pulse" />
        <span className="text-xs tracking-widest uppercase">graph — day 2</span>
      </div>
    </div>
  );
}
