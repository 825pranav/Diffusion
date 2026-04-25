"use client";

export default function GridBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      {/* grid lines */}
      <svg
        className="absolute inset-0 w-full h-full opacity-[0.04]"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#a855f7" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>

      {/* radial vignette — keeps edges dark */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 80% 60% at 50% 50%, transparent 30%, #0d0d0f 100%)",
        }}
      />

      {/* subtle corner glows */}
      <div
        className="absolute -top-32 -left-32 w-96 h-96 rounded-full opacity-10"
        style={{ background: "radial-gradient(circle, #a855f7 0%, transparent 70%)" }}
      />
      <div
        className="absolute -bottom-32 -right-32 w-96 h-96 rounded-full opacity-10"
        style={{ background: "radial-gradient(circle, #3b82f6 0%, transparent 70%)" }}
      />
    </div>
  );
}
