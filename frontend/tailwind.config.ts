import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        neon: {
          orange: "#f97316",
          yellow: "#eab308",
          purple: "#a855f7",
          red: "#ef4444",
          green: "#22c55e",
          blue: "#3b82f6",
        },
        surface: {
          DEFAULT: "#0d0d0f",
          1: "#111114",
          2: "#18181d",
          3: "#1f1f26",
        },
      },
      boxShadow: {
        "neon-orange": "0 0 12px #f97316aa",
        "neon-purple": "0 0 12px #a855f7aa",
        "neon-red": "0 0 12px #ef4444aa",
        "neon-green": "0 0 12px #22c55eaa",
      },
      fontFamily: {
        mono: ["var(--font-geist-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
