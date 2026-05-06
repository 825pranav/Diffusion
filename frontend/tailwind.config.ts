import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // ── Backgrounds ──────────────────────────────────────────────
        background:                 "#0a0d0f",  // base void
        "surface":                  "#0d1115",  // sidebar / topbar
        "surface-dim":              "#0d1115",  // alias
        "surface-bright":           "#1e2428",
        "surface-container-lowest": "#080d0a",  // terminal bg
        "surface-container-low":    "#0d1115",
        "surface-container":        "#111820",  // card surfaces
        "surface-container-high":   "#162018",
        "surface-container-highest":"#1e2820",
        "surface-variant":          "#1e2428",

        // ── Borders ───────────────────────────────────────────────────
        "outline-variant":          "#1e2428",  // 0.5px borders
        "outline":                  "#3a5045",  // muted / secondary borders

        // ── Text hierarchy ────────────────────────────────────────────
        "on-surface":               "#e8edf2",  // headlines
        "on-surface-variant":       "#c8e8e0",  // body
        "on-background":            "#e8edf2",
        "inverse-surface":          "#e8edf2",
        "inverse-on-surface":       "#111820",

        // ── Primary — electric teal ───────────────────────────────────
        "primary":                  "#4dd9ac",
        "on-primary":               "#00382a",
        "primary-container":        "#0f1a16",  // active sidebar bg
        "on-primary-container":     "#4dd9ac",
        "primary-fixed":            "#72facb",
        "primary-fixed-dim":        "#4dd9ac",
        "on-primary-fixed":         "#00211a",
        "on-primary-fixed-variant": "#005040",
        "inverse-primary":          "#006c51",
        "surface-tint":             "#4dd9ac",

        // ── Secondary — steel blue (GitHub) ──────────────────────────
        "secondary":                    "#6aa0c8",
        "on-secondary":                 "#00344e",
        "secondary-container":          "#003a58",
        "on-secondary-container":       "#6aa0c8",
        "secondary-fixed":              "#afd4f0",
        "secondary-fixed-dim":          "#6aa0c8",
        "on-secondary-fixed":           "#001e2f",
        "on-secondary-fixed-variant":   "#003a58",

        // ── Tertiary — amber alert ────────────────────────────────────
        "tertiary":                    "#ff9500",
        "on-tertiary":                 "#3a1e00",
        "tertiary-container":          "#ff9500",  // amber chips
        "on-tertiary-container":       "#3a1e00",
        "tertiary-fixed":              "#ffdcbf",
        "tertiary-fixed-dim":          "#ff9500",
        "on-tertiary-fixed":           "#2d1600",
        "on-tertiary-fixed-variant":   "#6a3b00",

        // ── Error ─────────────────────────────────────────────────────
        "error":                "#ff4d4d",
        "on-error":             "#690005",
        "error-container":      "#93000a",
        "on-error-container":   "#ffdad6",
      },

      fontFamily: {
        "h1":         ["var(--font-space-grotesk)", "Space Grotesk", "sans-serif"],
        "h2":         ["var(--font-space-grotesk)", "Space Grotesk", "sans-serif"],
        "h3":         ["var(--font-space-grotesk)", "Space Grotesk", "sans-serif"],
        "body-lg":    ["var(--font-manrope)",        "Manrope",        "sans-serif"],
        "body-sm":    ["var(--font-manrope)",        "Manrope",        "sans-serif"],
        "mono-data":  ["var(--font-space-grotesk)", "Space Grotesk", "sans-serif"],
        "mono-label": ["var(--font-space-grotesk)", "Space Grotesk", "sans-serif"],
      },
      fontSize: {
        "h1":         ["32px", { lineHeight: "1.2", letterSpacing: "-0.02em", fontWeight: "700" }],
        "h2":         ["24px", { lineHeight: "1.2", letterSpacing: "-0.01em", fontWeight: "600" }],
        "h3":         ["18px", { lineHeight: "1.3",                           fontWeight: "600" }],
        "body-lg":    ["16px", { lineHeight: "1.5",                           fontWeight: "400" }],
        "body-sm":    ["14px", { lineHeight: "1.5",                           fontWeight: "400" }],
        "mono-data":  ["13px", { lineHeight: "1.4",                           fontWeight: "400" }],
        "mono-label": ["12px", { lineHeight: "1",   letterSpacing: "0.05em",  fontWeight: "500" }],
      },
      spacing: {
        xs:     "4px",
        sm:     "8px",
        md:     "16px",
        lg:     "24px",
        xl:     "32px",
        gutter: "12px",
        margin: "20px",
      },
      borderRadius: {
        DEFAULT: "0.25rem",
        sm:      "0.25rem",
        md:      "0.5rem",
        lg:      "0.75rem",
        xl:      "1.5rem",
        full:    "9999px",
      },
    },
  },
  plugins: [],
};
export default config;
