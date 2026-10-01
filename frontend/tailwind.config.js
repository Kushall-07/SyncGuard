/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: {
          DEFAULT: "#0a0b0e",
          raised: "#12141a",
          elevated: "#181b22",
        },
        ink: {
          DEFAULT: "#edeff3",
          soft: "#8d96a8",
          faint: "#5b6270",
        },
        signal: {
          DEFAULT: "#22d3ee",
          light: "#67e8f9",
          dark: "#0e9bb3",
        },
        anomaly: {
          DEFAULT: "#f59e0b",
          light: "#fbbf65",
          dark: "#b9770a",
        },
        fusion: {
          DEFAULT: "#8b7cf6",
          light: "#ab9ff9",
          dark: "#6657d6",
        },
        // Legacy aliases kept so every page built on the old editorial theme
        // (sage = positive/synchronized, copper = warning/anomaly) resolves
        // straight onto the new dark forensic palette without a rename pass.
        sage: {
          DEFAULT: "#22d3ee",
          light: "#67e8f9",
          dark: "#0e9bb3",
        },
        copper: {
          DEFAULT: "#f59e0b",
          light: "#fbbf65",
          dark: "#b9770a",
        },
      },
      fontFamily: {
        display: ["'Space Grotesk'", "system-ui", "sans-serif"],
        body: ["'Inter'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      transitionTimingFunction: {
        editorial: "cubic-bezier(0.23, 1, 0.32, 1)",
      },
    },
  },
  plugins: [],
}
