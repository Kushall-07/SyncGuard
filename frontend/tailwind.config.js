/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#14170f",
        forest: {
          DEFAULT: "#132018",
          light: "#1c2e23",
        },
        ivory: "#f6f1e7",
        stone: {
          DEFAULT: "#e7ddc9",
          dark: "#d8cbae",
        },
        sage: {
          DEFAULT: "#7c9478",
          light: "#a3b89e",
          dark: "#5f7a5d",
        },
        copper: {
          DEFAULT: "#ab7440",
          light: "#c69361",
          dark: "#8a5c30",
        },
      },
      fontFamily: {
        display: ["'Fraunces'", "serif"],
        body: ["'Inter'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      transitionTimingFunction: {
        editorial: "cubic-bezier(0.22, 1, 0.36, 1)",
      },
    },
  },
  plugins: [],
}
