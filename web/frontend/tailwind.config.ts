import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        finlove: {
          bg: "#020617",
          panel: "#02081b",
          accent: "#22c55e",
          accentSoft: "#064e3b",
        },
      },
      boxShadow: {
        "soft-elevated":
          "0 18px 45px rgba(15, 23, 42, 0.85), 0 0 0 1px rgba(148, 163, 184, 0.1)",
      },
      borderRadius: {
        xl: "0.9rem",
      },
    },
  },
  plugins: [],
};

export default config;



