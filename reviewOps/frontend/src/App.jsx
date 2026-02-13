// src/App.jsx
import { useEffect, useState } from "react";
import SingleReview from "./pages/SingleReview";
import CsvDashboard from "./pages/CsvDashboard";

export default function App() {
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem("theme");
    if (stored) return stored;

    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  function toggleTheme() {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }

  return (
    <div className="min-h-dvh bg-gray-50 text-gray-900 transition-colors dark:bg-[#0B1020] dark:text-white">
      {/* subtle background glow (dark mode only) */}
      <div className="pointer-events-none fixed inset-0 -z-10 hidden dark:block">
        <div className="absolute left-[-10%] top-[-20%] h-[520px] w-[520px] rounded-full bg-violet-600/20 blur-3xl" />
        <div className="absolute right-[-10%] top-[10%] h-[520px] w-[520px] rounded-full bg-cyan-500/10 blur-3xl" />
      </div>

      <div className="mx-auto max-w-6xl px-4 py-8">
        <header className="mb-8 flex flex-col gap-2 rounded-2xl border border-gray-200 bg-white p-5 shadow-sm backdrop-blur dark:border-white/10 dark:bg-white/5 dark:shadow-[0_0_0_1px_rgba(255,255,255,0.06)]">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">ReviewOps</h1>
              <p className="mt-1 text-sm text-gray-600 dark:text-white/60">
                Aspect-based sentiment analysis for single reviews and CSV
                batches.
              </p>
            </div>

            <div className="mt-3 flex items-center gap-2 sm:mt-0">
              <span className="rounded-full border border-gray-200 bg-gray-100 px-3 py-1 text-xs text-gray-700 dark:border-white/10 dark:bg-black/30 dark:text-white/70">
                Dashboard
              </span>
              <span className="rounded-full border border-gray-200 bg-gray-100 px-3 py-1 text-xs text-gray-700 dark:border-white/10 dark:bg-black/30 dark:text-white/70">
                Vite + React
              </span>

              {/* Theme toggle */}
              <button
                onClick={toggleTheme}
                aria-label="Toggle theme"
                className="relative ml-2 inline-flex h-6 w-11 items-center rounded-full bg-gray-300 transition-colors dark:bg-violet-600"
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                    theme === "dark" ? "translate-x-6" : "translate-x-1"
                  }`}
                />
              </button>
            </div>
          </div>
        </header>

        <main className="grid gap-6 lg:grid-cols-2">
          <SingleReview />
          <CsvDashboard />
        </main>

        <footer className="mt-10 text-center text-xs text-gray-500 dark:text-white/40">
          © {new Date().getFullYear()} ReviewOps
        </footer>
      </div>
    </div>
  );
}
