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
      <div className="mx-auto max-w-6xl px-4 py-8">
        <header className="mb-8 flex items-center justify-between rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-white/10 dark:bg-white/5">
          <h1 className="text-2xl font-bold">ReviewOps</h1>

          <button
            onClick={toggleTheme}
            className="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-300 transition dark:bg-violet-600"
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                theme === "dark" ? "translate-x-6" : "translate-x-1"
              }`}
            />
          </button>
        </header>

        <main className="grid gap-6 lg:grid-cols-2">
          <SingleReview />
          <CsvDashboard />
        </main>
      </div>
    </div>
  );
}
