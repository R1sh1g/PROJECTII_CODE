// src/App.jsx
import SingleReview from "./pages/SingleReview";
import CsvDashboard from "./pages/CsvDashboard";

export default function App() {
  return (
    <div className="min-h-dvh bg-[#0B1020] text-white">
      {/* subtle background glow */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute left-[-10%] top-[-20%] h-[520px] w-[520px] rounded-full bg-violet-600/20 blur-3xl" />
        <div className="absolute right-[-10%] top-[10%] h-[520px] w-[520px] rounded-full bg-cyan-500/10 blur-3xl" />
      </div>

      <div className="mx-auto max-w-6xl px-4 py-8">
        <header className="mb-8 flex flex-col gap-2 rounded-2xl border border-white/10 bg-white/5 p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.06)] backdrop-blur">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">ReviewOps</h1>
              <p className="mt-1 text-sm text-white/60">
                Aspect-based sentiment analysis for single reviews and CSV batches.
              </p>
            </div>

            <div className="mt-3 flex items-center gap-2 sm:mt-0">
              <span className="rounded-full border border-white/10 bg-black/30 px-3 py-1 text-xs text-white/70">
                Dashboard
              </span>
              <span className="rounded-full border border-white/10 bg-black/30 px-3 py-1 text-xs text-white/70">
                Vite + React
              </span>
            </div>
          </div>
        </header>

        <main className="grid gap-6 lg:grid-cols-2">
          <SingleReview />
          <CsvDashboard />
        </main>

        <footer className="mt-10 text-center text-xs text-white/40">
          © {new Date().getFullYear()} ReviewOps
        </footer>
      </div>
    </div>
  );
}
