// src/pages/CsvDashboard.jsx
import { useMemo, useState } from "react";
import { predictCsv } from "../api";

export default function CsvDashboard() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [err, setErr] = useState("");

  async function onRun() {
    setErr("");
    setLoading(true);
    try {
      const data = await predictCsv(file);
      setResp(data);
    } catch (e) {
      setErr(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  const metrics = useMemo(() => {
    if (!resp?.results?.length) return null;
    const aspectFreq = {};

    for (const r of resp.results) {
      for (const p of r.predictions || []) {
        aspectFreq[p.aspect] = (aspectFreq[p.aspect] || 0) + 1;
      }
    }

    const topAspects = Object.entries(aspectFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    return { topAspects };
  }, [resp]);

  return (
    <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-white/10 dark:bg-white/5 dark:backdrop-blur">
      <h2 className="text-base font-semibold">CSV Batch Analysis</h2>

      <div className="mt-4 flex gap-3">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="
    block w-full cursor-pointer rounded-xl border
    border-gray-300 bg-gray-50 px-3 py-2 text-sm text-gray-800
    file:mr-4 file:rounded-lg file:border-0
    file:bg-violet-600 file:px-4 file:py-2
    file:text-sm file:font-semibold file:text-white
    hover:file:bg-violet-500
    focus:outline-none focus:ring-2 focus:ring-violet-500/30
    dark:border-white/10 dark:bg-black/30 dark:text-white
  "
        />

        <button
          onClick={onRun}
          disabled={!file || loading}
          className="rounded-xl bg-violet-600 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-500 disabled:bg-gray-300"
        >
          {loading ? "Running…" : "Run"}
        </button>
      </div>

      {metrics && (
        <ul className="mt-4 space-y-2">
          {metrics.topAspects.map(([a, c]) => (
            <li
              key={a}
              className="flex justify-between rounded-xl border border-gray-200 bg-gray-50 px-3 py-2 dark:border-white/10 dark:bg-black/20"
            >
              <span>{a}</span>
              <span className="font-semibold">{c}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
