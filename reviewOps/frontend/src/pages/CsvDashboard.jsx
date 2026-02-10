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
    const sentimentByAspect = {};

    for (const r of resp.results) {
      for (const p of r.predictions || []) {
        aspectFreq[p.aspect] = (aspectFreq[p.aspect] || 0) + 1;
        sentimentByAspect[p.aspect] =
          sentimentByAspect[p.aspect] || { negative: 0, neutral: 0, positive: 0 };
        sentimentByAspect[p.aspect][p.sentiment] =
          (sentimentByAspect[p.aspect][p.sentiment] || 0) + 1;
      }
    }

    const topAspects = Object.entries(aspectFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    const topNegative = Object.entries(sentimentByAspect)
      .map(([aspect, dist]) => ({
        aspect,
        negative: dist.negative || 0,
        neutral: dist.neutral || 0,
        positive: dist.positive || 0,
        total: (dist.negative || 0) + (dist.neutral || 0) + (dist.positive || 0),
      }))
      .sort((a, b) => b.negative - a.negative)
      .slice(0, 10);

    return { aspectFreq, sentimentByAspect, topAspects, topNegative };
  }, [resp]);

  const canRun = !!file && !loading;

  return (
    <section className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.06)] backdrop-blur">
      <div className="flex flex-col gap-1">
        <h2 className="text-base font-semibold tracking-tight text-white">
          CSV Batch Analysis
        </h2>
        <p className="text-sm text-white/60">
          Upload a CSV, run ACD + ASC, and inspect aggregate metrics.
        </p>
      </div>

      <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-1 flex-col gap-2 sm:flex-row sm:items-center">
          <label className="block w-full">
            <span className="sr-only">Choose CSV file</span>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full cursor-pointer rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-sm text-white/80 file:mr-3 file:cursor-pointer file:rounded-lg file:border-0 file:bg-white/10 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-white/15 focus:outline-none focus:ring-2 focus:ring-violet-500/30"
            />
          </label>

          <button
            onClick={onRun}
            disabled={!canRun}
            className="inline-flex items-center justify-center rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/40"
          >
            {loading ? (
              <span className="inline-flex items-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                Running…
              </span>
            ) : (
              "Run ACD + ASC"
            )}
          </button>
        </div>

        {err ? (
          <div
            role="alert"
            className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-sm text-rose-200"
          >
            {err}
          </div>
        ) : null}
      </div>

      {resp?.count != null && (
        <div className="mt-4 rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white/70">
          <span className="font-semibold text-white">Processed reviews:</span>{" "}
          {resp.count}
        </div>
      )}

      {metrics ? (
        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">Top Aspects</h3>
              <span className="text-xs text-white/50">Top 10</span>
            </div>

            <ul className="mt-3 space-y-2">
              {metrics.topAspects.map(([a, c]) => (
                <li
                  key={a}
                  className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-3 py-2"
                >
                  <span className="truncate text-sm text-white/85">{a}</span>
                  <span className="ml-3 inline-flex shrink-0 items-center rounded-lg bg-white/10 px-2 py-0.5 text-xs font-semibold text-white">
                    {c}
                  </span>
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">Top Negative Aspects</h3>
              <span className="text-xs text-white/50">Top 10</span>
            </div>

            <ul className="mt-3 space-y-2">
              {metrics.topNegative.map((x) => (
                <li
                  key={x.aspect}
                  className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-3 py-2"
                >
                  <span className="truncate text-sm text-white/85">{x.aspect}</span>

                  <div className="ml-3 flex shrink-0 items-center gap-2">
                    <span className="inline-flex items-center rounded-lg bg-rose-500/20 px-2 py-0.5 text-xs font-semibold text-rose-100">
                      {x.negative} neg
                    </span>
                    <span className="text-xs text-white/50">/ {x.total} total</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : (
        <div className="mt-5 rounded-2xl border border-white/10 bg-black/20 p-4 text-sm text-white/60">
          Upload a CSV and run analysis to see top aspects and sentiment distributions.
        </div>
      )}

      {resp?.results?.length ? (
        <div className="mt-5">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">Rows (first 20)</h3>
            <span className="text-xs text-white/50">
              Showing {Math.min(20, resp.results.length)} of {resp.results.length}
            </span>
          </div>

          <div className="mt-3 overflow-hidden rounded-2xl border border-white/10">
            <pre className="max-h-[360px] overflow-x-auto bg-black/60 p-4 text-xs leading-relaxed text-white/80">
              {JSON.stringify(resp.results.slice(0, 20), null, 2)}
            </pre>
          </div>
        </div>
      ) : null}
    </section>
  );
}
