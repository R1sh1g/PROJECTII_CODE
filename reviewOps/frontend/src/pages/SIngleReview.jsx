// src/pages/SingleReview.jsx
import { useState } from "react";
import { predictReview } from "../api";
import AspectTable from "../components/AspectTable";

export default function SingleReview() {
  const [review, setReview] = useState("The food is great and reasonably priced.");
  const [loading, setLoading] = useState(false);
  const [out, setOut] = useState(null);
  const [err, setErr] = useState("");

  async function onPredict() {
    setErr("");
    setLoading(true);
    try {
      const data = await predictReview(review);
      setOut(data);
    } catch (e) {
      setErr(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  const canPredict = !!review.trim() && !loading;

  return (
    <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-white/10 dark:bg-white/5 dark:shadow-[0_0_0_1px_rgba(255,255,255,0.06)] dark:backdrop-blur">
      <div className="flex flex-col gap-1">
        <h2 className="text-base font-semibold tracking-tight">Single Review</h2>
        <p className="text-sm text-gray-600 dark:text-white/60">
          Run aspect detection + sentiment classification on one review.
        </p>
      </div>

      <div className="mt-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-white/80">
          Review text
        </label>
        <textarea
          value={review}
          onChange={(e) => setReview(e.target.value)}
          rows={5}
          placeholder="Type or paste a review…"
          className="mt-2 w-full resize-y rounded-2xl border border-gray-300 bg-gray-50 px-4 py-3 text-sm text-gray-900 shadow-sm outline-none transition placeholder:text-gray-400 focus:border-gray-400 focus:ring-2 focus:ring-violet-500/30 dark:border-white/10 dark:bg-black/30 dark:text-white dark:placeholder:text-white/40"
        />
        <div className="mt-2 flex items-center justify-between text-xs text-gray-500 dark:text-white/50">
          <span>Tip: include multiple sentences for better coverage.</span>
          <span>{review.length.toLocaleString()} chars</span>
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={onPredict}
            disabled={!canPredict}
            className="
              inline-flex items-center justify-center rounded-xl
              bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm
              transition hover:bg-violet-500
              disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-500
              dark:disabled:bg-white/10 dark:disabled:text-white/40
            "
          >
            {loading ? (
              <span className="inline-flex items-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                Predicting…
              </span>
            ) : (
              "Predict"
            )}
          </button>

          <button
            type="button"
            onClick={() => {
              setErr("");
              setOut(null);
            }}
            className="
              inline-flex items-center justify-center rounded-xl
              border border-gray-300 bg-white px-4 py-2.5 text-sm font-semibold
              text-gray-800 shadow-sm transition hover:bg-gray-50
              dark:border-white/10 dark:bg-white/5 dark:text-white/80 dark:hover:bg-white/10
            "
          >
            Clear
          </button>
        </div>

        {err ? (
          <div
            role="alert"
            className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200"
          >
            {err}
          </div>
        ) : null}
      </div>

      {out && (
        <div className="mt-6 space-y-4">
          <div className="rounded-2xl border border-gray-200 bg-gray-50 p-4 dark:border-white/10 dark:bg-black/20">
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-white/50">
              Input
            </div>
            <div className="mt-2 text-sm text-gray-900 dark:text-white/90">
              {out.review}
            </div>
          </div>

          <div className="rounded-2xl border border-gray-200 bg-gray-50 p-4 dark:border-white/10 dark:bg-black/20">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold">Predictions</h3>
              <span className="text-xs text-gray-500 dark:text-white/50">
                {(out?.predictions?.length || 0).toLocaleString()} items
              </span>
            </div>

            <AspectTable predictions={out?.predictions || []} />
          </div>

          {/* Debug JSON */}
          <details className="rounded-2xl border border-gray-200 bg-gray-50 p-4 dark:border-white/10 dark:bg-black/20">
            <summary className="cursor-pointer select-none text-sm font-semibold text-gray-800 dark:text-white/85">
              Debug JSON
            </summary>
            <div className="mt-3 overflow-hidden rounded-2xl border border-gray-200 dark:border-white/10">
              <pre className="max-h-[260px] overflow-x-auto bg-gray-900 p-4 text-xs leading-relaxed text-gray-100 dark:bg-black/60 dark:text-white/80">
                {JSON.stringify(out, null, 2)}
              </pre>
            </div>
          </details>
        </div>
      )}
    </section>
  );
}
