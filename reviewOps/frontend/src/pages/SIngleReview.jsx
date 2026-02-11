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
    <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-white/10 dark:bg-white/5 dark:backdrop-blur">
      <div className="flex flex-col gap-1">
        <h2 className="text-base font-semibold tracking-tight">Single Review</h2>
        <p className="text-sm text-gray-500 dark:text-white/60">
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
          className="mt-2 w-full resize-y rounded-2xl border border-gray-300 bg-gray-50 px-4 py-3 text-sm text-gray-900 outline-none focus:ring-2 focus:ring-violet-500/30 dark:border-white/10 dark:bg-black/30 dark:text-white"
        />
      </div>

      <div className="mt-4 flex gap-2">
        <button
          onClick={onPredict}
          disabled={!canPredict}
          className="rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-violet-500 disabled:bg-gray-300"
        >
          {loading ? "Predicting…" : "Predict"}
        </button>

        <button
          onClick={() => {
            setErr("");
            setOut(null);
          }}
          className="rounded-xl border border-gray-300 px-4 py-2.5 text-sm font-semibold dark:border-white/10"
        >
          Clear
        </button>
      </div>

      {err && (
        <div className="mt-4 rounded-xl bg-rose-500/10 p-3 text-sm text-rose-600 dark:text-rose-200">
          {err}
        </div>
      )}

      {out && (
        <div className="mt-6">
          <AspectTable predictions={out?.predictions || []} />
        </div>
      )}
    </section>
  );
}
