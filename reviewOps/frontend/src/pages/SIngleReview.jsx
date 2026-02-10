import { useState } from "react";
import { predictReview } from "../api";
import AspectTable from "../components/AspectTable";

export default function SingleReview() {
  const [review, setReview] = useState(
    "The food is great and reasonably priced."
  );
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
      setErr(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={card}>
      <h2>Single Review</h2>

      <textarea
        value={review}
        onChange={(e) => setReview(e.target.value)}
        rows={4}
        style={{ width: "100%", padding: 10 }}
      />

      <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
        <button onClick={onPredict} disabled={loading || !review.trim()}>
          {loading ? "Predicting..." : "Predict"}
        </button>
        {err && <span style={{ color: "crimson" }}>{err}</span>}
      </div>

      {out && (
        <>
          <div style={{ marginTop: 12, fontSize: 14, color: "#555" }}>
            <div>
              <b>Input:</b> {out.review}
            </div>
          </div>

          <AspectTable predictions={out?.predictions || []} />

          {/* DEBUG: remove later */}
          <pre style={pre}>{JSON.stringify(out, null, 2)}</pre>
        </>
      )}
    </div>
  );
}

const card = {
  border: "1px solid #e6e6e6",
  borderRadius: 12,
  padding: 16,
  background: "white",
};

const pre = {
  marginTop: 12,
  background: "#0b1020",
  color: "#d7e0ff",
  padding: 12,
  borderRadius: 10,
  overflowX: "auto",
  maxHeight: 260,
};
