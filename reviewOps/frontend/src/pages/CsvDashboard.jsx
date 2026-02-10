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
      setErr(String(e.message || e));
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
        sentimentByAspect[p.aspect] = sentimentByAspect[p.aspect] || { negative: 0, neutral: 0, positive: 0 };
        sentimentByAspect[p.aspect][p.sentiment] = (sentimentByAspect[p.aspect][p.sentiment] || 0) + 1;
      }
    }

    const topAspects = Object.entries(aspectFreq).sort((a, b) => b[1] - a[1]).slice(0, 10);

    const topNegative = Object.entries(sentimentByAspect)
      .map(([aspect, dist]) => ({ aspect, negative: dist.negative || 0, total: (dist.negative||0)+(dist.neutral||0)+(dist.positive||0) }))
      .sort((a, b) => b.negative - a.negative)
      .slice(0, 10);

    return { aspectFreq, sentimentByAspect, topAspects, topNegative };
  }, [resp]);

  return (
    <div style={card}>
      <h2>CSV Upload + Dashboard</h2>

      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button onClick={onRun} disabled={!file || loading}>
          {loading ? "Running..." : "Run ACD + ASC"}
        </button>
        {err && <span style={{ color: "crimson" }}>{err}</span>}
      </div>

      {resp?.count != null && (
        <div style={{ marginTop: 12, color: "#444" }}>
          <b>Processed reviews:</b> {resp.count}
        </div>
      )}

      {metrics && (
        <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div style={panel}>
            <h3>Top Aspects</h3>
            <ul>
              {metrics.topAspects.map(([a, c]) => (
                <li key={a}>
                  {a}: <b>{c}</b>
                </li>
              ))}
            </ul>
          </div>

          <div style={panel}>
            <h3>Top Negative Aspects</h3>
            <ul>
              {metrics.topNegative.map((x) => (
                <li key={x.aspect}>
                  {x.aspect}: <b>{x.negative}</b> negatives
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {resp?.results?.length ? (
        <div style={{ marginTop: 16 }}>
          <h3>Rows (first 20)</h3>
          <pre style={pre}>{JSON.stringify(resp.results.slice(0, 20), null, 2)}</pre>
        </div>
      ) : null}
    </div>
  );
}

const card = {
  border: "1px solid #cf0404",
  borderRadius: 12,
  padding: 16,
  background: "white",
  marginTop: 16,
};

const panel = {
  border: "1px solid #eee",
  borderRadius: 10,
  padding: 12,
  background: "#fafafa",
};

const pre = {
  background: "#0b1020",
  color: "#e4560f",
  padding: 12,
  borderRadius: 10,
  overflowX: "auto",
  maxHeight: 360,
};
