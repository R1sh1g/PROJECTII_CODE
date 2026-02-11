export default function AspectTable({ predictions }) {
  const rows = Array.isArray(predictions) ? predictions : [];

  return (
    <div style={{ marginTop: 16, color: "#efefef" }}>
      <h3 className="text-white">Predictions ({rows.length})</h3>

      {rows.length === 0 ? (
        <div style={{ color: "#777" }}>No predictions to display.</div>
      ) : (
        <div className="text-white">
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={th}>Aspect</th>
              <th style={th}>Aspect Conf</th>
              <th style={th}>Sentiment</th>
              <th style={th}>Sent Conf</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p, i) => (
              <tr key={i}>
                <td style={td}>{p.aspect}</td>
                <td style={td}>{Number(p.aspect_confidence).toFixed(3)}</td>
                <td style={{ ...td, fontWeight: 700 }}>{p.sentiment}</td>
                <td style={td}>{Number(p.sentiment_confidence).toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
    </div>
  );
}

const th = { textAlign: "left", borderBottom: "1px solid #ddd", padding: "8px" };
const td = { borderBottom: "1px solid #eee", padding: "8px" };
