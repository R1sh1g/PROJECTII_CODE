const API = (import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");

export async function predictReview(review) {
  const res = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function predictCsv(file) {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${API}/predict_csv`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
