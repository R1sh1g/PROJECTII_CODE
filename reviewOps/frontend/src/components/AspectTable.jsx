export default function AspectTable({ predictions }) {
  const rows = Array.isArray(predictions) ? predictions : [];

  return (
    <div className="mt-4">
      <h3 className="mb-3 text-sm font-semibold">
        Predictions ({rows.length})
      </h3>

      {rows.length === 0 ? (
        <div className="text-sm text-gray-500 dark:text-white/50">
          No predictions to display.
        </div>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-gray-200 dark:border-white/10">
          <table className="w-full text-sm">
            <thead className="bg-gray-100 dark:bg-black/40">
              <tr>
                <th className="px-3 py-2 text-left">Aspect</th>
                <th className="px-3 py-2 text-left">Aspect Conf</th>
                <th className="px-3 py-2 text-left">Sentiment</th>
                <th className="px-3 py-2 text-left">Sent Conf</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p, i) => (
                <tr
                  key={i}
                  className="border-t border-gray-200 dark:border-white/10"
                >
                  <td className="px-3 py-2">{p.aspect}</td>
                  <td className="px-3 py-2">
                    {Number(p.aspect_confidence).toFixed(3)}
                  </td>
                  <td className="px-3 py-2 font-semibold">
                    {p.sentiment}
                  </td>
                  <td className="px-3 py-2">
                    {Number(p.sentiment_confidence).toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
