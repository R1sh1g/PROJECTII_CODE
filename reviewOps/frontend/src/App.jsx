import SingleReview from "./pages/SIngleReview";
import CsvDashboard from "./pages/CsvDashboard";

export default function App() {
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: 16, fontFamily: "system-ui, Arial" }}>
      <h1>ReviewOps</h1>
      <SingleReview />
      <CsvDashboard />
    </div>
  );
}
