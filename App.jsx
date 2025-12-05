import React, { useState } from "react";
import axios from "axios";
import Sidebar from "./components/Sidebar";
import FileDropzone from "./components/FileDropzone";
import SummaryCards from "./components/SummaryCards";
import AnomalyTable from "./components/AnomalyTable";
import Charts from "./components/Charts";
import "./App.css";

function App() {
  const [companyName, setCompanyName] = useState("");
  const [periodLabel, setPeriodLabel] = useState("");
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [darkMode, setDarkMode] = useState(false);

  const analyze = async () => {
    if (!companyName || !periodLabel || files.length === 0) {
      alert("Please fill all fields and upload at least one file.");
      return;
    }

    const formData = new FormData();
    formData.append("company_name", companyName);
    formData.append("period_label", periodLabel);

    for (let f of files) formData.append("files", f);

    setLoading(true);

    try {
      const res = await axios.post("http://127.0.0.1:8000/analyze", formData);
      setResults(res.data);
    } catch (err) {
      alert("Error: " + err.message);
    }

    setLoading(false);
  };

  return (
    <div className={darkMode ? "app dark" : "app"}>
      <Sidebar darkMode={darkMode} setDarkMode={setDarkMode} />

      <div className="main">
        <h1 className="title">Financial Anomaly Dashboard</h1>

        <div className="input-grid">
          <div className="input-card">
            <label>Company Name</label>
            <input
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              placeholder="e.g. DemoCo"
            />

            <label>Period Label</label>
            <input
              value={periodLabel}
              onChange={(e) => setPeriodLabel(e.target.value)}
              placeholder="e.g. FY2024"
            />

            <FileDropzone setFiles={setFiles} />
            <button className="analyze-btn" onClick={analyze}>
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>
        </div>

        {results && (
          <>
            <SummaryCards results={results} />

            <Charts anomalies={results.anomalies} />

            <AnomalyTable anomalies={results.anomalies} />
          </>
        )}
      </div>
    </div>
  );
}

export default App;
