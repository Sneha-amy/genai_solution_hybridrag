import React from "react";

export default function SummaryCards({ results }) {
  const high = results.anomalies.filter(a => a.severity === "high").length;
  const med = results.anomalies.filter(a => a.severity === "medium").length;
  const low = results.anomalies.filter(a => a.severity === "low").length;

  return (
    <div className="cards">
      <div className="card kpi">
        <h3>Total Records</h3>
        <p>{results.num_records}</p>
      </div>

      <div className="card kpi high">
        <h3>High Severity</h3>
        <p>{high}</p>
      </div>

      <div className="card kpi medium">
        <h3>Medium Severity</h3>
        <p>{med}</p>
      </div>

      <div className="card kpi low">
        <h3>Low Severity</h3>
        <p>{low}</p>
      </div>
    </div>
  );
}
