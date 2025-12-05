import React from "react";

export default function AnomalyTable({ anomalies }) {
  return (
    <div className="card table-card">
      <h2>Anomalies</h2>

      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Category</th>
            <th>Statement</th>
            <th>Period</th>
            <th>Severity</th>
            <th>Title</th>
            <th>Description</th>
          </tr>
        </thead>

        <tbody>
          {anomalies.map(a => (
            <tr key={a.id}>
              <td>{a.id}</td>
              <td>{a.category}</td>
              <td>{a.statement_type}</td>
              <td>{a.period}</td>
              <td className={"sev-" + a.severity}>{a.severity}</td>
              <td>{a.title}</td>
              <td>{a.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
