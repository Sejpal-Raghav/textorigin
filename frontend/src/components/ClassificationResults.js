const SCORE_ROWS = [
  ['ai_written', 'AI Written', 'ai'],
  ['ai_paraphrased', 'AI Paraphrased', 'paraphrased'],
  ['human', 'Human Written', 'human'],
];

export default function ClassificationResults({ results }) {
  if (!results) return null;

  return (
    <div className="glass-panel results-container">
      <h3>Classification Results</h3>
      <p className="result-note">Model scores are not calibrated confidence estimates.</p>

      {SCORE_ROWS.map(([key, label, className]) => (
        <div className="bar-group" key={key}>
          <div className="bar-header">
            <span>{label}</span>
            <span>{(results[key] * 100).toFixed(1)}%</span>
          </div>
          <div className="bar-track">
            <div
              className={`bar-fill ${className}`}
              style={{ width: `${results[key] * 100}%` }}
            />
          </div>
        </div>
      ))}

      {!results.truncated && <p className="result-note">{results.analyzed_tokens} content tokens analyzed across {results.window_count || 1} windows.</p>}
      {results.truncated && (
        <p className="warning-message" role="status">
          Only the first {results.analyzed_tokens} tokens were analyzed.
        </p>
      )}
      {results.warnings?.includes('insufficient_text') && (
        <p className="warning-message" role="status">
          This sample is very short; its model scores may be unreliable.
        </p>
      )}

      {results.observed_signals?.length > 0 && (
        <div className="features-list">
          <h4>Separate heuristic observations</h4>
          <p className="result-note">These indicators do not explain or alter the model score.</p>
          <ul>
            {results.observed_signals.map((signal) => <li key={signal}>{signal}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
