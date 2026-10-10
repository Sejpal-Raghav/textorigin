'use client';

import { useState } from 'react';

function Metric({ label, before, after }) {
  if (before === undefined || after === undefined) return null;
  const difference = after - before;
  return <div className="metric-card"><div className="label">{label}</div>
    <div className="value">{after.toFixed(2)}</div>
    <div>{difference > 0 ? '+' : ''}{difference.toFixed(2)}</div></div>;
}

export default function HumanizePanel({ result }) {
  const [notice, setNotice] = useState('');
  if (!result) return null;
  const metrics = result.final_metrics;
  return <section className="humanize-panel" aria-label="Rewrite result">
    <h3>Accepted rewrite</h3>
    <p className="result-note">Controlled edits with per-window meaning checks. These metrics describe the accepted text.</p>
    {metrics && <div className="metric-grid">
      <Metric label="Perplexity" before={result.original_metrics.perplexity} after={metrics.perplexity} />
      <Metric label="Burstiness" before={result.original_metrics.burstiness} after={metrics.burstiness} />
      <Metric label="AI phrases / 100 words" before={result.original_metrics.ai_phrase_rate} after={metrics.ai_phrase_rate} />
    </div>}
    <div className="diff-view">{result.humanized_text}</div>
    <div className="result-actions">
      <button type="button" onClick={async () => {
        try { await navigator.clipboard.writeText(result.humanized_text); setNotice('Copied.'); }
        catch { setNotice('Copy failed; select the text to copy manually.'); }
      }}>Copy result</button>
      <button type="button" onClick={() => {
        const url = URL.createObjectURL(new Blob([result.humanized_text], { type: 'text/plain;charset=utf-8' }));
        const link = document.createElement('a'); link.href = url; link.download = 'textorigin-humanized.txt';
        link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      }}>Download .txt</button>
    </div>
    <p role="status">{notice}</p>
    <div className="pipeline-stats">
      <strong>Status: {result.status.replaceAll('_', ' ')}</strong>
      <p>{result.chunks_processed} chunks processed · Minimum required similarity {result.similarity_threshold.toFixed(2)}</p>
      <ul>{result.warnings.map((warning, index) => <li className="warning-message" key={index}>{warning.replaceAll('_', ' ')}</li>)}</ul>
    </div>
  </section>;
}
