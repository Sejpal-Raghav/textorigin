'use client';

import { useState } from 'react';

import { humanizeText } from '../lib/api';

function Metric({ label, before, after }) {
  if (before === undefined || after === undefined) return null;
  const difference = after - before;
  return (
    <div className="metric-card">
      <div className="label">{label}</div>
      <div className="value">{after.toFixed(2)}</div>
      <div className={difference > 0 ? 'positive' : 'negative'}>
        {difference > 0 ? '+' : ''}{difference.toFixed(2)}
      </div>
    </div>
  );
}

export default function HumanizePanel({ originalText, onHumanizeComplete }) {
  const [loading, setLoading] = useState(false);
  const [useLlm, setUseLlm] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  async function handleHumanize() {
    setLoading(true);
    setError('');
    try {
      const data = await humanizeText(originalText, useLlm);
      setResult(data);
      if (data.status !== 'unchanged') onHumanizeComplete?.(data.humanized_text);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="humanize-panel">
      <h3>AI to Human Converter</h3>
      <p className="result-note">Rewrite while enforcing a server-side semantic similarity limit.</p>

      <div className="toggle-container">
        <input
          type="checkbox"
          id="llmToggle"
          checked={useLlm}
          onChange={(event) => setUseLlm(event.target.checked)}
        />
        <label htmlFor="llmToggle">Polish with Local LLM (Ollama)</label>
      </div>

      <button className="primary-btn" onClick={handleHumanize} disabled={loading || !originalText.trim()}>
        {loading ? <span className="spinner" aria-label="Humanizing" /> : 'Humanize Text'}
      </button>
      {error && <p className="error-message" role="alert">{error}</p>}

      {result && (
        <div className="result-section">
          <h4>Result</h4>
          <div className="metric-grid">
            <Metric label="Perplexity" before={result.original_metrics.perplexity} after={result.post_heuristic_metrics.perplexity} />
            <Metric label="Burstiness" before={result.original_metrics.burstiness} after={result.post_heuristic_metrics.burstiness} />
            <Metric label="AI phrases / 100 words" before={result.original_metrics.ai_phrase_rate} after={result.post_heuristic_metrics.ai_phrase_rate} />
          </div>

          <div className="diff-view">{result.humanized_text}</div>
          <div className="pipeline-stats">
            <strong>Pipeline status: {result.status.replaceAll('_', ' ')}</strong>
            <ul>
              <li>Heuristic similarity: {result.heuristic_similarity.toFixed(3)}</li>
              {result.llm_similarity !== null && <li>LLM similarity: {result.llm_similarity.toFixed(3)}</li>}
              <li>Required similarity: {result.similarity_threshold.toFixed(2)}</li>
              {result.warnings.map((warning) => <li className="warning-message" key={warning}>{warning.replaceAll('_', ' ')}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
