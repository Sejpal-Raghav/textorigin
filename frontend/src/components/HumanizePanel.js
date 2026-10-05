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

export default function HumanizePanel({ originalText }) {
  const [loading, setLoading] = useState(false);
  const [useDeepSeek, setUseDeepSeek] = useState(false);
  const [style, setStyle] = useState('natural');
  const [strength, setStrength] = useState('light');
  const [apiKey, setApiKey] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  async function handleHumanize() {
    setLoading(true);
    setError('');
    try {
      const data = await humanizeText(originalText, {
        useDeepSeek,
        style,
        strength,
        apiKey,
      });
      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="humanize-panel">
      <h3>Humanizer</h3>
      <p className="result-note">Make controlled English edits while preserving meaning and protected content.</p>

      <div className="humanizer-controls">
        <label>
          Style
          <select value={style} onChange={(event) => setStyle(event.target.value)}>
            <option value="natural">Natural</option>
            <option value="formal">Formal</option>
            <option value="casual">Casual</option>
          </select>
        </label>
        <label>
          Editing strength
          <select value={strength} onChange={(event) => setStrength(event.target.value)}>
            <option value="light">Light</option>
            <option value="medium">Medium</option>
          </select>
        </label>
      </div>

      <div className="toggle-container">
        <input
          type="checkbox"
          id="deepSeekToggle"
          checked={useDeepSeek}
          onChange={(event) => setUseDeepSeek(event.target.checked)}
        />
        <label htmlFor="deepSeekToggle">Add minor DeepSeek polish</label>
      </div>

      {useDeepSeek && (
        <div className="credential-panel">
          <label htmlFor="deepSeekKey">DeepSeek API key (optional when the server owner configured one)</label>
          <input
            id="deepSeekKey"
            type="password"
            autoComplete="off"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            placeholder="sk-…"
          />
          <p className="privacy-note" role="note">
            With polishing enabled, the editable text is sent to DeepSeek. This key stays in this page&apos;s memory and is sent only with this request.
          </p>
        </div>
      )}

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
          <div className="result-actions">
            <button type="button" onClick={() => navigator.clipboard.writeText(result.humanized_text)}>Copy result</button>
            <button
              type="button"
              onClick={() => {
                const url = URL.createObjectURL(new Blob([result.humanized_text], { type: 'text/plain;charset=utf-8' }));
                const link = document.createElement('a');
                link.href = url;
                link.download = 'textorigin-humanized.txt';
                link.click();
                URL.revokeObjectURL(url);
              }}
            >
              Download .txt
            </button>
          </div>
          <div className="pipeline-stats">
            <strong>Pipeline status: {result.status.replaceAll('_', ' ')}</strong>
            <ul>
              <li>Style: {result.style}; strength: {result.strength}</li>
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
