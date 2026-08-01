'use client';

import { useState } from 'react';
import '../styles/components.css';

function ClassificationResults({ results }) {
  if (!results) return null;

  return (
    <div className="glass-panel results-container">
      <h3>Classification Results</h3>
      
      <div className="bar-group">
        <div className="bar-header">
          <span>AI Written</span>
          <span>{(results.ai_written * 100).toFixed(1)}%</span>
        </div>
        <div className="bar-track">
          <div className="bar-fill ai" style={{ width: `${results.ai_written * 100}%` }}></div>
        </div>
      </div>

      <div className="bar-group">
        <div className="bar-header">
          <span>AI Paraphrased</span>
          <span>{(results.ai_paraphrased * 100).toFixed(1)}%</span>
        </div>
        <div className="bar-track">
          <div className="bar-fill paraphrased" style={{ width: `${results.ai_paraphrased * 100}%` }}></div>
        </div>
      </div>

      <div className="bar-group">
        <div className="bar-header">
          <span>Human Written</span>
          <span>{(results.human * 100).toFixed(1)}%</span>
        </div>
        <div className="bar-track">
          <div className="bar-fill human" style={{ width: `${results.human * 100}%` }}></div>
        </div>
      </div>

      {results.top_features && results.top_features.length > 0 && (
        <div className="features-list">
          <h4>Key Signals Detected</h4>
          <ul>
            {results.top_features.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function HumanizePanel({ originalText, onHumanizeComplete }) {
  const [loading, setLoading] = useState(false);
  const [useLlm, setUseLlm] = useState(true);
  const [result, setResult] = useState(null);

  const handleHumanize = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/humanize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: originalText, use_llm: useLlm, similarity_threshold: 0.85 })
      });
      const data = await res.json();
      setResult(data);
      if (onHumanizeComplete) {
        onHumanizeComplete(data.humanized_text);
      }
    } catch (e) {
      console.error(e);
      alert("Error humanizing text. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const renderMetric = (label, before, after) => {
    if (before === undefined || after === undefined) return null;
    const diff = after - before;
    const isPositive = diff > 0;
    return (
      <div className="metric-card">
        <div className="label">{label}</div>
        <div className="value">{after.toFixed(2)}</div>
        <div style={{ fontSize: '0.85rem', marginTop: '4px' }} className={isPositive ? 'positive' : 'negative'}>
          {isPositive ? '+' : ''}{diff.toFixed(2)}
        </div>
      </div>
    );
  };

  return (
    <div className="humanize-panel">
      <h3>AI to Human Converter</h3>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '16px' }}>
        Rewrite the text to mimic human statistical signatures.
      </p>

      <div className="toggle-container">
        <input 
          type="checkbox" 
          id="llmToggle" 
          checked={useLlm} 
          onChange={(e) => setUseLlm(e.target.checked)} 
        />
        <label htmlFor="llmToggle">Polish with Local LLM (Ollama)</label>
      </div>

      <button className="primary-btn" onClick={handleHumanize} disabled={loading || !originalText.trim()}>
        {loading ? <div className="spinner"></div> : "Humanize Text"}
      </button>

      {result && (
        <div style={{ marginTop: '24px' }}>
          <h4>Result</h4>
          <div className="metric-grid">
            {renderMetric("Perplexity", result.original_metrics.perplexity, result.post_heuristic_metrics.perplexity)}
            {renderMetric("Burstiness", result.original_metrics.burstiness, result.post_heuristic_metrics.burstiness)}
            {renderMetric("AI Phrase Ratio", result.original_metrics.ai_phrase_ratio, result.post_heuristic_metrics.ai_phrase_ratio)}
          </div>
          
          <div className="diff-view">
            {result.humanized_text}
          </div>
          
          <div style={{ marginTop: '16px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            <strong>Pipeline Stats:</strong>
            <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
              <li>Heuristic Similarity: {result.similarity_before?.toFixed(3)}</li>
              {result.used_llm && <li>LLM Similarity: {result.similarity_after?.toFixed(3)}</li>}
              {result.polish_failed && <li style={{ color: 'var(--accent-red)' }}>LLM Polish failed (fell back to heuristic).</li>}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleClassify = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      setResults(data);
    } catch (e) {
      console.error(e);
      alert("Error classifying text. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <header className="header">
        <h1>TextOrigin</h1>
        <p>Advanced AI vs Human Text Classifier</p>
      </header>

      <div className="grid-layout">
        <div className="editor-wrapper glass-panel">
          <h3>Input Text</h3>
          <textarea
            className="textarea"
            placeholder="Paste your text here to analyze..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button 
            className="primary-btn" 
            onClick={handleClassify}
            disabled={loading || !text.trim()}
          >
            {loading ? <div className="spinner"></div> : "Analyze Text"}
          </button>
        </div>

        <div>
          <ClassificationResults results={results} />
          <HumanizePanel 
            originalText={text} 
            onHumanizeComplete={(newText) => {
              setText(newText);
              setResults(null);
            }} 
          />
        </div>
      </div>
    </main>
  );
}
