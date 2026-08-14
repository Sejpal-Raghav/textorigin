'use client';

import { useState } from 'react';

import ClassificationResults from '../components/ClassificationResults';
import HumanizePanel from '../components/HumanizePanel';
import { classifyText, MAX_TEXT_CHARS } from '../lib/api';
import '../styles/components.css';

export default function Home() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  async function handleClassify() {
    if (!text.trim()) return;
    setLoading(true);
    setError('');
    try {
      setResults(await classifyText(text));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <header className="header">
        <h1>TextOrigin</h1>
        <p>AI and human text-origin analysis</p>
      </header>

      <div className="grid-layout">
        <div className="editor-wrapper glass-panel">
          <h3>Input Text</h3>
          <textarea
            className="textarea"
            placeholder="Paste your text here to analyze..."
            value={text}
            maxLength={MAX_TEXT_CHARS}
            onChange={(event) => setText(event.target.value)}
          />
          <div className="character-count">
            {text.length.toLocaleString()} / {MAX_TEXT_CHARS.toLocaleString()}
          </div>
          <button className="primary-btn" onClick={handleClassify} disabled={loading || !text.trim()}>
            {loading ? <span className="spinner" aria-label="Analyzing" /> : 'Analyze Text'}
          </button>
          {error && <p className="error-message" role="alert">{error}</p>}
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
