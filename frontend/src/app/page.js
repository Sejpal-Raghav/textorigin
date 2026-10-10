'use client';

import { useEffect, useState } from 'react';
import ClassificationResults from '../components/ClassificationResults';
import HumanizePanel from '../components/HumanizePanel';
import { getReadiness } from '../lib/api';
import useJob from '../lib/useJob';
import '../styles/components.css';

const LABELS = [['human', 'Human written'], ['ai_written', 'AI written'], ['ai_paraphrased', 'AI paraphrased']];

export default function Home() {
  const [text, setText] = useState('');
  const [style, setStyle] = useState('natural');
  const [strength, setStrength] = useState('light');
  const [useDeepSeek, setUseDeepSeek] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [readiness, setReadiness] = useState(null);
  const [serverError, setServerError] = useState('');
  const state = useJob(JSON.stringify([text, style, strength, useDeepSeek, apiKey]));
  async function refresh() {
    try { setReadiness(await getReadiness()); setServerError(''); }
    catch (error) { setServerError(error.message); setReadiness(null); }
  }
  useEffect(() => { let current = true; getReadiness().then((value) => { if (current) setReadiness(value); })
    .catch((error) => { if (current) setServerError(error.message); }); return () => { current = false; }; }, []);
  const classifierReady = Boolean(readiness?.classifier?.ready);
  const disable = state.busy || !text.trim();
  function run(operation) {
    state.run({ operation, text, style, strength, use_llm: operation !== 'classify' && useDeepSeek,
      include_signals: operation === 'classify', ...(apiKey && operation !== 'classify' && useDeepSeek ? { deepseek_api_key: apiKey } : {}) });
  }
  const comparison = state.result?.operation === 'compare' ? state.result : null;
  const classification = state.job?.operation === 'classify' ? state.result : null;
  const rewrite = comparison?.rewrite || (state.job?.operation === 'humanize' ? state.result : null);

  return <main className="container">
    <header className="header"><h1>TextOrigin</h1><p>Text-origin analysis and controlled English editing</p></header>
    <div className="capability-status" role="status">
      {serverError || (readiness ? (classifierReady ? 'Classifier artifact available. Local rewriting is independent of classification.' : 'Classifier unavailable. You can still rewrite when the local similarity and feature models are provisioned.') : 'Checking local capabilities…')}
      <button type="button" onClick={refresh}>Refresh capabilities</button>
    </div>
    <div className="grid-layout">
      <section className="editor-wrapper glass-panel">
        <label htmlFor="sourceText"><h2>Input text</h2></label>
        <textarea id="sourceText" className="textarea" placeholder="Paste English text to analyze or edit…" value={text} onChange={(event) => setText(event.target.value)} />
        <div className="character-count">{text.trim() ? text.trim().split(/\s+/).length.toLocaleString() : 0} words · {text.length.toLocaleString()} characters</div>
        <div className="humanizer-controls">
          <label>Style<select value={style} disabled={state.busy} onChange={(event) => setStyle(event.target.value)}>
            <option value="natural">Natural</option><option value="formal">Formal</option><option value="casual">Casual</option></select></label>
          <label>Editing strength<select value={strength} disabled={state.busy} onChange={(event) => setStrength(event.target.value)}>
            <option value="light">Light</option><option value="medium">Medium</option></select></label>
        </div>
        <div className="toggle-container"><input id="deepSeekToggle" type="checkbox" checked={useDeepSeek} disabled={state.busy} onChange={(event) => setUseDeepSeek(event.target.checked)} />
          <label htmlFor="deepSeekToggle">Add minor DeepSeek polish</label></div>
        {useDeepSeek && <div className="credential-panel">
          <label htmlFor="deepSeekKey">DeepSeek API key (optional if configured by the server owner)</label>
          <input id="deepSeekKey" type="password" autoComplete="off" value={apiKey} disabled={state.busy} onChange={(event) => setApiKey(event.target.value)} />
          <button type="button" disabled={state.busy} onClick={() => setApiKey('')}>Clear key</button>
          <p className="privacy-note">Polishing sends the editable passage to DeepSeek. Your key stays in this page’s memory and is sent only with a polishing request.</p>
        </div>}
        <div className="operation-actions">
          <button className="primary-btn" disabled={disable || !classifierReady} onClick={() => run('classify')}>Analyze text</button>
          <button className="primary-btn" disabled={disable} onClick={() => run('humanize')}>Rewrite text</button>
          <button className="primary-btn" disabled={disable || !classifierReady} onClick={() => run('compare')}>Compare scores</button>
        </div>
        {state.job && <div role="status" aria-live="polite">
          <span>{state.job.stage.replaceAll('_', ' ')}</span>{state.job.total_units != null && <span> · {state.job.completed_units}/{state.job.total_units}</span>}
          {state.job.total_units > 0 && <progress max={state.job.total_units} value={state.job.completed_units} aria-label="Operation progress" />}
        </div>}
        {state.busy && <button type="button" onClick={state.cancel} disabled={state.job?.state === 'cancel_requested'}>Cancel operation</button>}
        {state.error && <p className="error-message" role="alert">{state.error}</p>}
        {!state.busy && (state.result || state.error || state.job) && <button type="button" onClick={state.reset}>Clear result</button>}
      </section>
      <section aria-label="Results">
        <ClassificationResults results={classification} />
        {comparison && <div className="glass-panel"><h2>Compare model scores</h2>
          <p className="result-note">Score changes do not prove authorship or writing quality.</p>
          <table><thead><tr><th>Class</th><th>Original</th><th>Revised</th><th>Change (pp)</th></tr></thead><tbody>
            {LABELS.map(([key, label]) => <tr key={key}><th scope="row">{label}</th><td>{(comparison.original_scores[key] * 100).toFixed(1)}%</td>
              <td>{(comparison.revised_scores[key] * 100).toFixed(1)}%</td><td>{comparison.delta_percentage_points[key].toFixed(1)}</td></tr>)}
          </tbody></table>
          <p>{comparison.original_analysis.analyzed_tokens} original / {comparison.revised_analysis.analyzed_tokens} revised content tokens analyzed.</p>
        </div>}
        <HumanizePanel key={state.job?.id || 'empty'} result={rewrite} />
      </section>
    </div>
  </main>;
}
