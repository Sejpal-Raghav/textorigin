'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelJob, getJob, submitJob } from './api';

export default function useJob(inputVersion) {
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const generation = useRef(0);
  const active = useRef(null);
  const submitting = useRef(false);
  const cancelPending = useRef(null);

  const reset = useCallback(() => {
    generation.current += 1;
    if (active.current) cancelJob(active.current).catch(() => {});
    active.current = null;
    submitting.current = false; cancelPending.current = null;
    setJob(null); setResult(null); setError(''); setBusy(false);
  }, []);

  useEffect(() => {
    reset();
    return () => {
      generation.current += 1;
      if (active.current) cancelJob(active.current).catch(() => {});
      active.current = null;
    };
  }, [inputVersion, reset]);

  async function run(body) {
    if (active.current || submitting.current || busy) return;
    submitting.current = true; cancelPending.current = null;
    const version = ++generation.current;
    setBusy(true); setError(''); setResult(null);
    let id = null;
    try {
      let status = await submitJob(body);
      id = status.id;
      if (version !== generation.current) {
        await cancelJob(id);
        return;
      }
      active.current = id;
      if (cancelPending.current === version) status = await cancelJob(id);
      while (version === generation.current) {
        setJob(cancelPending.current === version && status.state === "running" ? { ...status, state:"cancel_requested", stage:"cancelling" } : status);
        if (status.state === 'completed') { setResult(status.result); break; }
        if (status.state === 'failed') throw new Error(status.error?.detail || 'Operation failed.');
        if (status.state === 'cancelled') break;
        await new Promise((resolve) => setTimeout(resolve, 500));
        if (version !== generation.current) break;
        status = await getJob(id);
      }
    } catch (requestError) {
      if (id) cancelJob(id).catch(() => {});
      if (version === generation.current) setError(requestError.message);
    } finally {
      if (version === generation.current) { active.current = null; submitting.current = false; cancelPending.current = null; setBusy(false); }
    }
  }

  async function cancel() {
    if (!active.current) {
      if (submitting.current) {
        cancelPending.current = generation.current;
        setJob({ state:"cancel_requested", stage:"cancelling_submission", completed_units:0, total_units:null });
      }
      return;
    }
    try { const status = await cancelJob(active.current); cancelPending.current = generation.current; setJob(status); }
    catch (requestError) { setError(requestError.message); }
  }

  return { job, result, error, busy, run, cancel, reset };
}
