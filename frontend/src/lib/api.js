const API_BASE_URL = (
  process.env.NEXT_PUBLIC_TEXTORIGIN_API_URL || 'http://localhost:8000'
).replace(/\/$/, '');

export const MAX_TEXT_CHARS = Number(
  process.env.NEXT_PUBLIC_TEXTORIGIN_MAX_TEXT_CHARS || 10_000
);

function errorMessage(payload, fallback) {
  if (payload?.error?.detail) return payload.error.detail;
  if (typeof payload?.detail === 'string') return payload.detail;
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item.msg).join('; ');
  }
  return fallback;
}

async function post(path, body, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(errorMessage(payload, `Request failed with status ${response.status}`));
    }
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('The request timed out. The local models may still be loading.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export function classifyText(text) {
  return post('/classify', { text, include_signals: true }, 60_000);
}

export function humanizeText(text, useLlm) {
  return post('/humanize', { text, use_llm: useLlm }, 75_000);
}
