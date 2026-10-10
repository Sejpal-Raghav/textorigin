const API_BASE_URL = (process.env.NEXT_PUBLIC_TEXTORIGIN_API_URL || 'http://localhost:8000').replace(/\/$/, '');

async function request(path, body, timeoutMs = 15_000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(API_BASE_URL + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
      cache: 'no-store',
    });
    const payload = await response.json();
    if (!response.ok && !(path === '/ready' && response.status === 503)) {
      throw new Error(payload?.error?.detail || payload?.detail || 'Local request failed.');
    }
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The local server did not respond. Check that it is running.');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export const getReadiness = () => request('/ready');
export const submitJob = (body) => request('/jobs', body);
export const getJob = (id) => request('/jobs/' + encodeURIComponent(id));
export const cancelJob = (id) => request('/jobs/' + encodeURIComponent(id) + '/cancel', {});
// Compatibility helpers for consumers of the previous synchronous API.
export const classifyText = (text) => request('/classify', { text, include_signals: true }, 60_000);
export const humanizeText = (text, options) => request('/humanize', {
  text, use_llm: options.useDeepSeek, style: options.style, strength: options.strength,
  ...(options.apiKey ? { deepseek_api_key: options.apiKey } : {}),
}, 75_000);
