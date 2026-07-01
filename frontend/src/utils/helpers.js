import { MICRO_STEPS, FLOW_STEPS, LAST_QUERY_STORAGE_KEY } from './constants';

export function isObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value);
}

export function displayText(value, fallback = 'Text unavailable') {
  const text = String(value ?? '').trim();
  return text || fallback;
}

export function stripMarkdown(value) {
  return displayText(value, 'No reading available')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/^#+\s+/gm, '')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

export function compactSummary(value, max = 86) {
  const text = stripMarkdown(value);
  return text.length > max ? `${text.slice(0, max).trim()}...` : text;
}

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[ch]));
}

export function htmlFromText(value) {
  const text = String(value ?? '').trim();
  if (!text) return '';
  return text
    .split('\n')
    .filter(Boolean)
    .map((line) => `<p>${escapeHtml(line)}</p>`)
    .join('');
}

export function displayLog(value) {
  return String(value ?? '').replace(/[^\x00-\x7F]+/g, '[original-language text]');
}

export function clampPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, Math.round(n * 100)));
}

export function percentText(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'Not rated';
  return `${Math.round(n * 100)}%`;
}

export function compactNumber(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '0';
  return new Intl.NumberFormat('en', { notation: n > 9999 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(n);
}

export function durationText(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 'No run yet';
  if (n < 60) return `${Math.round(n)} sec`;
  return `${Math.floor(n / 60)} min ${Math.round(n % 60)} sec`;
}

export function timeText(value) {
  if (!value) return 'No timestamp';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'No timestamp' : date.toLocaleString('en-US', { hour12: false });
}

export function msText(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 'No timing';
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60000) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)} sec`;
  return `${Math.floor(n / 60000)} min ${Math.round((n % 60000) / 1000)} sec`;
}

export function moneyText(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'Not metered';
  return `$${n.toFixed(n < 0.01 ? 4 : 2)}`;
}

export function urlDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return 'source';
  }
}

export function sourceTitle(source, index) {
  return displayText(source?.title, `Source ${index + 1} from ${urlDomain(source?.url)}`);
}

export function reportSeedHtml(output) {
  const summary = output?.synthesis?.summary || 'No report draft is available yet. Run an analysis or generate a report to start reviewing.';
  const insights = output?.synthesis?.top_insights || [];
  const tensions = output?.synthesis?.key_tensions || [];
  return `
    <h2>Executive Brief</h2>
    ${htmlFromText(summary)}
    <h2>Priority Insights</h2>
    ${insights.map((item) => `<h3>${escapeHtml(displayText(item.insight, 'Insight unavailable'))}</h3><p>${escapeHtml(displayText(item.basis, 'Evidence basis unavailable'))}</p>`).join('') || '<p>No insights available.</p>'}
    <h2>Open Risks</h2>
    ${tensions.map((item) => `<p><strong>${escapeHtml(displayText(item.tension, 'Risk unavailable'))}</strong><br>${escapeHtml(displayText(item.significance, 'Significance unavailable'))}</p>`).join('') || '<p>No unresolved risks available.</p>'}
  `;
}

export const SENSITIVE_INPUT_MESSAGE =
  'Your input contains blocked terms, so the report cannot be generated. Please revise the topic and try again.';

export function isSensitiveInputError(error) {
  return error?.code === 'sensitive_input' || error?.payload?.error_code === 'sensitive_input';
}

export function showSensitiveInputModal() {
  import('antd').then(({ Modal }) => {
    Modal.warning({
      title: 'Request Blocked',
      content: SENSITIVE_INPUT_MESSAGE,
      okText: 'OK',
      centered: true
    });
  });
}

export async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { success: false, error: text || 'Request failed' };
  }
  if (!response.ok) {
    const error = new Error(payload.message || payload.error || `Request failed with ${response.status}`);
    error.code = payload.error_code || '';
    error.payload = payload;
    throw error;
  }
  return payload;
}

export function downloadBlob(content, filename, type = 'text/html') {
  const blob = new Blob([content], { type });
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(href);
}

export function flowProgressMeta(progress) {
  const normalized = Math.max(0, Math.min(100, Number(progress) || 0));
  const microRaw = normalized >= 100 ? MICRO_STEPS.length - 1 : Math.floor((normalized / 100) * MICRO_STEPS.length);
  const microIndex = Math.max(0, Math.min(MICRO_STEPS.length - 1, microRaw));
  const micro = MICRO_STEPS[microIndex] || MICRO_STEPS[0];
  const stage = FLOW_STEPS[micro.stageIndex] || FLOW_STEPS[0];
  const stageStart = MICRO_STEPS.findIndex((item) => item.stageIndex === micro.stageIndex);
  const stageCount = stage.micro.length || 1;
  const localIndex = Math.max(0, microIndex - stageStart);
  return {
    normalized,
    micro,
    microIndex,
    activeIndex: micro.stageIndex,
    stage,
    stagePercent: Math.round(((localIndex + 1) / stageCount) * 100)
  };
}

export function langSmithProjectUrl(observability) {
  const endpoint = String(observability?.endpoint || 'https://smith.langchain.com').replace(/\/$/, '');
  const project = observability?.project;
  if (!project) return 'https://smith.langchain.com/';
  return `${endpoint.replace('api.smith.langchain.com', 'smith.langchain.com')}/o/default/projects/p/${encodeURIComponent(project)}`;
}

export function traceStatus(run) {
  if (run?.error) return 'error';
  const status = String(run?.status || '').toLowerCase();
  if (status.includes('error') || status.includes('fail')) return 'error';
  if (status.includes('run') || status.includes('pending')) return 'running';
  return 'success';
}

export function traceMetricRows(observabilityTrace) {
  const summary = observabilityTrace?.summary || {};
  return [
    { key: 'trace_count', label: 'Traces', value: compactNumber(summary.trace_count) },
    { key: 'run_count', label: 'Steps', value: compactNumber(summary.run_count) },
    { key: 'avg_duration_ms', label: 'Avg time', value: msText(summary.avg_duration_ms) },
    { key: 'error_count', label: 'Errors', value: compactNumber(summary.error_count) },
    { key: 'total_tokens', label: 'Tokens', value: compactNumber(summary.total_tokens) },
    { key: 'total_cost', label: 'Cost', value: moneyText(summary.total_cost) }
  ];
}

export function traceTimelineChartData(observabilityTrace) {
  return (observabilityTrace?.timeline || []).slice().reverse().map((run, index) => ({
    name: `T${index + 1}`,
    duration: Math.round(Number(run.duration_ms) || 0),
    steps: Array.isArray(run.children) ? run.children.length : 0,
    errors: traceStatus(run) === 'error' ? 1 : 0
  }));
}

export function traceChildrenByType(children = []) {
  const counts = {};
  children.forEach((child) => {
    const key = child.type || 'unknown';
    counts[key] = (counts[key] || 0) + 1;
  });
  return Object.entries(counts).map(([name, value]) => ({ name, value }));
}

export function readLastQuery() {
  try {
    return window.localStorage.getItem(LAST_QUERY_STORAGE_KEY) || '';
  } catch {
    return '';
  }
}

export function persistLastQuery(value) {
  try {
    window.localStorage.setItem(LAST_QUERY_STORAGE_KEY, String(value ?? ''));
  } catch {
    return undefined;
  }
}
