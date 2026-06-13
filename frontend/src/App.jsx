import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Badge,
  Button,
  Collapse,
  ConfigProvider,
  Drawer,
  Empty,
  Form,
  Input,
  Modal,
  Progress,
  Radio,
  Select,
  Slider,
  Space,
  Switch,
  Table,
  Tag,
  Timeline,
  Tooltip,
  message
} from 'antd';
import {
  BookOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloudDownloadOutlined,
  CommentOutlined,
  ControlOutlined,
  EditOutlined,
  ExperimentOutlined,
  FilePdfOutlined,
  FileTextOutlined,
  HighlightOutlined,
  LinkOutlined,
  LoadingOutlined,
  MessageOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  RadarChartOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
  SearchOutlined,
  SendOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  WarningOutlined,
  BgColorsOutlined
} from '@ant-design/icons';
import { motion, AnimatePresence } from 'framer-motion';
import { EditorContent, useEditor } from '@tiptap/react';
import { BubbleMenu } from '@tiptap/react/menus';
import StarterKit from '@tiptap/starter-kit';
import Highlight from '@tiptap/extension-highlight';
import Underline from '@tiptap/extension-underline';
import Link from '@tiptap/extension-link';
import Placeholder from '@tiptap/extension-placeholder';
import { TextStyle } from '@tiptap/extension-text-style';
import { Color } from '@tiptap/extension-color';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis
} from 'recharts';
import '@fontsource/nunito-sans/400.css';
import '@fontsource/nunito-sans/500.css';
import '@fontsource/nunito-sans/600.css';
import '@fontsource/inter/500.css';
import '@fontsource/instrument-serif/400.css';


const THEME_TOKENS = {
  green: {
    label: 'HKU Green',
    primary: '#115e59',
    primaryDark: '#024638',
    primaryMid: '#2d8b73',
    primarySoft: '#bce6d5',
    trail: 'rgba(17,94,89,.12)',
    accent: '#0b6b58'
  },
  blue: {
    label: 'Blue Tech',
    primary: '#2563eb',
    primaryDark: '#0f2f66',
    primaryMid: '#3b82f6',
    primarySoft: '#bfdbfe',
    trail: 'rgba(37,99,235,.13)',
    accent: '#06b6d4'
  }
};

const NAV_ITEMS = [
  { key: 'command', label: 'Home', icon: <ThunderboltOutlined /> },
  { key: 'intelligence', label: 'Readout', icon: <RadarChartOutlined /> },
  { key: 'evidence', label: 'Proof', icon: <SearchOutlined /> },
  { key: 'review', label: 'Edit', icon: <EditOutlined /> },
  { key: 'control', label: 'Monitor', icon: <ControlOutlined /> }
];

const CONFIG_GROUPS = [
  {
    title: 'Foundation Models',
    description: 'Model providers used by analysis, evidence retrieval, and report writing.',
    fields: [
      ['QUERY_ENGINE_API_KEY', 'Evidence model key', 'password'],
      ['QUERY_ENGINE_BASE_URL', 'Evidence model URL', 'text'],
      ['QUERY_ENGINE_MODEL_NAME', 'Evidence model name', 'text'],
      ['MEDIA_ENGINE_API_KEY', 'Media model key', 'password'],
      ['MEDIA_ENGINE_BASE_URL', 'Media model URL', 'text'],
      ['MEDIA_ENGINE_MODEL_NAME', 'Media model name', 'text'],
      ['REPORT_ENGINE_API_KEY', 'Report model key', 'password'],
      ['REPORT_ENGINE_BASE_URL', 'Report model URL', 'text'],
      ['REPORT_ENGINE_MODEL_NAME', 'Report model name', 'text']
    ]
  },
  {
    title: 'Search and Retrieval',
    description: 'External search providers used to collect public evidence.',
    fields: [
      ['SEARCH_TOOL_TYPE', 'Search provider', 'select'],
      ['TAVILY_API_KEY', 'Tavily key', 'password'],
      ['BOCHA_WEB_SEARCH_API_KEY', 'Bocha key', 'password'],
      ['ANSPIRE_API_KEY', 'Anspire key', 'password']
    ]
  },
  {
    title: 'Trace Quality',
    description: 'LangSmith tracing for model calls, timing, errors, and review quality.',
    fields: [
      ['LANGSMITH_TRACING', 'Tracing enabled', 'boolean'],
      ['LANGSMITH_API_KEY', 'LangSmith key', 'password'],
      ['LANGSMITH_ENDPOINT', 'LangSmith endpoint', 'text'],
      ['LANGSMITH_PROJECT', 'LangSmith project', 'text']
    ]
  }
];

const FLOW_STEPS = [
  { id: 'brief', label: 'Brief', sub: 'Topic', micro: ['Intent', 'Scope', 'Context'] },
  { id: 'collect', label: 'Collect', sub: 'Sources', micro: ['Search', 'Rank', 'Dedup', 'Trust'] },
  { id: 'map', label: 'Map', sub: 'Patterns', micro: ['Stance', 'Sentiment', 'Coverage', 'Divergence'] },
  { id: 'reason', label: 'Reason', sub: 'Tensions', micro: ['Debate', 'Consensus', 'Dissent'] },
  { id: 'verify', label: 'Verify', sub: 'Claims', micro: ['Facts', 'Opinions', 'Bias'] },
  { id: 'write', label: 'Write', sub: 'Report', micro: ['Outline', 'Draft', 'Review', 'Export'] }
];

const MICRO_STEPS = FLOW_STEPS.flatMap((step, stageIndex) => step.micro.map((name, microIndex) => ({
  id: `${step.id}-${microIndex}`,
  name,
  stageId: step.id,
  stageIndex,
  stageLabel: step.label
})));

function flowProgressMeta(progress) {
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


const TRACE_COLORS = {
  chain: '#2563eb',
  llm: '#0891b2',
  tool: '#7c3aed',
  retriever: '#16a34a',
  parser: '#f59e0b',
  unknown: '#64748b',
  'local step': '#2563eb'
};

const STANCE_COLORS = {
  support: '#024638',
  oppose: '#d9463e',
  neutral: '#4f8f7b',
  official: '#0b6b58',
  background: '#8bb9a8',
  unknown: '#8a978f'
};

function isObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value);
}

function displayText(value, fallback = 'Text unavailable') {
  const text = String(value ?? '').trim();
  return text || fallback;
}


function stripMarkdown(value) {
  return displayText(value, 'No reading available')
    .replace(/```[\\s\\S]*?```/g, ' ')
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

function compactSummary(value, max = 86) {
  const text = stripMarkdown(value);
  return text.length > max ? `${text.slice(0, max).trim()}...` : text;
}

function MarkdownText({ value }) {
  const escaped = escapeHtml(displayText(value, 'No reading available'));
  const html = escaped
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/_([^_]+)_/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
  return <span className="markdown-text" dangerouslySetInnerHTML={{ __html: html }} />;
}

function displayLog(value) {
  return String(value ?? '').replace(/[^\x00-\x7F]+/g, '[original-language text]');
}

function clampPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, Math.round(n * 100)));
}

function percentText(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'Not rated';
  return `${Math.round(n * 100)}%`;
}

function compactNumber(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '0';
  return new Intl.NumberFormat('en', { notation: n > 9999 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(n);
}

function durationText(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 'No run yet';
  if (n < 60) return `${Math.round(n)} sec`;
  return `${Math.floor(n / 60)} min ${Math.round(n % 60)} sec`;
}

function timeText(value) {
  if (!value) return 'No timestamp';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'No timestamp' : date.toLocaleString('en-US', { hour12: false });
}


function langSmithProjectUrl(observability) {
  const endpoint = String(observability?.endpoint || 'https://smith.langchain.com').replace(/\/$/, '');
  const project = observability?.project;
  if (!project) return 'https://smith.langchain.com/';
  return `${endpoint.replace('api.smith.langchain.com', 'smith.langchain.com')}/o/default/projects/p/${encodeURIComponent(project)}`;
}

function urlDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return 'source';
  }
}

function sourceTitle(source, index) {
  return displayText(source?.title, `Source ${index + 1} from ${urlDomain(source?.url)}`);
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[ch]));
}

function htmlFromText(value) {
  const text = String(value ?? '').trim();
  if (!text) return '';
  return text
    .split('\n')
    .filter(Boolean)
    .map((line) => `<p>${escapeHtml(line)}</p>`)
    .join('');
}

function reportSeedHtml(output) {
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

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { success: false, error: text || 'Request failed' };
  }
  if (!response.ok) {
    throw new Error(payload.message || payload.error || `Request failed with ${response.status}`);
  }
  return payload;
}

function downloadBlob(content, filename, type = 'text/html') {
  const blob = new Blob([content], { type });
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(href);
}

function SectionTitle({ eyebrow, title, action }) {
  return (
    <div className="section-title">
      <div>
        {eyebrow && <span className="section-eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  );
}

function InsightCard({ item, index, theme }) {
  return (
    <motion.article className="insight-card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.04 }}>
      <div className="insight-rank">{String(index + 1).padStart(2, '0')}</div>
      <div className="insight-copy">
        <h3>{displayText(item?.insight, 'Insight unavailable')}</h3>
        <p>{displayText(item?.basis, 'Evidence basis unavailable')}</p>
      </div>
      <Progress type="circle" size={58} percent={clampPct(item?.confidence)} strokeColor={theme.primary} trailColor={theme.trail} />
    </motion.article>
  );
}

function RiskCard({ item, index }) {
  return (
    <div className="risk-card">
      <Tag color={index === 0 ? 'red' : 'gold'}>Review</Tag>
      <h3>{displayText(item?.tension, 'Risk unavailable')}</h3>
      <p>{displayText(item?.significance, 'Significance unavailable')}</p>
      {Array.isArray(item?.between) && item.between.length > 0 && (
        <div className="chip-row">
          {item.between.slice(0, 3).map((entry, idx) => <span className="soft-chip" key={idx}>{displayText(entry, 'Viewpoint unavailable')}</span>)}
        </div>
      )}
    </div>
  );
}


function msText(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 'No timing';
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60000) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)} sec`;
  return `${Math.floor(n / 60000)} min ${Math.round((n % 60000) / 1000)} sec`;
}

function moneyText(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'Not metered';
  return `$${n.toFixed(n < 0.01 ? 4 : 2)}`;
}

function traceStatus(run) {
  if (run?.error) return 'error';
  const status = String(run?.status || '').toLowerCase();
  if (status.includes('error') || status.includes('fail')) return 'error';
  if (status.includes('run') || status.includes('pending')) return 'running';
  return 'success';
}

function traceMetricRows(observabilityTrace) {
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

function traceTimelineChartData(observabilityTrace) {
  return (observabilityTrace?.timeline || []).slice().reverse().map((run, index) => ({
    name: `T${index + 1}`,
    duration: Math.round(Number(run.duration_ms) || 0),
    steps: Array.isArray(run.children) ? run.children.length : 0,
    errors: traceStatus(run) === 'error' ? 1 : 0
  }));
}

function traceChildrenByType(children = []) {
  const counts = {};
  children.forEach((child) => {
    const key = child.type || 'unknown';
    counts[key] = (counts[key] || 0) + 1;
  });
  return Object.entries(counts).map(([name, value]) => ({ name, value }));
}

function Heatmap({ pairs }) {
  const entries = Object.entries(pairs || {}).map(([pair, value]) => {
    const [a, b] = pair.split('|');
    return { pair, a: displayText(a, 'Source A'), b: displayText(b, 'Source B'), value: Number(value) || 0 };
  }).sort((a, b) => b.value - a.value).slice(0, 9);
  if (!entries.length) return <Empty description="No divergence data" />;
  return (
    <div className="heatmap-grid">
      {entries.map((entry) => (
        <motion.div className="heatmap-tile" key={entry.pair} whileHover={{ scale: 1.02 }} style={{ '--heat': entry.value }}>
          <span>{entry.a}</span>
          <strong>{Math.round(entry.value * 100)}</strong>
          <small>{entry.b}</small>
        </motion.div>
      ))}
    </div>
  );
}

function SignalGraph({ output, task, onOpen, theme }) {
  const running = task?.status === 'running';
  const liveDetails = task?.details || {};
  const liveEvidence = Array.isArray(liveDetails.evidence) ? liveDetails.evidence.filter(Boolean).slice(0, 3) : [];
  const completed = running ? flowProgressMeta(task?.progress || 0).normalized : (output?.synthesis ? 100 : 0);
  const meta = running ? flowProgressMeta(completed) : flowProgressMeta(output?.synthesis ? 100 : 0);
  const exactMicroIndex = running
    ? MICRO_STEPS.findIndex((item) => item.stageId === task?.stage && item.name === task?.micro_stage)
    : -1;
  const currentMicroIndex = exactMicroIndex >= 0 ? exactMicroIndex : meta.microIndex;
  const currentStageIndex = exactMicroIndex >= 0 ? MICRO_STEPS[exactMicroIndex].stageIndex : meta.activeIndex;
  const currentStage = FLOW_STEPS[currentStageIndex] || meta.stage;
  const activeIndex = running ? currentStageIndex : output?.synthesis ? FLOW_STEPS.length - 1 : -1;
  const points = [
    { x: 110, y: 230 },
    { x: 285, y: 120 },
    { x: 470, y: 184 },
    { x: 650, y: 84 },
    { x: 820, y: 198 },
    { x: 1010, y: 118 }
  ];
  const d = `M ${points[0].x} ${points[0].y} C 190 112, 205 112, ${points[1].x} ${points[1].y} S 372 270, ${points[2].x} ${points[2].y} S 555 18, ${points[3].x} ${points[3].y} S 724 305, ${points[4].x} ${points[4].y} S 930 66, ${points[5].x} ${points[5].y}`;
  const remaining = Math.max(0, 100 - completed);
  const maskId = 'signal-run-mask';
  const stateClass = running ? 'is-running' : 'is-idle';
  return (
    <div className="signal-stage">
      <div className={`signal-graph ${stateClass} ${output?.synthesis ? 'has-output' : 'is-empty'}`} onClick={onOpen} role="button" tabIndex={0}>
      <svg viewBox="0 0 1120 360" aria-label="Signal flow">
        <defs>
          <linearGradient id="hkuLine" x1="0%" x2="100%" y1="0%" y2="0%">
            <stop offset="0%" stopColor={theme.primarySoft} />
            <stop offset="45%" stopColor={theme.primaryMid} />
            <stop offset="100%" stopColor={theme.primaryDark} />
          </linearGradient>
          <filter id="softGlow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <mask id={maskId} maskUnits="userSpaceOnUse">
            <rect x="0" y="0" width="1120" height="360" fill="black" />
            <path d={d} pathLength="100" stroke="white" strokeWidth="46" strokeLinecap="round" strokeDasharray={`${completed} ${remaining}`} />
          </mask>
        </defs>
        <path className="signal-shadow" d={d} />
        <path className="signal-path" d={d} />
        <path className="signal-idle-pulse" d={d} />
        <path className="signal-run-track" d={d} pathLength="100" strokeDasharray={`${completed} ${remaining}`} />
        <path className="signal-run-pulse" d={d} mask={`url(#${maskId})`} />
        {points.map((point, index) => {
          const active = index <= activeIndex;
          const step = FLOW_STEPS[index];
          return (
            <g className={`signal-node ${active ? 'active' : ''}`} key={step.id} transform={`translate(${point.x} ${point.y})`}>
              <circle r="35" />
              <circle className="node-core" r="12" />
              <text y="62">{step.label}</text>
              <text className="node-sub" y="83">{step.sub}</text>
            </g>
          );
        })}
      </svg>
      </div>
      <div className="graph-status-row">
        <div className="graph-status-main">
          <span>Run State</span>
          <strong>{running ? `${Math.round(completed)}%` : output?.synthesis ? 'Ready' : 'Idle'}</strong>
        </div>
        <div className="graph-trace compact">
          <em><span>Stage</span>{running ? currentStage.label : output?.synthesis ? 'Complete' : 'Standby'}</em>
          <em><span>Step</span>{running ? (task?.micro_stage || MICRO_STEPS[currentMicroIndex]?.name || meta.micro.name) : output?.synthesis ? 'Done' : 'Waiting'}</em>
          <em><span>Progress</span>{running ? `${meta.stagePercent}%` : 'Full path'}</em>
        </div>
      </div>
      <div className="micro-rail" aria-label="Workflow steps">
        {FLOW_STEPS.map((step, stageIndex) => (
          <div className={`micro-group ${stageIndex === currentStageIndex && running ? 'current' : ''}`} key={step.id}>
            <small>{step.label}</small>
            <div className="micro-items">
              {step.micro.map((name, microIndex) => {
                const absoluteIndex = MICRO_STEPS.findIndex((item) => item.stageIndex === stageIndex && item.name === name && item.id.endsWith(`-${microIndex}`));
                const done = running ? absoluteIndex < currentMicroIndex : output?.synthesis;
                const current = running && absoluteIndex === currentMicroIndex;
                return (
                  <b key={`${step.id}-${name}`} className={`${done ? 'done' : ''} ${current ? 'current' : ''}`} title={`${step.label}: ${name}`}>
                    {current && <span>Now</span>}{name}
                  </b>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {running && (
        <div className="live-evidence">
          <strong>{displayLog(liveDetails.message || task?.message || 'Working')}</strong>
          {liveEvidence.map((item, index) => <span key={index}>{displayLog(item)}</span>)}
        </div>
      )}
    </div>
  );
}


function TraceMetrics({ observabilityTrace }) {
  return (
    <div className="trace-metric-strip">
      {traceMetricRows(observabilityTrace).map((item) => (
        <div key={item.key}>
          <span>{item.label}</span>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  );
}

function TraceTypeBars({ data }) {
  if (!data?.length) return <Empty description="No trace types" />;
  const max = Math.max(...data.map((item) => item.value), 1);
  return (
    <div className="trace-type-bars">
      {data.map((item) => (
        <div key={item.name}>
          <span>{item.name}</span>
          <b><i style={{ width: `${Math.max(8, (item.value / max) * 100)}%`, background: TRACE_COLORS[item.name] || TRACE_COLORS.unknown }} /></b>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  );
}

function TraceTimeline({ traces, onSelect }) {
  if (!traces?.length) {
    return (
      <div className="trace-empty rich-empty">
        <RadarChartOutlined />
        <strong>No remote traces yet</strong>
        <span>Run an analysis with tracing enabled; model calls and graph steps will appear here.</span>
      </div>
    );
  }
  return (
    <div className="trace-run-list">
      {traces.map((run) => {
        const status = traceStatus(run);
        const childCount = Array.isArray(run.children) ? run.children.length : run.child_count;
        return (
          <motion.button key={run.id} className={`trace-run ${status}`} whileHover={{ y: -4, scale: 1.01 }} whileTap={{ scale: 0.99 }} onClick={() => onSelect(run)}>
            <span className="trace-dot" />
            <div className="trace-run-main">
              <strong>{displayText(run.name, 'Trace')}</strong>
              <small>{timeText(run.start_time)} · {msText(run.duration_ms)}</small>
            </div>
            <div className="trace-run-meta">
              <Tag color={status === 'error' ? 'red' : status === 'running' ? 'processing' : 'blue'}>{status}</Tag>
              <span>{compactNumber(childCount)} steps</span>
            </div>
          </motion.button>
        );
      })}
    </div>
  );
}

function LocalTraceReplay({ task, output }) {
  const live = Array.isArray(task?.timeline) ? task.timeline : [];
  const artifact = Array.isArray(output?.coordinator_trace) ? output.coordinator_trace.map((message, index) => ({
    node: `step_${index + 1}`,
    micro_stage: message.split(']')[0]?.replace('[', '') || `Step ${index + 1}`,
    message,
    evidence: []
  })) : [];
  const rows = live.length ? live.slice(-10).reverse() : artifact.slice(-10).reverse();
  if (!rows.length) {
    return (
      <div className="trace-empty rich-empty">
        <ThunderboltOutlined />
        <strong>No run replay yet</strong>
        <span>Start an analysis to see source collection, disagreement checks, fact review, and report handoff.</span>
      </div>
    );
  }
  return (
    <div className="live-timeline trace-timeline visual-timeline">
      {rows.map((entry, index) => (
        <div key={`${entry.node || 'trace'}-${index}`}>
          <span>{entry.micro_stage || entry.stage || entry.node || 'Step'}</span>
          <strong>{displayLog(entry.message)}</strong>
          {Array.isArray(entry.evidence) && entry.evidence.length > 0 && <small>{entry.evidence.slice(0, 2).map(displayLog).join(' / ')}</small>}
        </div>
      ))}
    </div>
  );
}

function TraceDetailDrawer({ run, open, onClose }) {
  const children = Array.isArray(run?.children) ? run.children : [];
  const childTypes = traceChildrenByType(children);
  return (
    <Drawer open={open} onClose={onClose} width={640} title={run ? displayText(run.name, 'Trace details') : 'Trace details'}>
      {run && (
        <div className="drawer-stack trace-detail">
          <div className="trace-detail-head">
            <Tag color={traceStatus(run) === 'error' ? 'red' : 'blue'}>{traceStatus(run)}</Tag>
            <strong>{msText(run.duration_ms)}</strong>
            <span>{timeText(run.start_time)}</span>
          </div>
          {run.error && <Alert type="error" showIcon message="Trace error" description={displayText(run.error, 'Error unavailable')} />}
          <TraceTypeBars data={childTypes} />
          <div className="trace-child-list">
            {children.length ? children.map((child) => (
              <a key={child.id} className={`trace-child ${traceStatus(child)}`} href={child.url || undefined} target="_blank" rel="noreferrer">
                <span style={{ background: TRACE_COLORS[child.type] || TRACE_COLORS.unknown }} />
                <div>
                  <strong>{displayText(child.name, 'Step')}</strong>
                  <small>{displayText(child.type, 'unknown')} · {msText(child.duration_ms)} · {compactNumber(child.total_tokens)} tokens</small>
                </div>
                {traceStatus(child) === 'error' && <WarningOutlined />}
              </a>
            )) : <Empty description="No child steps loaded" />}
          </div>
          {run.url && <Button type="primary" icon={<RadarChartOutlined />} href={run.url} target="_blank">Open Trace</Button>}
        </div>
      )}
    </Drawer>
  );
}

function MiniMetric({ icon, label, value, onClick }) {
  return (
    <motion.button className="mini-metric" whileHover={{ y: -6, scale: 1.025 }} whileTap={{ scale: 0.98 }} onClick={onClick}>
      <span>{icon}</span>
      <strong>{value}</strong>
      <label>{label}</label>
    </motion.button>
  );
}

function CommandDock({ latest, task, onOpen, onRun, onReport, onFeedback }) {
  return (
    <div className="command-dock">
      <motion.button whileHover={{ y: -5 }} whileTap={{ scale: 0.98 }} className="dock-primary" onClick={onRun} disabled={task?.status === 'running'}>
        {task?.status === 'running' ? <LoadingOutlined /> : <PlayCircleOutlined />}
        <span>{task?.status === 'running' ? 'Running' : 'Run'}</span>
      </motion.button>
      <motion.button whileHover={{ y: -5 }} whileTap={{ scale: 0.98 }} onClick={onReport}><FileTextOutlined /><span>Draft</span></motion.button>
      <motion.button whileHover={{ y: -5 }} whileTap={{ scale: 0.98 }} onClick={onFeedback}><MessageOutlined /><span>Revise</span></motion.button>
      <motion.button whileHover={{ y: -5 }} whileTap={{ scale: 0.98 }} onClick={onOpen}>{latest ? <CheckCircleOutlined /> : <ExperimentOutlined />}<span>{latest ? 'Open' : 'Setup'}</span></motion.button>
    </div>
  );
}

function ReviewEditor({ output, reportHtml, onReportHtmlChange, annotations, setAnnotations }) {
  const [commentDraft, setCommentDraft] = useState('');
  const [bubbleNoteOpen, setBubbleNoteOpen] = useState(false);
  const [pendingSelection, setPendingSelection] = useState(null);
  const citationSources = output?.source_data?.query_agent?.top_sources || [];
  const editor = useEditor({
    extensions: [
      StarterKit.configure({ link: false, underline: false }),
      Underline,
      TextStyle,
      Color.configure({ types: ['textStyle'] }),
      Highlight.configure({ multicolor: true }),
      Link.configure({ openOnClick: true, autolink: true, linkOnPaste: true }),
      Placeholder.configure({ placeholder: 'Draft, revise, cite.' })
    ],
    content: reportHtml || reportSeedHtml(output),
    editorProps: {
      attributes: {
        class: 'review-editor-content',
        spellcheck: 'false',
        autocorrect: 'off',
        autocapitalize: 'off',
        autocomplete: 'off',
        'data-gramm': 'false',
        'data-gramm_editor': 'false',
        'data-enable-grammarly': 'false',
        'data-lt-active': 'false'
      }
    },
    onUpdate: ({ editor: current }) => onReportHtmlChange(current.getHTML())
  });

  useEffect(() => {
    if (!editor) return;
    const next = reportHtml || reportSeedHtml(output);
    if (next && next !== editor.getHTML()) {
      editor.commands.setContent(next, false);
    }
  }, [editor, output, reportHtml]);

  const getSelection = () => {
    if (!editor) return null;
    const { from, to } = editor.state.selection;
    const quote = editor.state.doc.textBetween(from, to, ' ').trim();
    return { from, to, quote };
  };

  const captureSelection = () => {
    const selection = getSelection();
    if (!selection || selection.from === selection.to || !selection.quote) {
      message.warning('Select report text first');
      return null;
    }
    setPendingSelection(selection);
    return selection;
  };

  const highlightSelection = (color = '#d9f99d') => {
    if (!editor) return;
    const selection = captureSelection();
    if (!selection) return;
    editor.chain().focus().setTextSelection({ from: selection.from, to: selection.to }).setHighlight({ color }).run();
  };

  const addAnnotation = (noteText = commentDraft) => {
    if (!editor) return;
    const selection = pendingSelection || getSelection();
    const note = String(noteText || '').trim();
    if (!selection || selection.from === selection.to || !selection.quote) {
      message.warning('Select report text first');
      return;
    }
    editor.chain().focus().setTextSelection({ from: selection.from, to: selection.to }).setHighlight({ color: '#d9f99d' }).run();
    setAnnotations((items) => [{
      id: `note_${Date.now()}`,
      quote: selection.quote,
      note: note || 'Marked for review',
      createdAt: new Date().toISOString()
    }, ...items]);
    setCommentDraft('');
    setBubbleNoteOpen(false);
    setPendingSelection(null);
  };

  const openInlineNote = () => {
    const selection = captureSelection();
    if (selection) setBubbleNoteOpen(true);
  };

  const setLink = () => {
    if (!editor) return;
    const selection = captureSelection();
    if (!selection) return;
    const previousUrl = editor.getAttributes('link').href || '';
    const url = window.prompt('Source URL', previousUrl);
    if (url === null) return;
    if (!url) {
      editor.chain().focus().setTextSelection({ from: selection.from, to: selection.to }).extendMarkRange('link').unsetLink().run();
      return;
    }
    editor.chain().focus().setTextSelection({ from: selection.from, to: selection.to }).extendMarkRange('link').setLink({ href: url }).run();
  };

  return (
    <div className="review-shell">
      <div className="editor-surface">
        <div className="editor-toolbar">
          <Tooltip title="Heading"><Button onClick={() => editor?.chain().focus().toggleHeading({ level: 2 }).run()}>H2</Button></Tooltip>
          <Tooltip title="Subhead"><Button onClick={() => editor?.chain().focus().toggleHeading({ level: 3 }).run()}>H3</Button></Tooltip>
          <Tooltip title="Bold"><Button icon={<strong>B</strong>} onClick={() => editor?.chain().focus().toggleBold().run()} /></Tooltip>
          <Tooltip title="Italic"><Button icon={<em>I</em>} onClick={() => editor?.chain().focus().toggleItalic().run()} /></Tooltip>
          <Tooltip title="Underline"><Button icon={<u>U</u>} onClick={() => editor?.chain().focus().toggleUnderline().run()} /></Tooltip>
          <Tooltip title="Highlight"><Button icon={<HighlightOutlined />} onClick={() => highlightSelection('#d9f99d')} /></Tooltip>
          <Tooltip title="Source link"><Button icon={<LinkOutlined />} onClick={setLink} /></Tooltip>
          <Tooltip title="Accent text"><Button className="swatch-button green" onClick={() => editor?.chain().focus().setColor('var(--hku)').run()} /></Tooltip>
          <Tooltip title="Export HTML"><Button icon={<CloudDownloadOutlined />} onClick={() => downloadBlob(editor?.getHTML() || '', 'reviewed-report.html')}>Export</Button></Tooltip>
        </div>
        {editor && (
          <BubbleMenu
            editor={editor}
            options={{ placement: 'top', strategy: 'absolute' }}
            shouldShow={({ editor: activeEditor, state }) => activeEditor.isEditable && !state.selection.empty}
          >
            <div className="selection-menu">
              <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={() => highlightSelection('#d9f99d')}><HighlightOutlined /> Mark</button>
              <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={openInlineNote}><CommentOutlined /> Note</button>
              <button type="button" onMouseDown={(event) => event.preventDefault()} onClick={setLink}><LinkOutlined /> Cite</button>
              {bubbleNoteOpen && (
                <div className="bubble-note">
                  <Input.TextArea value={commentDraft} onChange={(event) => setCommentDraft(event.target.value)} placeholder="Note" autoSize={{ minRows: 2, maxRows: 4 }} />
                  <Button size="small" type="primary" onClick={() => addAnnotation(commentDraft)}>Add</Button>
                </div>
              )}
            </div>
          </BubbleMenu>
        )}
        <EditorContent editor={editor} />
      </div>
      <aside className="annotation-panel">
        <div className="annotation-compose compact-note-head">
          <h3>Notes</h3>
          <div className="note-count"><CommentOutlined /> {annotations.length}</div>
        </div>
        <div className="annotation-list">
          {annotations.length === 0 && <Empty description="No notes yet" />}
          {annotations.map((item) => (
            <div className="annotation-card" key={item.id}>
              <span>{timeText(item.createdAt)}</span>
              <strong>{item.quote || 'General note'}</strong>
              <p>{item.note}</p>
            </div>
          ))}
        </div>
        <div className="citation-panel">
          <h3>Citations</h3>
          {citationSources.slice(0, 6).map((source, index) => (
            <a key={`${source.url || index}`} href={source.url} target="_blank" rel="noreferrer">
              <span>{String(index + 1).padStart(2, '0')}</span>
              <strong>{sourceTitle(source, index)}</strong>
            </a>
          ))}
          {!citationSources.length && <Empty description="No sources" />}
        </div>
      </aside>
    </div>
  );
}

function ConfigDrawer({ open, onClose, config, setConfig, onSaved }) {
  const [saving, setSaving] = useState(false);
  const update = (key, value) => setConfig((current) => ({ ...current, [key]: value }));
  const save = async (startAfter = false) => {
    setSaving(true);
    try {
      const data = await apiJson('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      setConfig(data.config || config);
      if (startAfter) {
        await apiJson('/api/system/start', { method: 'POST' });
      }
      message.success(startAfter ? 'Configuration saved and system startup requested' : 'Configuration saved');
      onSaved?.();
      onClose();
    } catch (error) {
      message.error(error.message || 'Configuration update failed');
    } finally {
      setSaving(false);
    }
  };
  return (
    <Drawer open={open} onClose={onClose} width={620} title="Workspace Configuration" extra={<Button type="primary" loading={saving} onClick={() => save(false)}>Save</Button>}>
      <div className="drawer-stack">
        {CONFIG_GROUPS.map((group) => (
          <section className="config-card" key={group.title}>
            <h3>{group.title}</h3>
            <p>{group.description}</p>
            <div className="config-fields">
              {group.fields.map(([key, label, type]) => (
                <label key={key}>
                  <span>{label}</span>
                  {type === 'select' ? (
                    <Select value={config[key] || 'AnspireAPI'} onChange={(value) => update(key, value)} options={[{ value: 'AnspireAPI' }, { value: 'BochaAPI' }]} />
                  ) : type === 'boolean' ? (
                    <Switch checked={String(config[key]).toLowerCase() === 'true'} onChange={(checked) => update(key, checked ? 'True' : 'False')} />
                  ) : (
                    <Input.Password visibilityToggle={type === 'password'} type={type === 'password' ? 'password' : 'text'} value={config[key] || ''} onChange={(event) => update(key, event.target.value)} />
                  )}
                </label>
              ))}
            </div>
          </section>
        ))}
        <Button block size="large" type="primary" icon={<PlayCircleOutlined />} loading={saving} onClick={() => save(true)}>Save and Start Runtime</Button>
      </div>
    </Drawer>
  );
}

export default function App() {
  const [active, setActive] = useState('command');
  const [visualTheme, setVisualTheme] = useState(() => window.localStorage.getItem('signal-studio-theme') || 'blue');
  const [latest, setLatest] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [feedback, setFeedback] = useState({ records: [], summary: { count: 0 } });
  const [observability, setObservability] = useState(null);
  const [observabilityTrace, setObservabilityTrace] = useState(null);
  const [observabilityLoading, setObservabilityLoading] = useState(false);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [system, setSystem] = useState({ started: false, starting: false });
  const [reportStatus, setReportStatus] = useState(null);
  const [query, setQuery] = useState('');
  const [coordinatorTask, setCoordinatorTask] = useState(null);
  const [reportTask, setReportTask] = useState(null);
  const [reportEvents, setReportEvents] = useState([]);
  const [reportHtml, setReportHtml] = useState('');
  const [annotations, setAnnotations] = useState([]);
  const [config, setConfig] = useState({});
  const [configOpen, setConfigOpen] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [readoutOpen, setReadoutOpen] = useState(false);
  const [feedbackForm, setFeedbackForm] = useState({ target: 'Overall quality', action: 'Revise', priority: 'Normal', feedback: '' });
  const [loadingLatest, setLoadingLatest] = useState(false);
  const coordinatorPoll = useRef(null);
  const reportStream = useRef(null);

  const theme = THEME_TOKENS[visualTheme] || THEME_TOKENS.blue;

  const toggleVisualTheme = () => {
    setVisualTheme((current) => {
      const next = current === 'blue' ? 'green' : 'blue';
      window.localStorage.setItem('signal-studio-theme', next);
      return next;
    });
  };

  const output = latest || {};
  const synthesis = output.synthesis || {};
  const sourceData = output.source_data || {};
  const queryAgent = sourceData.query_agent || {};
  const stanceRows = Object.entries(queryAgent.stance_distribution || {}).map(([name, value]) => ({ name, value: Number(value) || 0 }));
  const topSources = queryAgent.top_sources || [];
  const insights = synthesis.top_insights || [];
  const risks = synthesis.key_tensions || [];
  const recommendations = synthesis.recommended_investigation || [];

  const healthScore = useMemo(() => {
    const confidence = clampPct(synthesis.overall_confidence || output.synthesis_confidence || 0);
    const sources = Math.min(100, Number(queryAgent.total_sources || 0));
    const errors = Array.isArray(output.agent_errors) ? output.agent_errors.length : 0;
    return Math.max(0, Math.round(confidence * 0.62 + sources * 0.28 - errors * 12));
  }, [output, queryAgent.total_sources, synthesis.overall_confidence]);


  const loadObservabilityTrace = async (quiet = true) => {
    setObservabilityLoading(true);
    try {
      const data = await apiJson('/api/observability/langsmith');
      setObservabilityTrace(data);
      if (!quiet) message.success(data.source === 'langsmith' ? 'Traces loaded' : 'Local trace loaded');
    } catch (error) {
      if (!quiet) message.warning(error.message || 'Trace data unavailable');
    } finally {
      setObservabilityLoading(false);
    }
  };

  const loadLatest = async (quiet = false) => {
    setLoadingLatest(true);
    try {
      const data = await apiJson('/api/coordinator/latest');
      setLatest(data.output || null);
      setMetadata(data.metadata || null);
      setFeedback(data.feedback || { records: [], summary: { count: 0 } });
      setObservability(data.observability || null);
      const loadedQuery = displayText(data.output?.query, 'Current analysis topic');
      if (loadedQuery && !query) setQuery(loadedQuery);
      if (!reportHtml) setReportHtml(reportSeedHtml(data.output));
      if (!quiet) message.success('Latest intelligence loaded');
    } catch (error) {
      if (!quiet) message.warning(error.message || 'No completed analysis is available');
    } finally {
      setLoadingLatest(false);
    }
  };

  const loadStatus = async () => {
    try {
      const [systemData, reportData, configData] = await Promise.allSettled([
        apiJson('/api/system/status'),
        apiJson('/api/report/status'),
        apiJson('/api/config')
      ]);
      if (systemData.status === 'fulfilled') setSystem(systemData.value);
      if (reportData.status === 'fulfilled') setReportStatus(reportData.value);
      if (configData.status === 'fulfilled') setConfig(configData.value.config || {});
    } catch {
      return undefined;
    }
  };

  useEffect(() => {
    loadLatest(true);
    loadStatus();
    loadObservabilityTrace(true);
    return () => {
      if (coordinatorPoll.current) window.clearInterval(coordinatorPoll.current);
      if (reportStream.current) reportStream.current.close();
    };
  }, []);

  const runAnalysis = async (extraFeedback = '') => {
    const analysisQuery = query.trim() || displayText(output.query, '');
    if (!analysisQuery) {
      message.warning('Enter an analysis brief first');
      return;
    }
    try {
      const data = await apiJson('/api/coordinator/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: analysisQuery, feedback: extraFeedback })
      });
      setCoordinatorTask(data.task);
      setActive('command');
      message.success('Analysis started');
      if (coordinatorPoll.current) window.clearInterval(coordinatorPoll.current);
      coordinatorPoll.current = window.setInterval(async () => {
        try {
          const taskData = await apiJson(`/api/coordinator/task/${data.task.task_id}`);
          setCoordinatorTask(taskData.task);
          if (['completed', 'error'].includes(taskData.task.status)) {
            window.clearInterval(coordinatorPoll.current);
            coordinatorPoll.current = null;
            if (taskData.task.status === 'completed') await loadLatest(true);
          }
        } catch (error) {
          window.clearInterval(coordinatorPoll.current);
          coordinatorPoll.current = null;
          message.error(error.message || 'Analysis status unavailable');
        }
      }, 1800);
    } catch (error) {
      message.error(error.message || 'Analysis failed to start');
    }
  };

  const generateReport = async () => {
    const topic = query.trim() || displayText(output.query, 'Intelligent Public Opinion Report');
    try {
      const data = await apiJson('/api/report/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: topic })
      });
      setReportTask(data.task);
      setReportEvents([{ type: 'status', message: 'Report generation started' }]);
      setActive('review');
      if (reportStream.current) reportStream.current.close();
      const stream = new EventSource(data.stream_url);
      reportStream.current = stream;
      stream.addEventListener('message', (event) => handleReportEvent(JSON.parse(event.data)));
      ['status', 'stage', 'progress', 'warning', 'html_ready', 'completed', 'error', 'log'].forEach((name) => {
        stream.addEventListener(name, (event) => handleReportEvent(JSON.parse(event.data)));
      });
      stream.onerror = () => {
        stream.close();
      };
    } catch (error) {
      message.error(error.message || 'Report generation failed to start');
    }
  };

  const handleReportEvent = async (event) => {
    const payload = event.payload || {};
    setReportEvents((items) => [{ type: event.type, message: displayLog(payload.message || payload.line || event.type), time: event.timestamp }, ...items].slice(0, 60));
    if (payload.task) setReportTask(payload.task);
    if (event.type === 'completed' || event.type === 'html_ready') {
      const taskId = event.task_id;
      try {
        const result = await apiJson(`/api/report/result/${taskId}/json`);
        setReportHtml(result.html_content || reportSeedHtml(output));
      } catch {
        return undefined;
      }
    }
  };

  const saveFeedback = async (runAfter = false) => {
    const text = feedbackForm.feedback.trim();
    if (!text) {
      message.warning('Write a concrete revision request first');
      return;
    }
    try {
      await apiJson('/api/coordinator/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query || output.query || '',
          target: feedbackForm.target,
          action: feedbackForm.action,
          priority: feedbackForm.priority,
          feedback: text,
          thread_id: coordinatorTask?.thread_id || ''
        })
      });
      message.success(runAfter ? 'Feedback saved. Refinement started.' : 'Feedback saved');
      setFeedbackForm((current) => ({ ...current, feedback: '' }));
      setFeedbackOpen(false);
      await loadLatest(true);
      if (runAfter) runAnalysis(text);
    } catch (error) {
      message.error(error.message || 'Feedback could not be saved');
    }
  };

  const startSystem = async () => {
    try {
      const data = await apiJson('/api/system/start', { method: 'POST' });
      message.success(data.message || 'System startup requested');
      loadStatus();
    } catch (error) {
      message.error(error.message || 'System startup failed');
    }
  };

  const shutdownSystem = () => {
    Modal.confirm({
      title: 'Shut down this workspace?',
      content: 'This stops the running backend services for the current session.',
      okText: 'Shut Down',
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          const data = await apiJson('/api/system/shutdown', { method: 'POST' });
          message.success(data.message || 'Shutdown requested');
        } catch (error) {
          message.error(error.message || 'Shutdown request failed');
        }
      }
    });
  };

  const evidenceColumns = [
    { title: 'Source', dataIndex: 'title', render: (_, item, index) => <a href={item.url} target="_blank" rel="noreferrer">{sourceTitle(item, index)}</a> },
    { title: 'Stance', dataIndex: 'stance', width: 110, render: (value) => <Tag color={value === 'support' ? 'green' : value === 'oppose' ? 'red' : 'blue'}>{displayText(value || 'neutral').toUpperCase()}</Tag> },
    { title: 'Trust', dataIndex: 'trust_score', width: 130, render: (value) => <Progress percent={clampPct(value)} size="small" strokeColor={theme.primary} /> }
  ];

  return (
    <ConfigProvider theme={{ token: { colorPrimary: theme.primary, borderRadius: 12, fontFamily: 'Inter, sans-serif' } }}>
      <div className={`studio-shell theme-${visualTheme}`}>
        <aside className="studio-nav">
          <div className="brand">
            <div className="brand-mark">S</div>
            <div>
              <strong>Signal Studio</strong>
              <span>Opinion intelligence</span>
            </div>
          </div>
          <nav>
            {NAV_ITEMS.map((item) => (
              <button key={item.key} className={active === item.key ? 'active' : ''} onClick={() => setActive(item.key)}>
                {item.icon}<span>{item.label}</span>
              </button>
            ))}
          </nav>
          <div className="nav-status">
            <span>Quality Score</span>
            <Progress percent={healthScore} strokeColor={theme.primarySoft} trailColor="rgba(255,255,255,.12)" />
            <small>{latest ? 'Latest analysis' : system.started ? 'Ready to run' : 'Start runtime'}</small>
          </div>
        </aside>

        <main className="studio-main">
          <header className="hero-bar compact-hero">
            <div className="hero-lockup">
              <span className="kicker">Signal Studio</span>
              <h1>Sense. Decide.</h1>
            </div>
            <Space wrap className="icon-actions">
              <Tooltip title={`Theme: ${theme.label}`}><Button shape="circle" icon={<BgColorsOutlined />} onClick={toggleVisualTheme} /></Tooltip>
              <Tooltip title="Refresh"><Button shape="circle" icon={<ReloadOutlined />} loading={loadingLatest || observabilityLoading} onClick={() => { loadLatest(false); loadStatus(); loadObservabilityTrace(false); }} /></Tooltip>
              <Tooltip title="Settings"><Button shape="circle" icon={<SettingOutlined />} onClick={() => setConfigOpen(true)} /></Tooltip>
              <Tooltip title="Shutdown"><Button danger shape="circle" icon={<PauseCircleOutlined />} onClick={shutdownSystem} /></Tooltip>
            </Space>
          </header>

          <section className="brief-panel minimal-brief">
            <SearchOutlined />
            <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Topic" />
            <CommandDock latest={latest} task={coordinatorTask} onRun={() => runAnalysis()} onReport={generateReport} onFeedback={() => setFeedbackOpen(true)} onOpen={() => setActive(latest ? 'intelligence' : 'control')} />
          </section>

          <AnimatePresence mode="wait">
            {active === 'command' && (
              <motion.section key="command" className="home-stage" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <SignalGraph output={output} task={coordinatorTask} theme={theme} onOpen={() => setActive('intelligence')} />
                <div className="metric-orbit">
                  <MiniMetric icon={<SafetyCertificateOutlined />} label="Trust" value={percentText(synthesis.overall_confidence)} onClick={() => setActive('intelligence')} />
                  <MiniMetric icon={<SearchOutlined />} label="Proof" value={compactNumber(queryAgent.total_sources)} onClick={() => setActive('evidence')} />
                  <MiniMetric icon={<ClockCircleOutlined />} label="Time" value={durationText(output.pipeline_duration_seconds).replace('No run yet', 'Idle')} onClick={() => setActive('control')} />
                  <MiniMetric icon={<WarningOutlined />} label="Risk" value={compactNumber(risks.length)} onClick={() => setActive('intelligence')} />
                </div>
              </motion.section>
            )}

            {active === 'intelligence' && (
              <motion.section key="intelligence" className="page-grid readout-grid" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <article className="span-7 studio-card lead-card compact-readout" onClick={() => setReadoutOpen(true)}>
                  <SectionTitle eyebrow="Readout" title="Signal" action={<Tag color="green">{percentText(synthesis.overall_confidence)}</Tag>} />
                  <h3>{displayText(insights[0]?.insight || synthesis.summary, 'No signal yet')}</h3>
                  <Button type="primary" ghost icon={<CheckCircleOutlined />}>Details</Button>
                </article>
                <aside className="span-5 studio-card risk-stack compact-risk">
                  <SectionTitle eyebrow="Risk" title="Watch" />
                  {risks.length ? risks.slice(0, 3).map((item, index) => (
                    <motion.button className="risk-token" key={index} whileHover={{ y: -4, scale: 1.015 }} onClick={() => setReadoutOpen(true)}>
                      <WarningOutlined />
                      <span>{displayText(item?.tension, 'Open risk')}</span>
                    </motion.button>
                  )) : <Empty description="Clear" />}
                </aside>
                <div className="span-12 insight-stack compact-insights">
                  {insights.length ? insights.map((item, index) => <InsightCard key={index} item={item} index={index} theme={theme} />) : <Empty description="No insights" />}
                </div>
              </motion.section>
            )}

            {active === 'evidence' && (
              <motion.section key="evidence" className="page-grid" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <div className="span-5 studio-card chart-card">
                  <SectionTitle eyebrow="Stance" title="Signal mix" />
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={stanceRows} layout="vertical" margin={{ left: 10, right: 12 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                      <YAxis dataKey="name" type="category" width={88} tickFormatter={(value) => displayText(value, 'Other')} />
                      <ChartTooltip formatter={(value) => percentText(value)} />
                      <Bar dataKey="value" radius={[0, 10, 10, 0]}>
                        {stanceRows.map((entry) => <Cell key={entry.name} fill={STANCE_COLORS[entry.name] || STANCE_COLORS.unknown} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="span-7 studio-card">
                  <SectionTitle eyebrow="Divergence" title="Where signals disagree" />
                  <Heatmap pairs={output.divergence_matrix?.pairs} />
                </div>
                <div className="span-8 studio-card source-card">
                  <SectionTitle eyebrow="Grounding" title="Top evidence" />
                  <Table columns={evidenceColumns} dataSource={topSources.map((item, index) => ({ ...item, key: index }))} pagination={{ pageSize: 6 }} size="middle" />
                </div>
                <div className="span-4 studio-card platform-card">
                  <SectionTitle eyebrow="Audience" title="Platform readings" />
                  <div className="platform-list collapsed-platform-list">
                    {Object.entries(output.platform_interpretations || {}).length ? (
                      <Collapse
                        ghost
                        accordion
                        items={Object.entries(output.platform_interpretations || {}).map(([platform, text]) => ({
                          key: platform,
                          label: (
                            <div className="platform-summary">
                              <strong>{displayText(platform, 'Platform')}</strong>
                              <span>{compactSummary(text)}</span>
                            </div>
                          ),
                          children: <p><MarkdownText value={text} /></p>
                        }))}
                      />
                    ) : <Empty description="No platform readings" />}
                  </div>
                </div>
              </motion.section>
            )}

            {active === 'review' && (
              <motion.section key="review" className="review-page" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <div className="review-header studio-card">
                  <SectionTitle eyebrow="Report board" title="Edit, highlight, and annotate the final narrative" />
                  <Space wrap>
                    <Button icon={<FileTextOutlined />} onClick={generateReport}>Generate Report</Button>
                    <Button icon={<CloudDownloadOutlined />} disabled={!reportTask?.task_id} href={reportTask?.task_id ? `/api/report/download/${reportTask.task_id}` : undefined}>HTML</Button>
                    <Button icon={<BookOutlined />} disabled={!reportTask?.task_id} href={reportTask?.task_id ? `/api/report/export/md/${reportTask.task_id}` : undefined}>Markdown</Button>
                    <Button icon={<FilePdfOutlined />} disabled={!reportTask?.task_id} href={reportTask?.task_id ? `/api/report/export/pdf/${reportTask.task_id}` : undefined}>PDF</Button>
                  </Space>
                </div>
                {reportTask?.status === 'running' && <Alert type="info" showIcon message="Report generation is running" description={<Progress percent={reportTask.progress || 0} strokeColor={theme.primary} />} />}
                <ReviewEditor output={output} reportHtml={reportHtml} onReportHtmlChange={setReportHtml} annotations={annotations} setAnnotations={setAnnotations} />
                <div className="studio-card event-card">
                  <SectionTitle eyebrow="Generation stream" title="Recent writing events" />
                  <Timeline items={(reportEvents.length ? reportEvents : [{ message: 'No report events yet', type: 'idle' }]).slice(0, 8).map((item) => ({ color: item.type === 'error' ? 'red' : item.type === 'warning' ? 'gold' : 'green', children: <span>{item.message}</span> }))} />
                </div>
              </motion.section>
            )}

            {active === 'control' && (
              <motion.section key="control" className="page-grid monitor-grid" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <div className="span-8 studio-card monitor-hero-card">
                  <SectionTitle
                    eyebrow="Monitor"
                    title="Run Replay"
                    action={(
                      <Space wrap>
                        <Button icon={<ReloadOutlined />} loading={observabilityLoading} onClick={() => { loadStatus(); loadObservabilityTrace(false); }}>Refresh</Button>
                        <Button type="primary" icon={<PlayCircleOutlined />} onClick={startSystem}>{system.started ? 'Runtime Ready' : 'Start Runtime'}</Button>
                      </Space>
                    )}
                  />
                  <LocalTraceReplay task={coordinatorTask} output={output} />
                </div>

                <div className="span-4 studio-card monitor-side-card">
                  <SectionTitle eyebrow="Quality" title="Latest Analysis" />
                  <div className="quality-orb">
                    <Progress type="circle" percent={healthScore} size={132} strokeColor={theme.primary} trailColor={theme.trail} />
                    <span>{latest ? 'Artifact loaded' : 'No run yet'}</span>
                  </div>
                  <div className="meta-list compact-meta">
                    <div><span>Sources</span><strong>{compactNumber(queryAgent.total_sources)}</strong></div>
                    <div><span>Confidence</span><strong>{percentText(synthesis.overall_confidence)}</strong></div>
                    <div><span>Runtime</span><strong>{durationText(output.pipeline_duration_seconds)}</strong></div>
                    <div><span>Errors</span><strong>{compactNumber(output.agent_errors?.length)}</strong></div>
                  </div>
                </div>

                <div className="span-12 studio-card langsmith-card">
                  <SectionTitle
                    eyebrow="Tracing"
                    title="LangSmith Traces"
                    action={(
                      <Space wrap>
                        <Tag color={observabilityTrace?.source === 'langsmith' ? 'blue' : observabilityTrace?.configured ? 'gold' : 'default'}>
                          {observabilityTrace?.source === 'langsmith' ? 'Live' : observabilityTrace?.configured ? 'Fallback' : 'Not configured'}
                        </Tag>
                        {(observabilityTrace?.project_url || observability?.enabled) && <Button icon={<RadarChartOutlined />} href={observabilityTrace?.project_url || langSmithProjectUrl(observability)} target="_blank">Open LangSmith</Button>}
                      </Space>
                    )}
                  />
                  <p className="card-brief">Local replay stays readable here. LangSmith adds the full model-call trace, timings, errors, and evaluation trail.</p>
                  <TraceMetrics observabilityTrace={observabilityTrace} />
                  <div className="trace-visual-grid">
                    <div className="trace-chart-panel">
                      <strong>Trace latency</strong>
                      <ResponsiveContainer width="100%" height={210}>
                        <LineChart data={traceTimelineChartData(observabilityTrace)} margin={{ top: 14, right: 18, left: 0, bottom: 6 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="name" />
                          <YAxis tickFormatter={(value) => value >= 1000 ? `${Math.round(value / 1000)}s` : `${value}ms`} width={52} />
                          <ChartTooltip formatter={(value, name) => [name === 'duration' ? msText(value) : value, name]} />
                          <Line type="monotone" dataKey="duration" stroke={theme.primary} strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="trace-chart-panel">
                      <strong>Step mix</strong>
                      <TraceTypeBars data={observabilityTrace?.type_breakdown || []} />
                    </div>
                  </div>
                  <TraceTimeline traces={observabilityTrace?.timeline || []} onSelect={setSelectedTrace} />
                </div>

                <div className="span-7 studio-card">
                  <SectionTitle eyebrow="Review loop" title="Revision Requests" action={<Button icon={<MessageOutlined />} onClick={() => setFeedbackOpen(true)}>New Request</Button>} />
                  <div className="feedback-list">
                    {(feedback.records || []).length === 0 && <Empty description="No feedback saved" />}
                    {(feedback.records || []).slice().reverse().map((item) => (
                      <div className="feedback-item" key={item.id}>
                        <Tag color={String(item.priority).toLowerCase() === 'critical' ? 'red' : 'blue'}>{displayText(item.priority, 'Normal')}</Tag>
                        <strong>{displayText(item.target, 'Overall quality')}</strong>
                        <p>{displayText(item.feedback, 'Feedback unavailable')}</p>
                        <span>{timeText(item.created_at)}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="span-5 studio-card">
                  <SectionTitle eyebrow="Current run" title="Artifact" />
                  <div className="meta-list">
                    <div><span>Updated</span><strong>{timeText(metadata?.modified_at)}</strong></div>
                    <div><span>Archives</span><strong>{compactNumber(metadata?.archive_count)}</strong></div>
                    <div><span>Format</span><strong>{metadata?.schema_version || 'Unknown'}</strong></div>
                    <div><span>Trace source</span><strong>{observabilityTrace?.source === 'langsmith' ? 'LangSmith' : observabilityTrace?.source === 'local_artifact' ? 'Local artifact' : 'Pending'}</strong></div>
                  </div>
                </div>
              </motion.section>
            )}

          </AnimatePresence>
        </main>
      </div>

      <Drawer open={feedbackOpen} onClose={() => setFeedbackOpen(false)} width={520} title="Revision Request">
        <div className="drawer-stack">
          <Form layout="vertical">
            <Form.Item label="Review target"><Select value={feedbackForm.target} onChange={(value) => setFeedbackForm((current) => ({ ...current, target: value }))} options={['Overall quality', 'Executive readout', 'Evidence grounding', 'Report narrative', 'Risk interpretation'].map((value) => ({ value }))} /></Form.Item>
            <Form.Item label="Requested action"><Radio.Group value={feedbackForm.action} onChange={(event) => setFeedbackForm((current) => ({ ...current, action: event.target.value }))}><Radio.Button value="Review">Review</Radio.Button><Radio.Button value="Revise">Revise</Radio.Button><Radio.Button value="Rerun">Rerun</Radio.Button></Radio.Group></Form.Item>
            <Form.Item label="Priority"><Slider marks={{ 0: 'Normal', 50: 'High', 100: 'Critical' }} step={null} value={{ Normal: 0, High: 50, Critical: 100 }[feedbackForm.priority]} onChange={(value) => setFeedbackForm((current) => ({ ...current, priority: value === 100 ? 'Critical' : value === 50 ? 'High' : 'Normal' }))} /></Form.Item>
            <Form.Item label="Specific request"><Input.TextArea value={feedbackForm.feedback} onChange={(event) => setFeedbackForm((current) => ({ ...current, feedback: event.target.value }))} placeholder="Explain what is wrong, what evidence is missing, or how the report should change" autoSize={{ minRows: 5, maxRows: 8 }} /></Form.Item>
          </Form>
          <Button block size="large" onClick={() => saveFeedback(false)}>Save Request</Button>
          <Button block size="large" type="primary" icon={<SendOutlined />} onClick={() => saveFeedback(true)}>Save and Run Refinement</Button>
        </div>
      </Drawer>


      <Modal open={readoutOpen} onCancel={() => setReadoutOpen(false)} footer={null} width={860} title="Readout Details">
        <div className="readout-modal">
          <p>{displayText(synthesis.summary, 'No synthesis is available yet.')}</p>
          <div className="modal-list">
            {recommendations.slice(0, 5).map((item, index) => <div key={index}><CheckCircleOutlined />{displayText(item, 'Follow-up')}</div>)}
          </div>
        </div>
      </Modal>

      <TraceDetailDrawer run={selectedTrace} open={Boolean(selectedTrace)} onClose={() => setSelectedTrace(null)} />

      <ConfigDrawer open={configOpen} onClose={() => setConfigOpen(false)} config={config} setConfig={setConfig} onSaved={loadStatus} />
    </ConfigProvider>
  );
}
