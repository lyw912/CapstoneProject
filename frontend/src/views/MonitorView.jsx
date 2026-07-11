import { useState } from 'react';
import {
  Button,
  Empty,
  Progress,
  Space,
  Tag,
  message
} from 'antd';
import { motion } from 'framer-motion';
import {
  ReloadOutlined,
  PlayCircleOutlined,
  RadarChartOutlined,
  MessageOutlined,
  SendOutlined
} from '@ant-design/icons';
import {
  Line,
  LineChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis
} from 'recharts';
import SectionTitle from '../components/SectionTitle';
import TraceMetrics from '../components/TraceMetrics';
import TraceTypeBars from '../components/TraceTypeBars';
import TraceTimeline from '../components/TraceTimeline';
import LocalTraceReplay from '../components/LocalTraceReplay';
import TraceDetailDrawer from '../components/TraceDetailDrawer';
import { apiJson, displayText, timeText, percentText, compactNumber, durationText, msText, langSmithProjectUrl, traceTimelineChartData, signalGraphSummary, signalProviderDiagnostics } from '../utils/helpers';

export default function MonitorView({
  output, theme, system, coordinatorTask,
  observabilityTrace, observability, observabilityLoading,
  loadStatus, loadObservabilityTrace, startSystem,
  feedback, setFeedbackOpen, metadata,
  queryAgent, synthesis
}) {
  const [selectedTrace, setSelectedTrace] = useState(null);
  const graphSummary = signalGraphSummary(output);
  const providerDiagnostics = signalProviderDiagnostics(output);

  return (
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
          <Progress type="circle" percent={(() => {
            const confidence = (synthesis.overall_confidence || 0) * 100;
            const sources = Math.min(100, Number(graphSummary.canonical_count || queryAgent.canonical_sources || queryAgent.total_sources || 0) * 10);
            const errors = Array.isArray(output.agent_errors) ? output.agent_errors.length : 0;
            return Math.max(0, Math.round(confidence * 0.62 + sources * 0.28 - errors * 12));
          })()} size={132} strokeColor={theme.primary} trailColor={theme.trail} />
          <span>{output.synthesis ? 'Artifact loaded' : 'No run yet'}</span>
        </div>
        <div className="meta-list compact-meta">
          <div><span>Distinct evidence</span><strong>{compactNumber(graphSummary.canonical_count || queryAgent.canonical_sources || queryAgent.total_sources)}</strong></div>
          <div><span>Confidence</span><strong>{percentText(synthesis.overall_confidence)}</strong></div>
          <div><span>Runtime</span><strong>{durationText(output.pipeline_duration_seconds)}</strong></div>
          <div><span>Diagnostics</span><strong>{compactNumber(providerDiagnostics.length || output.agent_errors?.length)}</strong></div>
        </div>
      </div>

      <div className="span-12 studio-card langsmith-card">
        <SectionTitle
          eyebrow="Tracing"
          title="LangSmith Traces"
          action={(
            <Space wrap>
              <Tag color={observabilityTrace?.source === 'langsmith' ? 'blue' : observabilityTrace?.configured ? 'gold' : 'default'}>
                {observabilityTrace?.source === 'langsmith' ? 'Live' : observabilityTrace?.configured ? 'Local' : 'Trace setup'}
              </Tag>
              {(observabilityTrace?.project_url || observability?.enabled) && <Button icon={<RadarChartOutlined />} href={observabilityTrace?.project_url || langSmithProjectUrl(observability)} target="_blank">Open LangSmith</Button>}
            </Space>
          )}
        />
        <p className="card-brief">Local replay stays readable here. LangSmith adds the full model-call trace, timings, diagnostics, and evaluation trail.</p>
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
              <p>{displayText(item.feedback, 'Feedback pending')}</p>
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

      <TraceDetailDrawer run={selectedTrace} open={Boolean(selectedTrace)} onClose={() => setSelectedTrace(null)} />
    </motion.section>
  );
}
