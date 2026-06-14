import { ThunderboltOutlined } from '@ant-design/icons';
import { displayLog } from '../utils/helpers';

export default function LocalTraceReplay({ task, output }) {
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
