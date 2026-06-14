import { Tag } from 'antd';
import { motion } from 'framer-motion';
import { RadarChartOutlined } from '@ant-design/icons';
import { displayText, timeText, msText, compactNumber, traceStatus } from '../utils/helpers';

export default function TraceTimeline({ traces, onSelect }) {
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
