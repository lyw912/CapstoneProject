import { Alert, Button, Drawer, Empty, Tag } from 'antd';
import { WarningOutlined, RadarChartOutlined } from '@ant-design/icons';
import { TRACE_COLORS } from '../utils/constants';
import { displayText, timeText, msText, compactNumber, traceStatus, traceChildrenByType } from '../utils/helpers';
import TraceTypeBars from './TraceTypeBars';

export default function TraceDetailDrawer({ run, open, onClose }) {
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
          {run.error && <Alert type="error" showIcon message="Trace diagnostic" description={displayText(run.error, 'Diagnostic pending')} />}
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
