import { TRACE_COLORS } from '../utils/constants';
import { traceMetricRows } from '../utils/helpers';

export default function TraceMetrics({ observabilityTrace }) {
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
