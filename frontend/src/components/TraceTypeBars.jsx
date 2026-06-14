import { Empty } from 'antd';
import { TRACE_COLORS } from '../utils/constants';

export default function TraceTypeBars({ data }) {
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
