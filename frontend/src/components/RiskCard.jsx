import { Tag } from 'antd';
import { displayText } from '../utils/helpers';

export default function RiskCard({ item, index }) {
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
