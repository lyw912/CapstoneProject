import { Empty } from 'antd';
import { motion } from 'framer-motion';
import { displayText } from '../utils/helpers';

export default function Heatmap({ pairs }) {
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
