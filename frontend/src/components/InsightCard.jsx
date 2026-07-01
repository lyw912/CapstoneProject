import { Progress } from 'antd';
import { motion } from 'framer-motion';
import { displayText, clampPct } from '../utils/helpers';

export default function InsightCard({ item, index, theme }) {
  return (
    <motion.article className="insight-card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.04 }}>
      <div className="insight-rank">{String(index + 1).padStart(2, '0')}</div>
      <div className="insight-copy">
        <h3>{displayText(item?.insight, 'Insight pending')}</h3>
        <p>{displayText(item?.basis, 'Evidence basis pending')}</p>
      </div>
      <Progress type="circle" size={58} percent={clampPct(item?.confidence)} strokeColor={theme.primary} trailColor={theme.trail} />
    </motion.article>
  );
}
