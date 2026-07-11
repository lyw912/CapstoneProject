import { Popover, Progress, Tag } from 'antd';
import { motion } from 'framer-motion';
import { compactSummary, displayText, clampPct } from '../utils/helpers';

export default function InsightCard({ item, index, theme }) {
  return (
    <motion.article className="insight-card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.04 }}>
      <div className="insight-rank">{String(index + 1).padStart(2, '0')}</div>
      <div className="insight-copy">
        <h3>{displayText(item?.insight, 'Insight pending')}</h3>
        <Popover
          trigger={['hover', 'click']}
          title="Evidence basis"
          content={<pre className="signal-full-popover">{displayText(item?.basis, 'Evidence basis pending')}</pre>}
        >
          <button type="button" className="insight-basis-button">{compactSummary(item?.basis, 210)}</button>
        </Popover>
        <div className="insight-tags">
          {(item?.meta_tags || []).slice(0, 3).map((tag) => <Tag key={tag}>{displayText(tag, "Evidence")}</Tag>)}
          {(item?.citation_spans || []).length > 0 && <Tag>{`${item.citation_spans.length} cited source${item.citation_spans.length === 1 ? '' : 's'}`}</Tag>}
        </div>
      </div>
      <Progress type="circle" size={58} percent={clampPct(item?.confidence)} strokeColor={theme.primary} trailColor={theme.trail} />
    </motion.article>
  );
}
