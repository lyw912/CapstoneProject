import { useState } from 'react';
import { Button, Empty, Modal, Tag } from 'antd';
import { motion } from 'framer-motion';
import {
  CheckCircleOutlined,
  WarningOutlined
} from '@ant-design/icons';
import SectionTitle from '../components/SectionTitle';
import InsightCard from '../components/InsightCard';
import { displayText, percentText } from '../utils/helpers';

export default function IntelligenceView({ output, theme, setReadoutOpen }) {
  const synthesis = output.synthesis || {};
  const insights = synthesis.top_insights || [];
  const risks = synthesis.key_tensions || [];
  const recommendations = synthesis.recommended_investigation || [];

  return (
    <motion.section key="intelligence" className="page-grid readout-grid" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
      <article className="span-7 studio-card lead-card compact-readout" onClick={() => setReadoutOpen(true)}>
        <SectionTitle eyebrow="Readout" title="Signal" action={<Tag color="green">{percentText(synthesis.overall_confidence)}</Tag>} />
        <h3>{displayText(insights[0]?.insight || synthesis.summary, 'No signal yet')}</h3>
        <Button type="primary" ghost icon={<CheckCircleOutlined />}>Details</Button>
      </article>
      <aside className="span-5 studio-card risk-stack compact-risk">
        <SectionTitle eyebrow="Risk" title="Watch" />
        {risks.length ? risks.slice(0, 3).map((item, index) => (
          <motion.button className="risk-token" key={index} whileHover={{ y: -4, scale: 1.015 }} onClick={() => setReadoutOpen(true)}>
            <WarningOutlined />
            <span>{displayText(item?.tension, 'Open risk')}</span>
          </motion.button>
        )) : <Empty description="Clear" />}
      </aside>
      <div className="span-12 insight-stack compact-insights">
        {insights.length ? insights.map((item, index) => <InsightCard key={index} item={item} index={index} theme={theme} />) : <Empty description="No insights" />}
      </div>
    </motion.section>
  );
}
