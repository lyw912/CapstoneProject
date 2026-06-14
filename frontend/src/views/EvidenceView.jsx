import { Collapse, Empty, Progress, Table, Tag } from 'antd';
import { motion } from 'framer-motion';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis
} from 'recharts';
import SectionTitle from '../components/SectionTitle';
import Heatmap from '../components/Heatmap';
import MarkdownText from '../components/MarkdownText';
import { STANCE_COLORS } from '../utils/constants';
import { displayText, clampPct, percentText, compactSummary, sourceTitle } from '../utils/helpers';

export default function EvidenceView({ output, theme }) {
  const sourceData = output.source_data || {};
  const queryAgent = sourceData.query_agent || {};
  const stanceRows = Object.entries(queryAgent.stance_distribution || {}).map(([name, value]) => ({ name, value: Number(value) || 0 }));
  const topSources = queryAgent.top_sources || [];

  const evidenceColumns = [
    { title: 'Source', dataIndex: 'title', render: (_, item, index) => <a href={item.url} target="_blank" rel="noreferrer">{sourceTitle(item, index)}</a> },
    { title: 'Stance', dataIndex: 'stance', width: 110, render: (value) => <Tag color={value === 'support' ? 'green' : value === 'oppose' ? 'red' : 'blue'}>{displayText(value || 'neutral').toUpperCase()}</Tag> },
    { title: 'Trust', dataIndex: 'trust_score', width: 130, render: (value) => <Progress percent={clampPct(value)} size="small" strokeColor={theme.primary} /> }
  ];

  return (
    <motion.section key="evidence" className="page-grid" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
      <div className="span-5 studio-card chart-card">
        <SectionTitle eyebrow="Stance" title="Signal mix" />
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={stanceRows} layout="vertical" margin={{ left: 10, right: 12 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tickFormatter={(value) => `${Math.round(value * 100)}%`} />
            <YAxis dataKey="name" type="category" width={88} tickFormatter={(value) => displayText(value, 'Other')} />
            <ChartTooltip formatter={(value) => percentText(value)} />
            <Bar dataKey="value" radius={[0, 10, 10, 0]}>
              {stanceRows.map((entry) => <Cell key={entry.name} fill={STANCE_COLORS[entry.name] || STANCE_COLORS.unknown} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="span-7 studio-card">
        <SectionTitle eyebrow="Divergence" title="Where signals disagree" />
        <Heatmap pairs={output.divergence_matrix?.pairs} />
      </div>
      <div className="span-8 studio-card source-card">
        <SectionTitle eyebrow="Grounding" title="Top evidence" />
        <Table columns={evidenceColumns} dataSource={topSources.map((item, index) => ({ ...item, key: index }))} pagination={{ pageSize: 6 }} size="middle" />
      </div>
      <div className="span-4 studio-card platform-card">
        <SectionTitle eyebrow="Audience" title="Platform readings" />
        <div className="platform-list collapsed-platform-list">
          {Object.entries(output.platform_interpretations || {}).length ? (
            <Collapse
              ghost
              accordion
              items={Object.entries(output.platform_interpretations || {}).map(([platform, text]) => ({
                key: platform,
                label: (
                  <div className="platform-summary">
                    <strong>{displayText(platform, 'Platform')}</strong>
                    <span>{compactSummary(text)}</span>
                  </div>
                ),
                children: <p><MarkdownText value={text} /></p>
              }))}
            />
          ) : <Empty description="No platform readings" />}
        </div>
      </div>
    </motion.section>
  );
}
