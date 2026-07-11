import { useState } from 'react';
import { Empty, Modal, Tag } from 'antd';
import { motion } from 'framer-motion';
import { compactNumber, displayText, platformLabel } from '../utils/helpers';

const DEFAULT_LOW_SAMPLE_THRESHOLD = 3;

function heatStyle(value) {
  const bounded = Math.max(0, Math.min(1, Number(value) || 0));
  const hue = 205 - Math.round(bounded * 195);
  const alpha = 0.14 + bounded * 0.42;
  return {
    '--heat': bounded,
    background: `linear-gradient(145deg, rgba(255,255,255,.90), hsla(${hue}, 78%, 58%, ${alpha}))`,
    borderColor: `hsla(${hue}, 72%, 44%, ${0.18 + bounded * 0.38})`
  };
}

function distributionText(distribution = {}) {
  const entries = Object.entries(distribution).sort((a, b) => Number(b[1]) - Number(a[1]));
  if (!entries.length) return 'Not available';
  return entries
    .map(([stance, value]) => `${displayText(stance, 'Other')} ${Math.round(Number(value) * 100)}%`)
    .join(' · ');
}

export default function Heatmap({
  pairs,
  platformCounts = {},
  groupDistributions = {},
  lowSampleThreshold = DEFAULT_LOW_SAMPLE_THRESHOLD
}) {
  const [selected, setSelected] = useState(null);
  const entries = Object.entries(pairs || {}).map(([pair, value]) => {
    const [aRaw, bRaw] = pair.split('|');
    const a = displayText(aRaw, 'Source A');
    const b = displayText(bRaw, 'Source B');
    const aKey = a.toLowerCase();
    const bKey = b.toLowerCase();
    const aCount = Number(platformCounts[a] ?? platformCounts[aKey] ?? 0);
    const bCount = Number(platformCounts[b] ?? platformCounts[bKey] ?? 0);
    return {
      pair,
      a,
      b,
      aLabel: platformLabel(a, true),
      bLabel: platformLabel(b, true),
      aCount,
      bCount,
      aDistribution: groupDistributions[a] || groupDistributions[aKey] || {},
      bDistribution: groupDistributions[b] || groupDistributions[bKey] || {},
      value: Number(value) || 0,
      lowSample: aCount < lowSampleThreshold || bCount < lowSampleThreshold
    };
  }).sort((a, b) => b.value - a.value).slice(0, 9);
  if (!entries.length) return <Empty description="No divergence data" />;
  return (
    <>
      <div className="heatmap-grid">
        {entries.map((entry) => {
          const score = Math.round(entry.value * 100);
          return (
            <motion.button
              type="button"
              className="heatmap-tile"
              key={entry.pair}
              whileHover={{ scale: 1.02 }}
              style={heatStyle(entry.value)}
              onClick={() => setSelected(entry)}
            >
              <span>{`${entry.aLabel} · n=${compactNumber(entry.aCount)}`}</span>
              <strong>{score}</strong>
              <small>{`${entry.bLabel} · n=${compactNumber(entry.bCount)}`}</small>
            </motion.button>
          );
        })}
      </div>
      <Modal
        open={Boolean(selected)}
        onCancel={() => setSelected(null)}
        footer={null}
        title={selected ? `${selected.aLabel} vs ${selected.bLabel}` : 'Divergence details'}
      >
        {selected && (
          <div className="heatmap-detail">
            <div>
              <span>Divergence score</span>
              <strong>{`${Math.round(selected.value * 100)}/100`}</strong>
            </div>
            <div>
              <span>{selected.aLabel}</span>
              <strong>{`${compactNumber(selected.aCount)} canonical clusters`}</strong>
            </div>
            <div>
              <span>{selected.bLabel}</span>
              <strong>{`${compactNumber(selected.bCount)} canonical clusters`}</strong>
            </div>
            <div>
              <span>{`${selected.aLabel} stance mix`}</span>
              <strong>{distributionText(selected.aDistribution)}</strong>
            </div>
            <div>
              <span>{`${selected.bLabel} stance mix`}</span>
              <strong>{distributionText(selected.bDistribution)}</strong>
            </div>
            {selected.lowSample && <Tag color="gold">Low sample size</Tag>}
            <p>Higher values mean the sampled channels have more different Laplace-smoothed stance-label distributions. This is evidence-sample diagnostics, not a population-level conclusion.</p>
          </div>
        )}
      </Modal>
    </>
  );
}
