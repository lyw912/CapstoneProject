import { Button, Empty, Alert, Tag } from 'antd';
import { motion } from 'framer-motion';
import {
  BranchesOutlined,
  CheckCircleOutlined,
  FieldTimeOutlined,
  LinkOutlined,
  SearchOutlined,
  WarningOutlined
} from '@ant-design/icons';
import SectionTitle from '../components/SectionTitle';
import InsightCard from '../components/InsightCard';
import { compactNumber, displayText, percentText, readableWarning, signalFreshnessSummary, signalQualitySummary, signalWarnings } from '../utils/helpers';
import { buildExecutiveReadout } from '../utils/readout';

function numeric(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function sourceAgeLabel(hours) {
  if (!Number.isFinite(hours)) return 'age unavailable';
  if (hours >= 48) return Math.round(hours / 24) + 'd median age';
  return Math.round(hours) + 'h median age';
}

function buildEvidenceGaps(readout, quality, freshness) {
  const stats = readout.stats || {};
  const canonicalCount = numeric(stats.canonicalCount ?? quality.canonical_count);
  const rawCount = numeric(stats.rawCount ?? quality.raw_count);
  const hasEvidence = canonicalCount > 0 || rawCount > 0;
  const amplification = numeric(stats.amplificationRatio ?? quality.amplification_ratio);
  const exactDuplicates = numeric(stats.exactDuplicateCount ?? quality.exact_duplicate_count);
  const lowRelevance = numeric(quality.low_relevance_ratio);
  const lowInformation = numeric(quality.low_information_ratio);
  const lowStanceConfidence = numeric(quality.stance_low_confidence_ratio);
  const staleSourceRatio = numeric(freshness.stale_source_ratio);
  const sourceAge = numeric(stats.medianAgeHours ?? freshness.median_age_hours, NaN);
  const officialCount = numeric(stats.officialCount);
  const platformCount = numeric(stats.socialPlatformCount);
  const pricingClaims = numeric(stats.pricingClaims);
  const supportOrOfficial = numeric(stats.pricingSupportOrOfficial);
  const oppose = numeric(stats.pricingOppose);
  const gaps = [];

  if (!hasEvidence) return [];

  if (officialCount <= 0) {
    gaps.push({
      key: 'official',
      icon: <LinkOutlined />,
      label: 'Source',
      score: 96,
      title: 'Official anchor',
      detail: 'No first-party source group found; add official docs before factual wording.'
    });
  }

  if (amplification >= 0.35 || exactDuplicates > 0) {
    gaps.push({
      key: 'independent',
      icon: <BranchesOutlined />,
      label: 'Sample',
      score: 78 + Math.round(amplification * 20),
      title: 'Independent samples',
      detail: percentText(amplification) + ' repeated coverage; add non-duplicate sources from new domains or platforms.'
    });
  }

  if ((Number.isFinite(sourceAge) && sourceAge > 72) || staleSourceRatio >= 0.25) {
    gaps.push({
      key: 'fresh',
      icon: <FieldTimeOutlined />,
      label: 'Freshness',
      score: 72 + Math.round(staleSourceRatio * 18),
      title: 'Fresh reactions',
      detail: sourceAgeLabel(sourceAge) + '; collect recent 24-72h platform signals.'
    });
  }

  if (platformCount > 0 && platformCount < 3) {
    gaps.push({
      key: 'breadth',
      icon: <SearchOutlined />,
      label: 'Coverage',
      score: 70,
      title: 'Platform breadth',
      detail: compactNumber(platformCount) + ' platform' + (platformCount === 1 ? '' : 's') + ' in sample; add more channels before comparing audiences.'
    });
  }

  if (pricingClaims > 0 && (supportOrOfficial === 0 || oppose === 0)) {
    gaps.push({
      key: 'counter',
      icon: <SearchOutlined />,
      label: 'Balance',
      score: 68,
      title: 'Counter-evidence',
      detail: supportOrOfficial === 0
        ? 'No supporting or official pricing claim found; search for direct confirmations.'
        : 'No opposing pricing claim found; search explicit criticism before final judgment.'
    });
  }

  if (lowStanceConfidence >= 0.25) {
    gaps.push({
      key: 'stance',
      icon: <WarningOutlined />,
      label: 'Review',
      score: 62 + Math.round(lowStanceConfidence * 20),
      title: 'Clearer stance evidence',
      detail: percentText(lowStanceConfidence) + ' low-confidence stance labels; add quotes that clearly support or oppose.'
    });
  }

  if (lowRelevance >= 0.12 || lowInformation >= 0.12) {
    gaps.push({
      key: 'precision',
      icon: <SearchOutlined />,
      label: 'Query',
      score: 58,
      title: 'Query precision',
      detail: percentText(Math.max(lowRelevance, lowInformation)) + ' weak evidence; tighten keywords or exclude generic mentions.'
    });
  }

  return gaps.sort((a, b) => b.score - a.score).slice(0, 3);
}

export default function IntelligenceView({ output, theme, setReadoutOpen }) {
  const readout = buildExecutiveReadout(output);
  const insights = readout.cards || [];
  const risks = readout.risks || [];
  const quality = signalQualitySummary(output);
  const freshness = signalFreshnessSummary(output);
  const warnings = signalWarnings(output);
  const sourceAge = Number(freshness.median_age_hours);
  const evidenceGaps = buildEvidenceGaps(readout, quality, freshness);
  const hasEvidence = numeric(readout.stats?.canonicalCount ?? quality.canonical_count) > 0
    || numeric(readout.stats?.rawCount ?? quality.raw_count) > 0;

  return (
    <motion.section key="intelligence" className="page-grid readout-grid" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
      <article className="span-7 studio-card lead-card compact-readout" onClick={() => setReadoutOpen(true)}>
        <SectionTitle eyebrow="Readout" title="Main Takeaway" action={<Tag color={readout.confidenceColor}>{readout.confidenceLabel}</Tag>} />
        <h3>{displayText(readout.headline, 'No signal yet')}</h3>
        <Button type="primary" ghost icon={<CheckCircleOutlined />}>Details</Button>
      </article>
      <aside className="span-5 studio-card risk-stack compact-risk">
        <SectionTitle eyebrow="Risk" title="Watch" />
        {risks.length ? risks.slice(0, 3).map((item, index) => (
          <motion.button className="risk-token" key={index} whileHover={{ y: -4, scale: 1.015 }} onClick={() => setReadoutOpen(true)}>
            <WarningOutlined />
            <span>
              <strong>{displayText(item?.tension, 'Open risk')}</strong>
              {item?.detail && <small>{displayText(item.detail, '')}</small>}
            </span>
          </motion.button>
        )) : <Empty description="Clear" />}
      </aside>
      <div className="span-12 insight-stack compact-insights">
        {insights.length ? insights.map((item, index) => <InsightCard key={index} item={item} index={index} theme={theme} />) : <Empty description="No insights" />}
      </div>
      <div className="span-7 studio-card">
        <SectionTitle eyebrow="Quality" title="Evidence Health" />
        <div className="signal-facts">
          <div><span>Distinct evidence</span><strong>{compactNumber(readout.stats?.canonicalCount ?? quality.canonical_count ?? 0)}</strong></div>
          <div><span>Repeated coverage</span><strong>{percentText(quality.amplification_ratio)}</strong></div>
          <div><span>Weak matches</span><strong>{percentText(quality.low_relevance_ratio)}</strong></div>
          <div><span>Source age</span><strong>{Number.isFinite(sourceAge) ? `${Math.round(sourceAge)} h` : 'n/a'}</strong></div>
        </div>
        {warnings.slice(0, 3).map((warning) => <Alert key={warning} type="warning" showIcon message={readableWarning(warning)} />)}
      </div>
      <aside className="span-5 studio-card evidence-gap-card">
        <SectionTitle eyebrow="Next Run" title="Evidence Gap" />
        <div className="evidence-gap-list">
          {evidenceGaps.length ? evidenceGaps.map((gap, index) => (
            <div className="evidence-gap-item" key={gap.key}>
              <span className="gap-icon">{gap.icon}</span>
              <span className="gap-copy">
                <strong>{index + 1}. {gap.title}</strong>
                <small>{gap.detail}</small>
              </span>
              <Tag>{gap.label}</Tag>
            </div>
          )) : (
            <div className="evidence-gap-item gap-clear">
              <span className="gap-icon"><CheckCircleOutlined /></span>
              <span className="gap-copy">
                <strong>{hasEvidence ? 'No major gap' : 'No evidence yet'}</strong>
                <small>{hasEvidence ? 'Current evidence is enough for the displayed readout.' : 'Run analysis to surface the next evidence gap.'}</small>
              </span>
              <Tag color="green">Clear</Tag>
            </div>
          )}
        </div>
      </aside>
    </motion.section>
  );
}
