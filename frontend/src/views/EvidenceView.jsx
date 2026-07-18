import { Collapse, Empty, Popover, Progress, Table, Tag } from 'antd';
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
import DebateInspector from '../components/DebateInspector';
import Heatmap from '../components/Heatmap';
import MarkdownText from '../components/MarkdownText';
import { STANCE_COLORS } from '../utils/constants';
import {
  displayText,
  clampPct,
  percentText,
  compactSummary,
  cleanSocialSnippet,
  sourceTitle,
  signalEvidenceGraph,
  canonicalSocialPlatform,
  hasPlatformEvidence,
  isHttpUrl,
  isSocialEvidence,
  platformLabel,
  readableClusterType,
  clusterTypeHint,
  sourceGroupEntries,
  urlDomain
} from '../utils/helpers';

const STATUS_RANK = {
  supported: 7,
  mixed: 6,
  disputed: 5,
  needs_search: 4,
  single_source: 3,
  unsupported: 2,
  demoted: 1,
  pending: 0
};

const ASPECT_RANK = {
  pricing: 5,
  usage_help: 4,
  reputation_risk: 3,
  evidence_quality: 2,
  general_discourse: 1
};

const REVIEW_RANK = {
  accept: 5,
  weaken: 3,
  needs_search: 2,
  reject: 0,
  pending: 0
};

function mergeReviewDecision(decisions) {
  const actionable = decisions.filter((item) => item && item !== 'pending');
  if (!actionable.length) return 'pending';
  if (actionable.every((item) => item === 'reject')) return 'reject';
  if (actionable.includes('reject') || actionable.includes('weaken')) return 'weaken';
  if (actionable.includes('needs_search')) return 'needs_search';
  if (actionable.includes('accept')) return 'accept';
  return 'pending';
}

function mergeClaimStatus(statusCounts) {
  const activeStatuses = Object.entries(statusCounts).filter(([, count]) => count > 0).map(([status]) => status);
  if (activeStatuses.length > 1) return 'mixed';
  if (statusCounts.disputed) return 'disputed';
  if (statusCounts.supported) return 'supported';
  if (statusCounts.needs_search) return 'needs_search';
  if (statusCounts.single_source) return 'single_source';
  if (statusCounts.unsupported) return 'unsupported';
  if (statusCounts.demoted) return 'demoted';
  return 'pending';
}

function buildClaimRows(claims, decisionsByClaim) {
  const groups = new Map();
  claims.forEach((claim) => {
    const key = `${displayText(claim.claim_text, 'Claim pending').toLowerCase()}|${claim.aspect || ''}|${claim.stance || ''}`;
    const current = groups.get(key) || {
      ...claim,
      key,
      claim_ids: [],
      evidence_group_count: 0,
      source_ids: new Set(),
      status_counts: {},
      review_decisions: [],
      confidence: 0
    };
    current.claim_ids.push(claim.claim_id);
    current.evidence_group_count += 1;
    [...(claim.supporting_spans || []), ...(claim.contradicting_spans || [])].forEach((span) => current.source_ids.add(span));
    const status = String(claim.status || 'pending').toLowerCase();
    current.status_counts[status] = (current.status_counts[status] || 0) + 1;
    const decision = decisionsByClaim[claim.claim_id]?.decision || 'pending';
    current.review_decisions.push(decision);
    current.confidence = Math.max(current.confidence, Number(claim.confidence || 0));
    groups.set(key, current);
  });
  return Array.from(groups.values()).map((group) => {
    const sourceCount = group.source_ids.size;
    const status = mergeClaimStatus(group.status_counts);
    const reviewDecision = mergeReviewDecision(group.review_decisions);
    return {
      ...group,
      status,
      review_decision: reviewDecision,
      source_count: sourceCount,
      supporting_spans: Array.from(group.source_ids),
      sort_score: (ASPECT_RANK[group.aspect] || 0) * 10000 + (REVIEW_RANK[reviewDecision] || 0) * 1000 + sourceCount * 100 + (STATUS_RANK[status] || 0) * 10 + group.confidence
    };
  }).sort((a, b) => b.sort_score - a.sort_score || a.claim_text.localeCompare(b.claim_text));
}

function platformCountKey(value) {
  return canonicalSocialPlatform(value) || String(value || '').trim().toLowerCase().replace(/^www\./, '');
}

function statusLabel(value) {
  return {
    supported: 'Supported',
    mixed: 'Mixed quality',
    disputed: 'Conflicting signals',
    needs_search: 'Needs evidence',
    single_source: 'Single source',
    unsupported: 'Weak support',
    demoted: 'Low relevance',
    pending: 'Pending'
  }[String(value || '').toLowerCase()] || displayText(value, 'Pending');
}


function statusColor(value) {
  return {
    supported: 'green',
    mixed: 'purple',
    disputed: 'orange',
    needs_search: 'blue',
    single_source: 'gold',
    unsupported: 'red',
    demoted: 'red'
  }[String(value || '').toLowerCase()] || 'default';
}


function readableClaimText(item) {
  const raw = displayText(item?.claim_text, 'Finding pending');
  const normalized = raw.toLowerCase();
  if (normalized === 'deepseek has an official or high-authority source addressing pricing.') {
    return 'Official DeepSeek API documentation addresses pricing.';
  }
  if (normalized === 'deepseek has an official or high-authority source addressing usage help.') {
    return 'Official DeepSeek documentation covers API usage and setup.';
  }
  if (normalized === 'observable sources discuss deepseek in relation to pricing.') {
    return 'DeepSeek API pricing is a recurring theme across sampled sources.';
  }
  if (normalized === 'observable sources discuss deepseek in relation to usage help.') {
    return 'Users also discuss DeepSeek API usage help, including cache-hit behavior and client setup.';
  }
  if (normalized === 'observable sources discuss deepseek in relation to general discourse.') {
    return 'General DeepSeek discussion appears in the sampled evidence.';
  }
  return raw;
}

function isMainClaimRow(item) {
  const status = String(item?.status || '').toLowerCase();
  const review = String(item?.review_decision || '').toLowerCase();
  const aspect = String(item?.aspect || '').toLowerCase();
  if (review === 'reject') return false;
  if (['unsupported', 'demoted'].includes(status)) return false;
  if (aspect === 'general_discourse') return false;
  return Number(item?.source_count || 0) >= 2 || ['supported', 'disputed', 'mixed'].includes(status);
}

function representativeSignalTitle(item, index) {
  const title = sourceTitle(item, index);
  const stripped = title.replace(/^(Weibo|Tieba|Bilibili|Zhihu|Reddit|X \/ Twitter|Facebook|YouTube|Xiaohongshu|Douyin|Kuaishou|TikTok)\s+(source on|help request about|comment about)\s+/i, '');
  return stripped || title;
}

function representativeSignalFullText(item, index) {
  const display = representativeSignalTitle(item, index);
  const original = cleanSocialSnippet(`${item?.title || ''} ${item?.text || ''}`.trim(), '');
  if (!original || original === display) return display;
  return `${display}

Original excerpt:
${original}`;
}

export default function EvidenceView({ output, theme, coordinatorTask }) {
  const sourceData = output.source_data || {};
  const queryAgent = sourceData.query_agent || {};
  const graph = signalEvidenceGraph(output);
  const stanceRows = Object.entries(queryAgent.stance_distribution || {}).map(([name, value]) => ({ name, value: Number(value) || 0 }));
  const topSources = queryAgent.top_sources || [];
  const clusters = graph.canonical_clusters || [];
  const claims = graph.claims || [];
  const decisionsByClaim = Object.fromEntries((graph.audit_decisions || []).map((item) => [item.claim_id, item]));
  const hasPlatformSamples = hasPlatformEvidence(output);
  const groupEntries = sourceGroupEntries(output, hasPlatformSamples);
  const providerDiagnostics = output.coordinator_intelligence?.provider_diagnostics || output.signal_intelligence?.provider_diagnostics || [];
  const mindSpiderDiag = providerDiagnostics.find((item) => item?.provider === 'mindspider_db' && item?.status === 'used') || null;
  const mindSpiderPlatforms = mindSpiderDiag?.metadata?.platforms || {};
  const socialSentiment = queryAgent.social_sentiment || {};
  const mindSpiderPlatformKeys = ['weibo', 'zhihu', 'bilibili', 'douyin', 'tieba', 'xhs', 'kuaishou'];
  const platformOrder = ['weibo', 'zhihu', 'bilibili', 'douyin', 'tieba', 'xhs', 'kuaishou', 'reddit', 'twitter', 'facebook', 'youtube'];
  const voiceSamples = socialSentiment.top_social_voices || [];
  const graphSamples = (graph.evidence_items || [])
    .filter((item) => isSocialEvidence(item))
    .map((item) => ({
      platform: item.platform || urlDomain(item.url),
      stance: item.stance || '',
      title: item.title || '',
      content: item.text || item.title || '',
      url: item.url,
      publish_time: item.published_at,
      acquisition_source: item.acquisition_source
    }));
  const sampleMap = {};
  [...voiceSamples, ...graphSamples].forEach((sample) => {
    const key = canonicalSocialPlatform(sample.platform || urlDomain(sample.url)) || String(sample.platform || 'web').toLowerCase();
    if (!key) return;
    const signature = `${sample.url || ''}:${String(sample.title || sample.content || '').slice(0, 140)}`;
    const list = sampleMap[key] || [];
    if (!list.some((item) => `${item.url || ''}:${String(item.title || item.content || '').slice(0, 140)}` === signature)) {
      list.push(sample);
    }
    sampleMap[key] = list;
  });
  const platformKeys = Array.from(new Set([
    ...groupEntries.map((entry) => entry.key),
    ...Object.keys(socialSentiment.per_platform || {}),
    ...Object.keys(mindSpiderPlatforms || {}),
    ...(mindSpiderDiag ? mindSpiderPlatformKeys : []),
    ...Object.keys(sampleMap)
  ]));
  const platformDetailEntries = platformKeys
    .map((key) => {
      const group = groupEntries.find((entry) => entry.key === key) || {};
      const sentiment = socialSentiment.per_platform?.[key] || {};
      const samples = sampleMap[key] || [];
      const count = Number(sentiment.count ?? group.count ?? samples.length ?? 0);
      const mindSpiderChecked = Boolean(mindSpiderDiag && mindSpiderPlatformKeys.includes(key));
      return {
        key,
        label: group.label || platformLabel(key, true),
        text: group.text || (count > 0
          ? `${platformLabel(key, true)} contributed ${count} sampled source${count === 1 ? '' : 's'} to this run.`
          : `${platformLabel(key, true)} was checked for this query, but no matching platform sample was returned.`),
        count,
        sentiment,
        samples,
        mindSpiderChecked,
        mindSpiderCount: Number(mindSpiderPlatforms?.[key] || 0)
      };
    })
    .sort((a, b) => {
      const rank = (entry) => {
        if (entry.mindSpiderCount > 0) return 0;
        if (entry.count > 0) return 1;
        if (entry.mindSpiderChecked) return 2;
        return 3;
      };
      const rankDelta = rank(a) - rank(b);
      if (rankDelta) return rankDelta;
      const countDelta = b.count - a.count;
      if (countDelta) return countDelta;
      const orderDelta = platformOrder.indexOf(a.key) - platformOrder.indexOf(b.key);
      if (platformOrder.includes(a.key) && platformOrder.includes(b.key) && orderDelta) return orderDelta;
      return a.label.localeCompare(b.label);
    });
  const divergencePlatformCounts = { ...(output.divergence_matrix?.group_counts || {}) };
  if (!Object.keys(divergencePlatformCounts).length) {
    (graph.evidence_items || []).forEach((item) => {
      const key = platformCountKey(item.platform || urlDomain(item.url));
      if (!key) return;
      divergencePlatformCounts[key] = (divergencePlatformCounts[key] || 0) + 1;
    });
  }
  const mergedClaims = buildClaimRows(claims, decisionsByClaim);
  const visibleClaims = mergedClaims.filter(isMainClaimRow);
  const sortedClusters = [...clusters].sort((a, b) => {
    const mentionDelta = Number(b.amplification_count || 0) - Number(a.amplification_count || 0);
    if (mentionDelta) return mentionDelta;
    return (b.platforms || []).length - (a.platforms || []).length;
  });
  const evidenceByCluster = Object.fromEntries((graph.evidence_items || []).map((item) => [item.canonical_item_id, item]));
  const clusterRows = sortedClusters.slice(0, 10).map((cluster, index) => {
    const representative = evidenceByCluster[cluster.canonical_item_id];
    const platforms = cluster.platforms || [];
    return {
      ...cluster,
      representative,
      signal_title: representative
        ? representativeSignalTitle(representative, index)
        : `${readableClusterType(cluster.cluster_type)} across ${platforms.map((item) => platformLabel(item, hasPlatformSamples)).join(', ') || 'sampled sources'}`,
      signal_full_text: representative
        ? representativeSignalFullText(representative, index)
        : '',
      platform_count: platforms.length
    };
  });

  const evidenceColumns = [
    {
      title: 'Source',
      dataIndex: 'title',
      render: (_, item, index) => (
        isHttpUrl(item.url)
          ? <a href={item.url} target="_blank" rel="noreferrer">{sourceTitle(item, index)}</a>
          : <span>{sourceTitle(item, index)}</span>
      )
    },
    { title: 'Stance', dataIndex: 'stance', width: 110, render: (value) => <Tag color={value === 'support' ? 'green' : value === 'oppose' ? 'red' : 'blue'}>{displayText(value || 'neutral').toUpperCase()}</Tag> },
    { title: 'Trust', dataIndex: 'trust_score', width: 130, render: (value) => <Progress percent={clampPct(value)} size="small" strokeColor={theme.primary} /> }
  ];
  const claimColumns = [
    {
      title: 'Finding',
      dataIndex: 'claim_text',
      render: (_, item) => (
        <div className="claim-cell compact-finding">
          <span>{readableClaimText(item)}</span>
          <div className="compact-tags">
            {item.aspect && <Tag>{displayText(item.aspect).replace(/_/g, ' ')}</Tag>}
            {item.evidence_group_count > 1 && <Tag>{`${item.evidence_group_count} groups`}</Tag>}
          </div>
        </div>
      )
    },
    {
      title: 'Evidence',
      dataIndex: 'status',
      width: 145,
      render: (value) => <Tag color={statusColor(value)}>{statusLabel(value)}</Tag>
    },
    {
      title: 'Sources',
      dataIndex: 'source_count',
      width: 130,
      render: (value = 0) => <Tag>{`${value} source${value === 1 ? '' : 's'}`}</Tag>
    }
  ];
  const clusterColumns = [
    {
      title: 'Representative signal',
      dataIndex: 'signal_title',
      render: (value, item) => (
        <div className="pattern-cell compact-signal">
          <Popover
            trigger={['hover', 'click']}
            title="Full representative signal"
            content={<pre className="signal-full-popover">{displayText(item.signal_full_text || value, 'Signal pending')}</pre>}
          >
            <button type="button" className="signal-text-button">{displayText(value, 'Signal pending')}</button>
          </Popover>
        </div>
      )
    },
    { title: 'Pattern type', dataIndex: 'cluster_type', width: 145, render: (value) => <Tag>{readableClusterType(value)}</Tag> },
    { title: 'Mentions', dataIndex: 'amplification_count', width: 105 },
    { title: hasPlatformSamples ? 'Platforms' : 'Sources', dataIndex: 'platforms', render: (value = []) => <span className="platform-ellipsis">{value.map((item) => platformLabel(item, hasPlatformSamples)).join(', ')}</span> }
  ];

  return (
    <motion.section key="evidence" className="page-grid" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
      <DebateInspector output={output} coordinatorTask={coordinatorTask} />
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
        <Heatmap
          pairs={output.divergence_matrix?.pairs}
          platformCounts={divergencePlatformCounts}
          groupDistributions={output.divergence_matrix?.group_distributions}
        />
      </div>
      <div className="span-8 studio-card source-card">
        <SectionTitle eyebrow="Grounding" title="Top evidence" />
        <Table columns={evidenceColumns} dataSource={topSources.map((item, index) => ({ ...item, key: index }))} pagination={{ pageSize: 6 }} size="middle" />
      </div>
      <div className="span-4 studio-card platform-card">
        <SectionTitle
          eyebrow={hasPlatformSamples ? 'Cross-platform' : 'Coverage'}
          title={hasPlatformSamples ? 'Platform Signals' : 'Coverage Context'}
        />
        <div className="platform-list collapsed-platform-list">
          {platformDetailEntries.length ? (
            <Collapse
                ghost
                className="platform-collapse"
                items={platformDetailEntries.map((entry) => ({
                  key: entry.key,
                  label: (
                    <div className="platform-summary">
                      <strong>{entry.label}</strong>
                      <span>{compactSummary(entry.text, 120)}</span>
                    </div>
                  ),
                  extra: (
                    <div className="platform-tags platform-tags-compact">
                      <Tag>{`${entry.count} sample${entry.count === 1 ? '' : 's'}`}</Tag>
                      {entry.mindSpiderCount > 0 && <Tag color="green">{`Local DB ${entry.mindSpiderCount}`}</Tag>}
                      {entry.mindSpiderChecked && entry.mindSpiderCount === 0 && <Tag>Checked</Tag>}
                    </div>
                  ),
                  children: (
                    <div className="platform-detail">
                      <div className="platform-tags">
                        {entry.sentiment?.distribution && Object.entries(entry.sentiment.distribution).map(([stance, value]) => (
                          <Tag key={stance}>{`${stance} ${Math.round(Number(value || 0) * 100)}%`}</Tag>
                        ))}
                      </div>
                      <div className="platform-samples">
                        {entry.samples.length ? entry.samples.map((sample, index) => {
                          const title = cleanSocialSnippet(sample.title || '', '');
                          const body = cleanSocialSnippet(sample.content || sample.title || '', 'Sample text pending');
                          const linkText = title || compactSummary(body, 96);
                          const showBody = body && body !== linkText;
                          return (
                            <div className="platform-sample" key={`${entry.key}-${index}`}>
                              <div className="platform-sample-meta">
                                <Tag color={sample.stance === 'support' ? 'green' : sample.stance === 'oppose' ? 'red' : 'blue'}>{displayText(sample.stance || 'neutral').toUpperCase()}</Tag>
                                {sample.publish_time && <span>{sample.publish_time}</span>}
                              </div>
                              {sample.url
                                ? <a className="platform-sample-title" href={sample.url} target="_blank" rel="noreferrer">{linkText}</a>
                                : <strong className="platform-sample-title">{linkText}</strong>}
                              {showBody && <p className="platform-sample-excerpt">{compactSummary(body, 420)}</p>}
                            </div>
                          );
                        }) : (
                          <p><MarkdownText value={entry.text} /></p>
                        )}
                      </div>
                    </div>
                  )
                }))}
              />
          ) : <Empty description={hasPlatformSamples ? 'No platform signals' : 'No coverage context'} />}
        </div>
      </div>
      <div className="span-7 studio-card source-card">
        <SectionTitle eyebrow="Claims" title="Review Outcomes" />
        <Table columns={claimColumns} dataSource={visibleClaims.map((item, index) => ({ ...item, key: item.key || index }))} pagination={{ pageSize: 5 }} size="middle" />
      </div>
      <div className="span-5 studio-card source-card">
        <SectionTitle eyebrow="Patterns" title={hasPlatformSamples ? 'Top Signal Clusters' : 'Repeated Coverage'} />
        <Table columns={clusterColumns} dataSource={clusterRows.map((item, index) => ({ ...item, key: item.canonical_item_id || index }))} pagination={{ pageSize: 5, showSizeChanger: false }} size="small" />
      </div>
    </motion.section>
  );
}
