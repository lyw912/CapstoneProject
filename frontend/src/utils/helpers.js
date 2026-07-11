import { MICRO_STEPS, FLOW_STEPS, LAST_QUERY_STORAGE_KEY } from './constants';

export function isObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value);
}

export function displayText(value, fallback = 'Text pending') {
  const text = String(value ?? '').trim();
  return text || fallback;
}

export function stripMarkdown(value) {
  return displayText(value, 'Reading pending')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
    .replace(/^#+\s+/gm, '')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

export function compactSummary(value, max = 86) {
  const text = stripMarkdown(value);
  return text.length > max ? `${text.slice(0, max).trim()}...` : text;
}

export function cleanSocialSnippet(value, fallback = 'Sample text pending') {
  const input = String(value ?? '').trim();
  if (!input) return fallback;
  const raw = stripMarkdown(input);
  let text = raw
    .replace(/Skip to main content/gi, ' ')
    .replace(/Image\s*\d+\s*:?\s*/gi, ' ')
    .replace(/\bGo to\s+[^.。!?]{1,80}[.。!?]?/gi, ' ')
    .replace(/\bu\/[A-Za-z0-9_-]+\s+avatar\b/gi, ' ')
    .replace(/\br\/[A-Za-z0-9_]+•\d+[hdwmy]\s+ago\b/gi, ' ')
    .replace(/\s*#\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const parts = text.match(/[^.!?。！？]+[.!?。！？]?/g) || [];
  if (parts.length > 1) {
    const seen = new Set();
    text = parts
      .map((part) => part.trim())
      .filter((part) => {
        const key = part.toLowerCase().replace(/[^a-z0-9\u4e00-\u9fa5]+/g, ' ').trim();
        if (!key || seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  return text || fallback;
}

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[ch]));
}

export function htmlFromText(value) {
  const text = String(value ?? '').trim();
  if (!text) return '';
  return text
    .split('\n')
    .filter(Boolean)
    .map((line) => `<p>${escapeHtml(line)}</p>`)
    .join('');
}

export function displayLog(value) {
  return String(value ?? '').replace(/[^\x00-\x7F]+/g, '[original-language text]');
}

export function clampPct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, Math.round(n * 100)));
}

export function percentText(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'Not rated';
  return `${Math.round(n * 100)}%`;
}

export function compactNumber(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '0';
  return new Intl.NumberFormat('en', { notation: n > 9999 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(n);
}

export function durationText(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 'No run yet';
  if (n < 60) return `${Math.round(n)} sec`;
  return `${Math.floor(n / 60)} min ${Math.round(n % 60)} sec`;
}

export function timeText(value) {
  if (!value) return 'No timestamp';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'No timestamp' : date.toLocaleString('en-US', { hour12: false });
}

export function msText(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 'No timing';
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60000) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)} sec`;
  return `${Math.floor(n / 60000)} min ${Math.round((n % 60000) / 1000)} sec`;
}

export function moneyText(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'Not metered';
  return `$${n.toFixed(n < 0.01 ? 4 : 2)}`;
}

export function urlDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return 'source';
  }
}

export function isHttpUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

export function containsCjk(value) {
  return /[\u3400-\u9fff]/.test(String(value ?? ''));
}

function englishSourceSummaryTitle(source, index) {
  const raw = cleanSocialSnippet(source?.title || source?.text || '', '');
  const platform = platformLabel(source?.platform || urlDomain(source?.url), true);
  const lower = raw.toLowerCase();
  const isHelp = /求解|请问|请教|怎么|如何|有没有|缓存命中率|命中率|教程|大佬|小白/.test(raw);
  const subject = lower.includes('deepseek') || raw.includes('DeepSeek') ? (lower.includes('v4') || raw.includes('V4') ? 'DeepSeek V4' : 'DeepSeek') : 'DeepSeek';
  const detailParts = [];
  if (/缓存命中率|命中率|cache hit/.test(raw)) detailParts.push('cache-hit rate');
  if (/高峰|峰谷|低峰|peak|off-peak/.test(raw)) detailParts.push('peak-hour API pricing');
  if (/降价|下调|便宜|price reduction|price cut/.test(raw)) detailParts.push('price reduction');
  if (/涨价|翻倍|price increase|price hike/.test(raw) && !detailParts.some((item) => item.includes('pricing'))) detailParts.push('price increase');
  if (/DSpark|推理|inference|效率|加速/.test(raw)) detailParts.push('inference efficiency');
  if (/API|api|接口|定价|价格|收费|计费|token/i.test(raw) && !detailParts.some((item) => item.includes('pricing'))) detailParts.push('API pricing');
  const topic = detailParts.length ? `${subject} ${detailParts.slice(0, 2).join(' and ')}` : `${subject} API pricing`;
  const relation = isHelp ? 'help request about' : String(source?.source_type || '').toLowerCase() === 'comment' ? 'comment about' : 'source on';
  return `${platform} ${relation} ${topic}` || `Source ${index + 1}`;
}

export function sourceTitle(source, index) {
  const fallback = `Source ${index + 1} from ${urlDomain(source?.url)}`;
  const cleaned = cleanSocialSnippet(source?.title, fallback);
  if (containsCjk(cleaned)) return englishSourceSummaryTitle(source, index);
  return displayText(cleaned, fallback);
}

const SOCIAL_PLATFORM_LABELS = {
  bilibili: 'Bilibili',
  douyin: 'Douyin',
  facebook: 'Facebook',
  instagram: 'Instagram',
  kuaishou: 'Kuaishou',
  reddit: 'Reddit',
  tiktok: 'TikTok',
  tieba: 'Tieba',
  weibo: 'Weibo',
  twitter: 'X / Twitter',
  xhs: 'Xiaohongshu',
  youtube: 'YouTube',
  zhihu: 'Zhihu'
};

const SOCIAL_PLATFORM_ALIASES = {
  bilibili: 'bilibili',
  'bilibili.com': 'bilibili',
  douyin: 'douyin',
  'douyin.com': 'douyin',
  'iesdouyin.com': 'douyin',
  facebook: 'facebook',
  'facebook.com': 'facebook',
  'fb.com': 'facebook',
  instagram: 'instagram',
  'instagram.com': 'instagram',
  kuaishou: 'kuaishou',
  'kuaishou.com': 'kuaishou',
  reddit: 'reddit',
  'reddit.com': 'reddit',
  'old.reddit.com': 'reddit',
  tieba: 'tieba',
  'tieba.baidu.com': 'tieba',
  tiktok: 'tiktok',
  'tiktok.com': 'tiktok',
  weibo: 'weibo',
  'weibo.com': 'weibo',
  'm.weibo.cn': 'weibo',
  x: 'twitter',
  'x.com': 'twitter',
  twitter: 'twitter',
  'twitter.com': 'twitter',
  't.co': 'twitter',
  xhs: 'xhs',
  xiaohongshu: 'xhs',
  'xiaohongshu.com': 'xhs',
  'xiaohongshu.cn': 'xhs',
  youtube: 'youtube',
  'youtube.com': 'youtube',
  'youtu.be': 'youtube',
  zhihu: 'zhihu',
  'zhihu.com': 'zhihu'
};

export function canonicalSocialPlatform(value) {
  const key = String(value || '').trim().toLowerCase().replace(/^www\./, '');
  if (!key) return '';
  if (SOCIAL_PLATFORM_ALIASES[key]) return SOCIAL_PLATFORM_ALIASES[key];
  const parts = key.split('.');
  for (let i = 0; i < parts.length; i += 1) {
    const candidate = parts.slice(i).join('.');
    if (SOCIAL_PLATFORM_ALIASES[candidate]) return SOCIAL_PLATFORM_ALIASES[candidate];
  }
  return '';
}

const QUALITY_WARNING_LABELS = {
  disputed: 'Conflicting evidence remains.',
  high_copy_ratio: 'Repeated wording may overstate independent support.',
  low_relevance: 'Some sources only weakly match the brief.',
  needs_more_evidence: 'More evidence is needed for stronger wording.',
  one_sided: 'The sampled evidence is one-sided.',
  single_source: 'Some findings rely on a narrow source base.',
  stale_evidence: 'Some sources may be stale.',
  ugc_only: 'Some findings rely only on user-generated content.'
};

const WORDING_POLICY_LABELS = {
  amplification_not_consensus: 'Repeated coverage checked',
  observable_sample_only: 'Observable sample',
  verified_sources_only: 'Verified sources only'
};

export function platformLabel(value, socialMode = false) {
  const raw = displayText(value, socialMode ? 'Platform' : 'Source').trim();
  const key = canonicalSocialPlatform(raw) || raw.toLowerCase().replace(/^www\./, '');
  if (socialMode && SOCIAL_PLATFORM_LABELS[key]) return SOCIAL_PLATFORM_LABELS[key];
  if (SOCIAL_PLATFORM_LABELS[key]) return SOCIAL_PLATFORM_LABELS[key];
  return raw;
}

export function readableClusterType(value) {
  const key = String(value || '').toLowerCase();
  if (key.includes('semantic') || key.includes('paraphrase')) return 'Similar meaning';
  if (key.includes('duplicate') || key.includes('near')) return 'Repeated wording';
  if (key.includes('original')) return 'Original item';
  if (key.includes('single')) return 'Single source';
  if (key.includes('mixed')) return 'Mixed sources';
  return displayText(value, 'Source group').replace(/_/g, ' ');
}

export function clusterTypeHint(value) {
  const key = String(value || '').toLowerCase();
  if (key.includes('semantic') || key.includes('paraphrase')) return 'Same topic or claim, different wording.';
  if (key.includes('duplicate') || key.includes('near')) return 'Near-identical wording repeated.';
  if (key.includes('original')) return 'One unique source item.';
  if (key.includes('single')) return 'Only one observed source item.';
  if (key.includes('mixed')) return 'Grouped from mixed source types.';
  return 'Evidence grouped by source similarity.';
}

export function readableWarning(value) {
  const key = String(value || '').toLowerCase();
  return QUALITY_WARNING_LABELS[key] || displayText(value, 'Evidence note').replace(/_/g, ' ');
}

export function readableWordingPolicy(value) {
  const key = String(value || '').toLowerCase();
  return WORDING_POLICY_LABELS[key] || displayText(value, 'Evidence scoped').replace(/_/g, ' ');
}

export function hasMindSpiderEvidence(output) {
  const artifact = signalArtifact(output);
  const retrievalResults = artifact?.evidence_graph?.retrieval_results || [];
  const diagnostics = artifact?.provider_diagnostics || [];
  return retrievalResults.some((item) => item?.provider === 'mindspider_db' && item?.status === 'ok' && Number(item?.items_returned || 0) > 0)
    || diagnostics.some((item) => item?.provider === 'mindspider_db' && item?.status === 'used' && Number(item?.metadata?.items || 0) > 0);
}

export function isSocialEvidence(item) {
  const sourceType = String(item?.source_type || '').toLowerCase();
  const platform = canonicalSocialPlatform(item?.platform || urlDomain(item?.url));
  return Boolean(platform) && ['ugc', 'comment', 'search_result'].includes(sourceType);
}

export function hasPlatformEvidence(output) {
  const graph = signalEvidenceGraph(output);
  return (graph?.evidence_items || []).some((item) => isSocialEvidence(item)) || hasMindSpiderEvidence(output);
}

function sourceBucketLabel(key) {
  return {
    social: 'Observable Social',
    official: 'Official Sources',
    media: 'News and Media',
    web: 'Other Web',
    replay: 'Replay Fixture'
  }[key] || displayText(key, 'Source Context');
}

function sourceBucketFor(item) {
  const sourceType = String(item?.source_type || '').toLowerCase();
  const platform = canonicalSocialPlatform(item?.platform || urlDomain(item?.url));
  if (sourceType === 'replay_fixture' || item?.acquisition_source === 'local_fixture') return 'replay';
  if (platform && (sourceType === 'ugc' || sourceType === 'comment' || sourceType === 'search_result')) return 'social';
  if (sourceType === 'official' || platform === 'official') return 'official';
  const rawPlatform = String(item?.platform || urlDomain(item?.url)).toLowerCase().replace(/^www\./, '');
  if (sourceType === 'mainstream_media' || rawPlatform === 'news' || rawPlatform.includes('news') || rawPlatform.includes('media')) return 'media';
  return 'web';
}

export function sourceGroupEntries(output, platformMode = false) {
  const graph = signalEvidenceGraph(output);
  const evidenceItems = graph?.evidence_items || [];
  const interpretations = output?.platform_interpretations || {};
  if (!platformMode) {
    const buckets = {};
    evidenceItems.forEach((item) => {
      const key = sourceBucketFor(item);
      const platform = displayText(item?.platform || urlDomain(item?.url), 'source');
      const bucket = buckets[key] || { key, count: 0, platforms: {} };
      bucket.count += 1;
      bucket.platforms[platform] = (bucket.platforms[platform] || 0) + 1;
      buckets[key] = bucket;
    });
    const order = ['social', 'official', 'media', 'web', 'replay'];
    return order
      .map((key) => buckets[key])
      .filter(Boolean)
      .map((bucket) => {
        const platforms = Object.entries(bucket.platforms)
          .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
          .map(([name, count]) => `${platformLabel(name)} (${compactNumber(count)})`)
          .join(', ');
        return {
          key: bucket.key,
          count: bucket.count,
          label: sourceBucketLabel(bucket.key),
          text: `${sourceBucketLabel(bucket.key)} account for ${compactNumber(bucket.count)} distinct evidence group${bucket.count === 1 ? '' : 's'}. ${platforms ? `Observed sources: ${platforms}.` : ''}`.trim()
        };
      });
  }
  const counts = {};
  const interpretationByPlatform = {};
  evidenceItems.forEach((item) => {
    if (!isSocialEvidence(item)) return;
    const key = canonicalSocialPlatform(item?.platform || urlDomain(item?.url)) || displayText(item?.platform || urlDomain(item?.url), 'source');
    counts[key] = (counts[key] || 0) + 1;
  });
  Object.entries(interpretations).forEach(([key, text]) => {
    const platformKey = canonicalSocialPlatform(key);
    if (!platformKey) return;
    if (!counts[platformKey]) counts[platformKey] = 0;
    if (!interpretationByPlatform[platformKey]) interpretationByPlatform[platformKey] = text;
  });
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([key, count]) => ({
      key,
      count,
      label: platformLabel(key, true),
      text: interpretationByPlatform[key] || `${platformLabel(key)} contributed ${compactNumber(count)} sampled source${count === 1 ? '' : 's'} to this run.`
    }));
}

export function spanLabelIndex(graph) {
  const labels = {};
  let sourceNumber = 1;
  (graph?.evidence_items || []).forEach((item) => {
    const label = `Source ${sourceNumber}`;
    sourceNumber += 1;
    (item?.spans || []).forEach((span) => {
      if (span?.span_id) {
        labels[span.span_id] = {
          label,
          title: sourceTitle(item, sourceNumber - 2),
          text: span?.text || item?.text || ''
        };
      }
    });
  });
  return labels;
}

export function signalArtifact(output) {
  return output?.coordinator_intelligence
    || output?.synthesis_context?.coordinator_intelligence
    || output?.signal_intelligence
    || output?.synthesis_context?.signal_intelligence
    || null;
}

export function signalGraphSummary(output) {
  return signalArtifact(output)?.evidence_graph_summary || {};
}

export function signalEvidenceGraph(output) {
  return signalArtifact(output)?.evidence_graph || {};
}

export function signalQualitySummary(output) {
  return signalArtifact(output)?.quality_summary || output?.source_data?.query_agent?.quality_summary || {};
}

export function signalFreshnessSummary(output) {
  return signalArtifact(output)?.freshness_summary || output?.source_data?.query_agent?.freshness_summary || {};
}

export function signalProviderDiagnostics(output) {
  return signalArtifact(output)?.provider_diagnostics || [];
}

function strengthConfidence(strength) {
  const value = String(strength || '').toLowerCase();
  if (value === 'strong') return 0.86;
  if (value === 'moderate') return 0.68;
  if (value === 'weak') return 0.48;
  if (value === 'uncertain') return 0.32;
  return undefined;
}

export function signalInsights(output) {
  const artifact = signalArtifact(output);
  const graphInsights = artifact?.insights || artifact?.evidence_graph?.insights || [];
  if (Array.isArray(graphInsights) && graphInsights.length) {
    return graphInsights.map((item) => ({
      insight: item?.insight || item?.title,
      basis: item?.basis || item?.body,
      confidence: item?.confidence ?? strengthConfidence(item?.strength),
      claim_ids: item?.claim_ids || [],
      citation_spans: item?.citation_spans || [],
      quality_warnings: item?.quality_warnings || [],
      wording_policy: item?.wording_policy
    }));
  }
  return output?.synthesis?.top_insights || [];
}

export function signalWarnings(output) {
  const artifact = signalArtifact(output);
  return artifact?.analysis_warnings || artifact?.source_coverage_limitations || signalQualitySummary(output)?.quality_warnings || [];
}

export function reportSeedHtml(output) {
  const noise = /(audited evidence contains|claims address the same aspect|distinct source groups|claim-level review|repeated coverage|flag_uncertain)/i;
  const graph = signalEvidenceGraph(output);
  const claims = graph.claims || [];
  const quality = signalQualitySummary(output);
  const pricingClaims = claims.filter((claim) => claim?.aspect === "pricing" && !["demoted", "unsupported"].includes(String(claim?.status || "").toLowerCase()));
  const supportOrOfficial = pricingClaims.filter((claim) => ["support", "official"].includes(String(claim?.stance || "").toLowerCase())).length;
  const negative = pricingClaims.filter((claim) => String(claim?.stance || "").toLowerCase() === "oppose").length;
  const officialDomains = new Set((graph.evidence_items || [])
    .map((item) => urlDomain(item?.url || "").toLowerCase())
    .filter((host) => host === "deepseek.com" || host.endsWith(".deepseek.com")));
  const rawSummary = displayText(output?.synthesis?.summary, "");
  const summary = noise.test(rawSummary) || !rawSummary
    ? `DeepSeek API pricing is supported by ${officialDomains.size} official DeepSeek domain${officialDomains.size === 1 ? "" : "s"} and ${pricingClaims.length} usable sampled pricing claim${pricingClaims.length === 1 ? "" : "s"}.`
    : rawSummary;
  const cleanInsights = signalInsights(output)
    .filter((item) => !noise.test(`${item?.insight || ""} ${item?.basis || ""}`))
    .slice(0, 4);
  const fallbackInsights = [
    { insight: "Official pricing evidence is present", basis: `${Array.from(officialDomains).join(", ") || "No official DeepSeek domain"} appears in the evidence set.` },
    { insight: negative > 0 ? "Sampled reactions are mixed" : "Pricing discussion is visible in the sample", basis: `${supportOrOfficial} support/official pricing signal${supportOrOfficial === 1 ? "" : "s"}; ${negative} negative pricing signal${negative === 1 ? "" : "s"}.` },
    { insight: "Duplicate wording is not independent agreement", basis: `${compactNumber(quality.raw_count)} raw item${Number(quality.raw_count) === 1 ? "" : "s"} were reduced to ${compactNumber(quality.canonical_count)} distinct evidence group${Number(quality.canonical_count) === 1 ? "" : "s"}.` }
  ];
  const insights = cleanInsights.length ? cleanInsights : fallbackInsights;
  const tensions = (output?.synthesis?.key_tensions || [])
    .filter((item, index, arr) => !noise.test(`${item?.tension || ""} ${item?.significance || ""}`) && arr.findIndex((other) => other?.tension === item?.tension) === index)
    .slice(0, 4);
  return `
    <h2>Executive Brief</h2>
    ${htmlFromText(summary)}
    <h2>Priority Insights</h2>
    ${insights.map((item) => `<h3>${escapeHtml(displayText(item.insight, "Insight pending"))}</h3><p>${escapeHtml(displayText(item.basis, "Evidence basis pending"))}</p>`).join("") || "<p>Insights will appear after analysis.</p>"}
    <h2>Open Tensions</h2>
    ${tensions.map((item) => `<p><strong>${escapeHtml(displayText(item.tension, "Tension pending"))}</strong><br>${escapeHtml(displayText(item.significance, "Review cited sources before drawing a broad conclusion."))}</p>`).join("") || "<p>Risks are handled in the Readout and Evidence views.</p>"}
  `;
}

export const SENSITIVE_INPUT_MESSAGE =
  'Your input contains blocked terms, so the report cannot be generated. Please revise the topic and try again.';

export function isSensitiveInputError(error) {
  return error?.code === 'sensitive_input' || error?.payload?.error_code === 'sensitive_input';
}

export function showSensitiveInputModal() {
  import('antd').then(({ Modal }) => {
    Modal.warning({
      title: 'Request Blocked',
      content: SENSITIVE_INPUT_MESSAGE,
      okText: 'OK',
      centered: true
    });
  });
}

export async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { success: false, error: text || 'Request failed' };
  }
  if (!response.ok) {
    const error = new Error(payload.message || payload.error || `Request failed with ${response.status}`);
    error.code = payload.error_code || '';
    error.payload = payload;
    throw error;
  }
  return payload;
}

export function downloadBlob(content, filename, type = 'text/html') {
  const blob = new Blob([content], { type });
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(href);
}

export function flowProgressMeta(progress) {
  const normalized = Math.max(0, Math.min(100, Number(progress) || 0));
  const microRaw = normalized >= 100 ? MICRO_STEPS.length - 1 : Math.floor((normalized / 100) * MICRO_STEPS.length);
  const microIndex = Math.max(0, Math.min(MICRO_STEPS.length - 1, microRaw));
  const micro = MICRO_STEPS[microIndex] || MICRO_STEPS[0];
  const stage = FLOW_STEPS[micro.stageIndex] || FLOW_STEPS[0];
  const stageStart = MICRO_STEPS.findIndex((item) => item.stageIndex === micro.stageIndex);
  const stageCount = stage.micro.length || 1;
  const localIndex = Math.max(0, microIndex - stageStart);
  return {
    normalized,
    micro,
    microIndex,
    activeIndex: micro.stageIndex,
    stage,
    stagePercent: Math.round(((localIndex + 1) / stageCount) * 100)
  };
}

export function langSmithProjectUrl(observability) {
  const endpoint = String(observability?.endpoint || 'https://smith.langchain.com').replace(/\/$/, '');
  const project = observability?.project;
  if (!project) return 'https://smith.langchain.com/';
  return `${endpoint.replace('api.smith.langchain.com', 'smith.langchain.com')}/o/default/projects/p/${encodeURIComponent(project)}`;
}

export function traceStatus(run) {
  if (run?.error) return 'error';
  const status = String(run?.status || '').toLowerCase();
  if (status.includes('error') || status.includes('fail')) return 'error';
  if (status.includes('run') || status.includes('pending')) return 'running';
  return 'success';
}

export function traceMetricRows(observabilityTrace) {
  const summary = observabilityTrace?.summary || {};
  return [
    { key: 'trace_count', label: 'Traces', value: compactNumber(summary.trace_count) },
    { key: 'run_count', label: 'Steps', value: compactNumber(summary.run_count) },
    { key: 'avg_duration_ms', label: 'Avg time', value: msText(summary.avg_duration_ms) },
    { key: 'error_count', label: 'Diagnostics', value: compactNumber(summary.error_count) },
    { key: 'total_tokens', label: 'Tokens', value: compactNumber(summary.total_tokens) },
    { key: 'total_cost', label: 'Cost', value: moneyText(summary.total_cost) }
  ];
}

export function traceTimelineChartData(observabilityTrace) {
  return (observabilityTrace?.timeline || []).slice().reverse().map((run, index) => ({
    name: `T${index + 1}`,
    duration: Math.round(Number(run.duration_ms) || 0),
    steps: Array.isArray(run.children) ? run.children.length : 0,
    errors: traceStatus(run) === 'error' ? 1 : 0
  }));
}

export function traceChildrenByType(children = []) {
  const counts = {};
  children.forEach((child) => {
    const key = child.type || 'unknown';
    counts[key] = (counts[key] || 0) + 1;
  });
  return Object.entries(counts).map(([name, value]) => ({ name, value }));
}

export function readLastQuery() {
  try {
    return window.localStorage.getItem(LAST_QUERY_STORAGE_KEY) || '';
  } catch {
    return '';
  }
}

export function persistLastQuery(value) {
  try {
    window.localStorage.setItem(LAST_QUERY_STORAGE_KEY, String(value ?? ''));
  } catch {
    return undefined;
  }
}
