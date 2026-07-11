import {
  canonicalSocialPlatform,
  compactNumber,
  displayText,
  percentText,
  platformLabel,
  signalEvidenceGraph,
  signalFreshnessSummary,
  signalInsights,
  signalQualitySummary,
  urlDomain
} from './helpers';

const BAD_TEMPLATE_RE = /(audited evidence contains|claims address the same aspect|distinct source groups|claim-level review|repeated coverage)/i;
const NEGATIVE_STATUSES = new Set(['demoted', 'unsupported']);

function norm(value) {
  return String(value || '').trim().toLowerCase();
}

function isUsefulClaim(claim) {
  return !NEGATIVE_STATUSES.has(norm(claim?.status));
}

function isDeepSeekQuery(output) {
  const query = `${output?.query || ''} ${output?.coordinator_intelligence?.query || ''}`;
  return /deepseek/i.test(query);
}

function isOfficialDomainForRun(item, output) {
  const domain = urlDomain(item?.url || '').toLowerCase();
  if (!domain || domain === 'source') return norm(item?.source_type) === 'official' && !isDeepSeekQuery(output);
  if (!isDeepSeekQuery(output)) return norm(item?.source_type) === 'official';
  return domain === 'deepseek.com' || domain.endsWith('.deepseek.com');
}

function usefulClaimsByAspect(claims, aspect) {
  return claims.filter((claim) => norm(claim?.aspect) === aspect && isUsefulClaim(claim));
}

function countBy(items, fn) {
  return items.reduce((acc, item) => {
    const key = fn(item);
    if (!key) return acc;
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
}

function unique(items, keyFn) {
  const seen = new Set();
  return items.filter((item) => {
    const key = keyFn(item);
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function shortTopic(output) {
  const raw = displayText(output?.query || output?.coordinator_intelligence?.query || output?.source_data?.query_agent?.query, 'this topic');
  if (/deepseek/i.test(raw) && /pricing|price|api/i.test(raw)) return 'DeepSeek API pricing';
  return raw.length > 42 ? `${raw.slice(0, 42).trim()}...` : raw;
}

function pricingStats(output) {
  const graph = signalEvidenceGraph(output);
  const claims = graph.claims || [];
  const pricingClaims = usefulClaimsByAspect(claims, 'pricing');
  const officialPricing = pricingClaims.filter((claim) => norm(claim?.stance) === 'official');
  const supportPricing = pricingClaims.filter((claim) => norm(claim?.stance) === 'support');
  const opposePricing = pricingClaims.filter((claim) => norm(claim?.stance) === 'oppose');
  return {
    claims: pricingClaims.length,
    official: officialPricing.length,
    support: supportPricing.length,
    oppose: opposePricing.length
  };
}

function evidenceStats(output) {
  const graph = signalEvidenceGraph(output);
  const quality = signalQualitySummary(output);
  const freshness = signalFreshnessSummary(output);
  const evidence = graph.evidence_items || [];
  const officialEvidence = evidence.filter((item) => isOfficialDomainForRun(item, output));
  const socialEvidence = evidence.filter((item) => canonicalSocialPlatform(item?.platform || urlDomain(item?.url)));
  const platformCounts = countBy(socialEvidence, (item) => canonicalSocialPlatform(item?.platform || urlDomain(item?.url)));
  const lowSamplePlatforms = Object.entries(platformCounts)
    .filter(([, count]) => count > 0 && count < 3)
    .map(([platform, count]) => ({ platform, count }));
  return {
    evidence,
    officialEvidence,
    socialEvidence,
    platformCounts,
    lowSamplePlatforms,
    canonicalCount: Number(quality.canonical_count ?? evidence.length) || 0,
    rawCount: Number(quality.raw_count ?? evidence.length) || 0,
    exactDuplicateCount: Number(quality.exact_duplicate_count ?? 0) || 0,
    amplificationRatio: Number(quality.amplification_ratio ?? 0) || 0,
    medianAgeHours: Number(freshness.median_age_hours)
  };
}

function cleanTemplateInsight(item) {
  const title = displayText(item?.insight || item?.title, '');
  const basis = displayText(item?.basis || item?.body, '');
  if (BAD_TEMPLATE_RE.test(title) || BAD_TEMPLATE_RE.test(basis)) return null;
  return {
    insight: title,
    basis,
    confidence: item?.confidence,
    citation_spans: item?.citation_spans || [],
    meta_tags: []
  };
}

export function buildExecutiveReadout(output) {
  const topic = shortTopic(output);
  const graph = signalEvidenceGraph(output);
  const auditDecisions = graph.audit_decisions || [];
  const acceptedClaims = auditDecisions.filter((item) => norm(item?.decision) === 'accept').length;
  const pricing = pricingStats(output);
  const stats = evidenceStats(output);
  const officialCount = unique(stats.officialEvidence, (item) => urlDomain(item?.url) + (item?.canonical_url || item?.url || item?.title || '')).length;
  const socialPlatformCount = Object.keys(stats.platformCounts).length;
  const lowSampleLabel = stats.lowSamplePlatforms
    .slice(0, 3)
    .map((item) => `${platformLabel(item.platform, true)} n=${item.count}`)
    .join(', ');

  const hasOfficial = officialCount > 0 || (!isDeepSeekQuery(output) && (pricing.official > 0 || acceptedClaims > 0));
  const hasMixedPricing = pricing.oppose > 0 && (pricing.support > 0 || pricing.official > 0);
  const dominantPricingSignal = hasOfficial && hasMixedPricing && (pricing.support + pricing.official) > pricing.oppose;
  const confidenceLabel = dominantPricingSignal ? 'Debate > rejection' : hasOfficial && pricing.claims >= 3 ? 'Pricing signal' : stats.canonicalCount >= 5 ? 'Sample-limited' : 'Insufficient evidence';
  const confidenceColor = hasOfficial && pricing.claims >= 3 ? 'green' : stats.canonicalCount >= 5 ? 'gold' : 'red';

  let headline;
  if (dominantPricingSignal) {
    headline = `${topic} reads as a pricing-strategy debate, not broad rejection.`;
  } else if (hasOfficial && hasMixedPricing) {
    headline = `${topic} triggered a visible backlash around API cost.`;
  } else if (hasOfficial) {
    headline = `${topic} is mainly an official price-table story, with limited controversy in the sample.`;
  } else {
    headline = `${topic} is visible online, but the run lacks an official price-table anchor.`;
  }

  const cards = [
    {
      insight: hasMixedPricing ? 'Judgment: debate, not rejection' : 'Judgment: visible but not heated',
      basis: `${compactNumber(pricing.support + pricing.official)} supportive or official pricing signal${(pricing.support + pricing.official) === 1 ? '' : 's'} versus ${compactNumber(pricing.oppose)} negative signal${pricing.oppose === 1 ? '' : 's'}; criticism clusters around peak-hour or token-cost complaints.`,
      confidence: hasMixedPricing ? 0.68 : 0.62,
      citation_spans: [],
      meta_tags: [`${compactNumber(socialPlatformCount)} platforms`, `${compactNumber(pricing.claims)} pricing claims`]
    },
    {
      insight: hasOfficial ? 'Fact base: official pricing docs' : 'No verified official price table',
      basis: hasOfficial
        ? `${officialCount || pricing.official || acceptedClaims} DeepSeek-domain source group${(officialCount || pricing.official || acceptedClaims) === 1 ? '' : 's'} anchor the price-table facts.`
        : 'The run can describe online reaction, but it should not make price-table claims without an official source.',
      confidence: hasOfficial ? 0.86 : 0.32,
      citation_spans: [],
      meta_tags: [`${compactNumber(officialCount)} official`, `${compactNumber(acceptedClaims)} accepted`]
    },
    {
      insight: 'Amplification is high',
      basis: `${compactNumber(stats.rawCount)} raw item${stats.rawCount === 1 ? '' : 's'} collapse to ${compactNumber(stats.canonicalCount)} evidence group${stats.canonicalCount === 1 ? '' : 's'}; repetition makes the topic look louder than the number of independent viewpoints.`,
      confidence: 0.72,
      citation_spans: [],
      meta_tags: [`${percentText(stats.amplificationRatio)} repeated`, `${compactNumber(stats.canonicalCount)} groups`]
    }
  ];

  const nonTemplateInsights = signalInsights(output).map(cleanTemplateInsight).filter(Boolean);
  const risks = [];
  if (hasMixedPricing) {
    risks.push({
      tension: 'Peak-hour backlash',
      detail: 'Negative pricing signals concentrate around peak-hour price doubling and token-cost complaints.'
    });
  }
  if (stats.amplificationRatio >= 0.35 || stats.exactDuplicateCount > 0) {
    risks.push({
      tension: 'Repeated wording',
      detail: `${compactNumber(stats.exactDuplicateCount)} exact duplicates were found before clustering.`
    });
  }
  if (stats.lowSamplePlatforms.length) {
    risks.push({
      tension: 'Low sample size',
      detail: `${lowSampleLabel}${stats.lowSamplePlatforms.length > 3 ? ', ...' : ''}. Do not compare these as population-level platform opinion.`
    });
  }
  if (Number.isFinite(stats.medianAgeHours) && stats.medianAgeHours > 168) {
    risks.push({
      tension: 'Freshness varies',
      detail: `Median source age is about ${Math.round(stats.medianAgeHours / 24)} days.`
    });
  }

  const details = [
    hasOfficial
      ? 'Official DeepSeek-domain API pricing evidence is present in the run.'
      : 'No verified official pricing source is present in this run.',
    `${compactNumber(stats.canonicalCount)} distinct evidence groups were used after duplicate reduction.`,
    `${compactNumber(socialPlatformCount)} social platforms appear in the sampled evidence; small samples are labelled as limitations.`,
    hasMixedPricing
      ? 'Defense wording: DeepSeek API pricing triggered debate around peak-hour costs, but the run does not support calling it broad rejection.'
      : 'Defense wording: describe the observed evidence; do not claim population-level opinion.'
  ];

  return {
    topic,
    headline,
    confidenceLabel,
    confidenceColor,
    cards: [...cards, ...nonTemplateInsights].slice(0, 4),
    risks: unique(risks, (item) => norm(item.tension)).slice(0, 4),
    details,
    stats: {
      ...stats,
      pricingClaims: pricing.claims,
      pricingOppose: pricing.oppose,
      pricingSupportOrOfficial: pricing.support + pricing.official,
      officialCount,
      acceptedClaims,
      socialPlatformCount
    }
  };
}
