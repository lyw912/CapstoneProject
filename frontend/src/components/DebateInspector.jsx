import { Alert, Empty, Skeleton, Tag, Tooltip } from 'antd';
import {
  AuditOutlined,
  BranchesOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  ExclamationCircleOutlined,
  FileSearchOutlined,
  SafetyCertificateOutlined,
  TeamOutlined
} from '@ant-design/icons';
import { useMemo, useState } from 'react';

import SectionTitle from './SectionTitle';
import { displayText, isHttpUrl, signalArtifact, signalEvidenceGraph } from '../utils/helpers';

const GROUP_META = {
  audited_findings: { label: 'Audited finding', color: 'green' },
  contested_findings: { label: 'Contested finding', color: 'orange' },
  perspective_tensions: { label: 'Perspective tension', color: 'purple' },
  rejected_claims: { label: 'Rejected claim', color: 'red' },
  evidence_gaps: { label: 'Evidence gap', color: 'blue' }
};

const ACTIVE_TASK_STATES = new Set(['queued', 'pending', 'running', 'processing']);
const REVIEWER_IDS = new Set(['skeptic', 'methodologist']);

function readableCode(value, fallback = 'Not recorded') {
  const text = displayText(value, fallback);
  return text.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function compactText(value, limit = 260) {
  const text = displayText(value, '').trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(0, limit - 1)).trim()}...`;
}

function confidenceText(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'Confidence not recorded';
  return `${Math.round(Math.max(0, Math.min(1, numeric)) * 100)}% confidence`;
}

function sessionFromOutput(output) {
  const direct = output?.debate;
  if (direct && Object.keys(direct).length) return direct;
  return signalArtifact(output)?.debate_session || null;
}

function briefFromOutput(output, session) {
  return session?.investigation_brief
    || output?.investigation_brief
    || signalArtifact(output)?.investigation_brief
    || {};
}

function buildSpanIndex(graph) {
  const index = new Map();
  (graph?.evidence_items || []).forEach((item, sourceIndex) => {
    (item?.spans || []).forEach((span) => {
      if (!span?.span_id) return;
      index.set(span.span_id, {
        ...span,
        sourceLabel: `Source ${sourceIndex + 1}`,
        sourceTitle: displayText(item.title || item.source_name || item.url, `Source ${sourceIndex + 1}`),
        url: item.url || ''
      });
    });
  });
  return index;
}

function buildClaimRows(session, graph) {
  const claims = new Map((graph?.claims || []).map((claim) => [claim.claim_id, claim]));
  const assignments = new Map((session?.material_claims || []).map((item) => [item.claim_id, item]));
  const groupByClaim = new Map();
  Object.entries(session?.output_groups || {}).forEach(([group, claimIds]) => {
    (claimIds || []).forEach((claimId) => groupByClaim.set(claimId, group));
  });

  const materialIds = (session?.material_claims || []).map((item) => item.claim_id);
  const fallbackIds = [
    ...Object.values(session?.output_groups || {}).flat(),
    ...(session?.positions || []).map((item) => item.claim_id),
    ...(session?.argument_acts || []).map((item) => item.target_claim_id),
    ...(session?.revisions || []).map((item) => item.claim_id),
    ...(session?.verdicts || []).map((item) => item.claim_id)
  ];
  const claimIds = Array.from(new Set(materialIds.length ? materialIds : fallbackIds)).filter(Boolean);

  return claimIds
    .map((claimId) => {
      const claim = claims.get(claimId) || {};
      const assignment = assignments.get(claimId) || {};
      return {
        ...claim,
        claim_id: claimId,
        claim_text: displayText(claim.claim_text, `Claim ${claimId}`),
        assignment,
        output_group: groupByClaim.get(claimId) || '',
        position_count: (session?.positions || []).filter((item) => item.claim_id === claimId).length,
        challenge_count: (session?.argument_acts || []).filter((item) => item.target_claim_id === claimId && REVIEWER_IDS.has(item.actor_id)).length,
        verdict_count: (session?.verdicts || []).filter((item) => item.claim_id === claimId).length
      };
    })
    .sort((left, right) => Number(right.assignment.score || 0) - Number(left.assignment.score || 0));
}

function EvidenceRefs({ ids = [], spanIndex }) {
  if (!ids.length) return <span className="debate-no-evidence">No evidence span cited</span>;
  return (
    <div className="debate-evidence-refs" aria-label="Cited evidence spans">
      {ids.map((spanId) => {
        const span = spanIndex.get(spanId);
        if (!span) {
          return <Tag key={spanId} color="red" icon={<ExclamationCircleOutlined />}>{compactText(spanId, 24)}</Tag>;
        }
        const tag = <Tag color="blue" icon={<DatabaseOutlined />}>{span.sourceLabel}</Tag>;
        return (
          <Tooltip
            key={spanId}
            title={(
              <div className="debate-span-tooltip">
                <strong>{span.sourceTitle}</strong>
                <span>{compactText(span.text, 420)}</span>
                <code>{spanId}</code>
              </div>
            )}
          >
            {isHttpUrl(span.url)
              ? <a href={span.url} target="_blank" rel="noreferrer" aria-label={`Open ${span.sourceTitle}`}>{tag}</a>
              : <span>{tag}</span>}
          </Tooltip>
        );
      })}
    </div>
  );
}

function OpeningRecord({ profile, position, view, spanIndex }) {
  return (
    <article className={`debate-record ${position ? '' : 'is-muted'}`}>
      <div className="debate-record-marker"><TeamOutlined /></div>
      <div className="debate-record-body">
        <div className="debate-record-head">
          <div>
            <strong>{displayText(profile?.name, profile?.role_id || 'Perspective agent')}</strong>
            <span>{displayText(profile?.analytical_lens, 'Independent analytical lens')}</span>
          </div>
          {position ? (
            <div className="debate-record-tags">
              <Tag color="geekblue">{readableCode(position.stance)}</Tag>
              <Tag>{confidenceText(position.confidence)}</Tag>
              <Tag>{`Evidence v${position.evidence_version}`}</Tag>
            </div>
          ) : <Tag>No opening on this claim</Tag>}
        </div>
        {position ? (
          <>
            <p>{displayText(position.argument, 'No opening argument recorded.')}</p>
            <EvidenceRefs ids={position.evidence_span_ids} spanIndex={spanIndex} />
            {Boolean(position.uncertainties?.length) && (
              <div className="debate-inline-note"><strong>Uncertainty</strong><span>{position.uncertainties.join('; ')}</span></div>
            )}
          </>
        ) : (
          <p>{view ? 'This role executed with a sealed EvidenceView but did not open on the selected claim.' : 'No sealed EvidenceView or position was recorded for this role.'}</p>
        )}
      </div>
    </article>
  );
}

function ArgumentRecord({ act, actorName, spanIndex }) {
  const isRequest = act.act_type === 'request_evidence';
  return (
    <article className="debate-record">
      <div className="debate-record-marker"><SafetyCertificateOutlined /></div>
      <div className="debate-record-body">
        <div className="debate-record-head">
          <div>
            <strong>{actorName}</strong>
            <span>{readableCode(act.act_type, 'Review act')}</span>
          </div>
          <div className="debate-record-tags">
            <Tag color={isRequest ? 'blue' : 'orange'}>{readableCode(act.act_type)}</Tag>
            <Tag>{`Evidence v${act.evidence_version}`}</Tag>
          </div>
        </div>
        <p>{displayText(act.content, 'No argument text recorded.')}</p>
        <EvidenceRefs ids={act.evidence_span_ids} spanIndex={spanIndex} />
        {Boolean(act.reason_codes?.length) && <div className="debate-reason-line">{act.reason_codes.map((code) => <Tag key={code}>{readableCode(code)}</Tag>)}</div>}
        {isRequest && act.requested_evidence && (
          <div className="debate-request"><BranchesOutlined /><span>{Object.entries(act.requested_evidence).map(([key, value]) => `${readableCode(key)}: ${displayText(value)}`).join(' · ')}</span></div>
        )}
      </div>
    </article>
  );
}

function RevisionRecord({ revision, profile, spanIndex }) {
  return (
    <article className="debate-record">
      <div className="debate-record-marker"><BranchesOutlined /></div>
      <div className="debate-record-body">
        <div className="debate-record-head">
          <div>
            <strong>{displayText(profile?.name, revision.agent_id)}</strong>
            <span>Original proposer response</span>
          </div>
          <div className="debate-record-tags">
            <Tag color={revision.revision_type === 'concede' ? 'red' : 'cyan'}>{readableCode(revision.revision_type)}</Tag>
            <Tag>{`Evidence v${revision.evidence_version}`}</Tag>
          </div>
        </div>
        <p>{displayText(revision.revised_argument, 'No revised argument recorded.')}</p>
        {revision.revised_claim_text && <blockquote>{revision.revised_claim_text}</blockquote>}
        <EvidenceRefs ids={revision.evidence_span_ids} spanIndex={spanIndex} />
        <div className="debate-inline-note"><strong>Revision basis</strong><span>{displayText(revision.reason)}</span></div>
      </div>
    </article>
  );
}

function VerdictRecord({ verdict, spanIndex }) {
  const color = { accept: 'green', weaken: 'gold', reject: 'red', needs_search: 'blue', unresolved: 'orange' }[verdict.decision] || 'default';
  return (
    <article className="debate-record">
      <div className="debate-record-marker"><AuditOutlined /></div>
      <div className="debate-record-body">
        <div className="debate-record-head">
          <div>
            <strong>{verdict.judge_id === 'review_judge' ? 'Review Judge' : 'Primary Judge'}</strong>
            <span>{verdict.order_variant === 'reversed' ? 'Reversed argument order' : 'Primary argument order'}</span>
          </div>
          <div className="debate-record-tags">
            <Tag color={color}>{readableCode(verdict.decision)}</Tag>
            <Tag>{confidenceText(verdict.confidence)}</Tag>
          </div>
        </div>
        <p>{displayText(verdict.explanation, 'No judge explanation recorded.')}</p>
        {verdict.final_wording && <blockquote>{verdict.final_wording}</blockquote>}
        {verdict.required_edit && <div className="debate-inline-note"><strong>Required edit</strong><span>{verdict.required_edit}</span></div>}
        <EvidenceRefs ids={verdict.evidence_span_ids} spanIndex={spanIndex} />
      </div>
    </article>
  );
}

function Stage({ icon, eyebrow, title, children, emptyText }) {
  return (
    <section className="debate-stage">
      <div className="debate-stage-title">
        <span aria-hidden="true">{icon}</span>
        <div><small>{eyebrow}</small><h4>{title}</h4></div>
      </div>
      <div className="debate-stage-content">
        {children || <p className="debate-stage-empty">{emptyText}</p>}
      </div>
    </section>
  );
}

export default function DebateInspector({ output, coordinatorTask }) {
  const session = sessionFromOutput(output);
  const graph = signalEvidenceGraph(output);
  const brief = briefFromOutput(output, session);
  const isTaskActive = ACTIVE_TASK_STATES.has(String(coordinatorTask?.status || '').toLowerCase());
  const isSessionActive = session && !['complete', 'completed', 'failed', 'budget_exhausted'].includes(String(session.status || '').toLowerCase());
  const isLoading = isTaskActive || isSessionActive;
  const [selectedClaimId, setSelectedClaimId] = useState('');

  const spanIndex = useMemo(() => buildSpanIndex(graph), [graph]);
  const claimRows = useMemo(() => buildClaimRows(session, graph), [session, graph]);
  const profileIndex = useMemo(() => new Map((session?.profiles || []).map((profile) => [profile.role_id, profile])), [session]);
  const perspectiveProfiles = useMemo(() => (session?.profiles || []).filter((profile) => profile.chamber === 'perspective'), [session]);
  const evidenceViewIndex = useMemo(() => {
    const index = new Map();
    (session?.evidence_views || []).forEach((view) => index.set(`${view.agent_id}:${view.evidence_version}`, view));
    return index;
  }, [session]);

  if (!session || !session.session_id) {
    return (
      <section className="span-12 studio-card debate-inspector debate-inspector-empty">
        <SectionTitle eyebrow="Multi-agent proof" title="Debate Inspector" />
        {isLoading ? (
          <div className="debate-loading" aria-live="polite">
            <Skeleton active paragraph={{ rows: 5 }} />
            <span>Independent openings and evidence review are running.</span>
          </div>
        ) : (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={coordinatorTask?.status === 'error'
              ? 'The latest run failed before a DebateSession was recorded.'
              : 'No DebateSession is attached to this artifact. Run a fresh analysis with deliberation enabled.'}
          />
        )}
      </section>
    );
  }

  const activeClaimId = claimRows.some((item) => item.claim_id === selectedClaimId)
    ? selectedClaimId
    : claimRows[0]?.claim_id;
  const selectedClaim = claimRows.find((item) => item.claim_id === activeClaimId);
  const positions = (session.positions || []).filter((item) => item.claim_id === activeClaimId);
  const acts = (session.argument_acts || []).filter((item) => item.target_claim_id === activeClaimId);
  const reviews = acts.filter((item) => REVIEWER_IDS.has(item.actor_id));
  const responses = acts.filter((item) => !REVIEWER_IDS.has(item.actor_id));
  const revisions = (session.revisions || []).filter((item) => item.claim_id === activeClaimId);
  const verdicts = (session.verdicts || []).filter((item) => item.claim_id === activeClaimId);
  const selectedFailures = (session.protocol_failures || []).filter((item) => !item.claim_id || item.claim_id === activeClaimId);
  const versions = Array.from(new Set([
    ...positions, ...acts, ...revisions, ...verdicts
  ].map((item) => Number(item.evidence_version)).filter(Number.isFinite))).sort((left, right) => left - right);
  const requests = acts.filter((item) => item.act_type === 'request_evidence');
  const decisionSet = new Set(verdicts.map((item) => item.decision).filter(Boolean));
  const judgeDisagreement = verdicts.length > 1 && decisionSet.size > 1;
  const groupMeta = GROUP_META[selectedClaim?.output_group] || { label: 'Unrouted outcome', color: 'default' };
  const openingAgentCount = new Set((session.positions || []).map((item) => item.agent_id)).size;
  const reviewAgentCount = new Set((session.argument_acts || []).filter((item) => REVIEWER_IDS.has(item.actor_id)).map((item) => item.actor_id)).size;
  const judgeCount = new Set((session.verdicts || []).map((item) => item.judge_id)).size;
  const budget = session.budget_summary || {};
  const independence = session.independence_summary || {};

  return (
    <section className="span-12 studio-card debate-inspector">
      <SectionTitle
        eyebrow="Multi-agent proof"
        title="Debate Inspector"
        action={(
          <div className="debate-header-tags">
            {isLoading && <Tag icon={<ClockCircleOutlined />} color="processing">Running</Tag>}
            <Tag color={session.status === 'complete' ? 'green' : 'blue'}>{readableCode(session.status)}</Tag>
            <Tag>{displayText(session.schema_version, 'Evidence debate')}</Tag>
          </div>
        )}
      />

      <div className="debate-summary-band">
        <div><TeamOutlined /><span>Sealed openings</span><strong>{`${openingAgentCount} / ${Math.max(4, perspectiveProfiles.length)}`}</strong></div>
        <div><SafetyCertificateOutlined /><span>Independent reviewers</span><strong>{`${reviewAgentCount} / 2`}</strong></div>
        <div><AuditOutlined /><span>Blind judges</span><strong>{`${judgeCount} / 2`}</strong></div>
        <div><ClockCircleOutlined /><span>Debate calls</span><strong>{`${Number(budget.llm_calls || 0)} / ${Number(budget.max_llm_calls || 18)}`}</strong></div>
      </div>

      <div className="debate-independence-strip" aria-label="Agent independence dimensions">
        <strong>{independence.configured_mode === 'heterogeneous' ? 'Heterogeneous model routing' : 'Same-model deployment'}</strong>
        <Tag color={independence.context_isolated ? 'green' : 'red'}>{`Context isolated: ${independence.context_isolated ? 'yes' : 'no'}`}</Tag>
        <Tag color={independence.objective_distinct ? 'green' : 'red'}>{`Objectives distinct: ${independence.objective_distinct ? 'yes' : 'no'}`}</Tag>
        <Tag color={independence.model_family_distinct ? 'green' : 'gold'}>{`Model families distinct: ${independence.model_family_distinct ? 'yes' : 'no'}`}</Tag>
        {budget.termination_reason && <span>{`Termination: ${readableCode(budget.termination_reason)}`}</span>}
      </div>

      <section className="debate-brief" aria-labelledby="investigation-brief-title">
        <div className="debate-brief-head">
          <div><FileSearchOutlined /><span>Read-only execution contract</span><h3 id="investigation-brief-title">Investigation Brief</h3></div>
          <Tag>{displayText(brief.brief_version, 'Brief version not recorded')}</Tag>
        </div>
        <dl className="debate-brief-grid">
          <div><dt>Original topic</dt><dd>{displayText(brief.original_query, output.query)}</dd></div>
          <div><dt>Target / mode</dt><dd>{`${displayText(brief.target_entity)} · ${readableCode(brief.analysis_type)}`}</dd></div>
          <div className="is-wide"><dt>Factual question</dt><dd>{displayText(brief.factual_question)}</dd></div>
          <div className="is-wide"><dt>Public-discourse question</dt><dd>{displayText(brief.discourse_question)}</dd></div>
          <div><dt>Time scope</dt><dd>{displayText(brief.time_scope)}</dd></div>
          <div><dt>Sample boundary</dt><dd>{displayText(brief.sample_boundary)}</dd></div>
        </dl>
        <details className="debate-obligations">
          <summary>Selected roles and evidence obligations</summary>
          <div>
            {perspectiveProfiles.map((profile) => (
              <section key={profile.role_id}>
                <strong>{profile.name}</strong>
                <span>{(profile.evidence_obligations || []).join(' · ') || 'No role-specific obligation recorded'}</span>
              </section>
            ))}
          </div>
        </details>
      </section>

      {Boolean(session.protocol_failures?.length) && (
        <Alert
          className="debate-alert"
          type="warning"
          showIcon
          message={`${session.protocol_failures.length} protocol diagnostic${session.protocol_failures.length === 1 ? '' : 's'} recorded`}
          description={(
            <details>
              <summary>Inspect failures</summary>
              <ul>{session.protocol_failures.map((failure) => <li key={failure.failure_id}><strong>{`${readableCode(failure.phase)} · ${displayText(failure.agent_id)}`}</strong><span>{`${readableCode(failure.failure_type)}: ${failure.message}`}</span></li>)}</ul>
            </details>
          )}
        />
      )}

      {!claimRows.length ? (
        <Empty className="debate-no-claims" image={Empty.PRESENTED_IMAGE_SIMPLE} description="The evidence gate found no material claim requiring LLM review." />
      ) : (
        <div className="debate-workspace">
          <aside className="debate-claim-rail" aria-label="Material claims">
            <div className="debate-claim-rail-title">
              <span>Material claims</span>
              <Tag>{claimRows.length}</Tag>
            </div>
            {claimRows.map((claim, index) => {
              const meta = GROUP_META[claim.output_group] || { label: 'Pending route', color: 'default' };
              const active = claim.claim_id === activeClaimId;
              return (
                <button
                  key={claim.claim_id}
                  type="button"
                  className={active ? 'active' : ''}
                  onClick={() => setSelectedClaimId(claim.claim_id)}
                  aria-pressed={active}
                >
                  <span className="debate-claim-index">{String(index + 1).padStart(2, '0')}</span>
                  <strong>{compactText(claim.claim_text, 150)}</strong>
                  <span className="debate-claim-counts">{`${claim.position_count} openings · ${claim.challenge_count} reviews · ${claim.verdict_count} verdicts`}</span>
                  <Tag color={meta.color}>{meta.label}</Tag>
                </button>
              );
            })}
          </aside>

          <article className="debate-claim-pane">
            <header className="debate-claim-header">
              <div>
                <div className="debate-claim-kicker"><Tag color={groupMeta.color}>{groupMeta.label}</Tag><code>{activeClaimId}</code></div>
                <h3>{selectedClaim?.claim_text}</h3>
                <div className="debate-claim-meta">
                  {selectedClaim?.claim_type && <Tag>{readableCode(selectedClaim.claim_type)}</Tag>}
                  {selectedClaim?.aspect && <Tag>{readableCode(selectedClaim.aspect)}</Tag>}
                  {selectedClaim?.stance && <Tag>{readableCode(selectedClaim.stance)}</Tag>}
                  {(selectedClaim?.assignment?.reason_codes || []).map((code) => <Tag key={code} color="gold">{readableCode(code)}</Tag>)}
                </div>
              </div>
              <div className="debate-gate-state">
                {groupMeta.color === 'green' ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
                <span>Evidence gate + paired verdict</span>
              </div>
            </header>

            <div className="debate-version-flow" aria-label="Evidence version trajectory">
              <span><DatabaseOutlined />{`Evidence v${versions[0] ?? 0}`}</span>
              <i aria-hidden="true">→</i>
              {requests.length ? <><span><BranchesOutlined />Typed retrieval</span><i aria-hidden="true">→</i><span><DatabaseOutlined />{`Evidence v${versions.at(-1) ?? versions[0] ?? 0}`}</span><i aria-hidden="true">→</i></> : null}
              <span><AuditOutlined />Paired adjudication</span>
            </div>

            {judgeDisagreement && (
              <Alert className="debate-alert" type="warning" showIcon message="Judge disagreement preserved" description="Primary and Review Judges returned incompatible decisions; the protocol does not force a majority verdict." />
            )}
            {Boolean(selectedFailures.length) && selectedFailures.some((item) => item.claim_id === activeClaimId) && (
              <Alert className="debate-alert" type="error" showIcon message="This claim has a protocol failure" description={selectedFailures.filter((item) => item.claim_id === activeClaimId).map((item) => item.message).join(' · ')} />
            )}

            <Stage icon={<TeamOutlined />} eyebrow="Perspective chamber" title="Four sealed openings" emptyText="No perspective openings were recorded.">
              {perspectiveProfiles.length ? perspectiveProfiles.map((profile) => {
                const position = positions.find((item) => item.agent_id === profile.role_id);
                const view = evidenceViewIndex.get(`${profile.role_id}:${position?.evidence_version ?? versions[0]}`);
                return <OpeningRecord key={profile.role_id} profile={profile} position={position} view={view} spanIndex={spanIndex} />;
              }) : null}
            </Stage>

            <Stage icon={<SafetyCertificateOutlined />} eyebrow="Evidence review chamber" title="Skeptic and Methodologist" emptyText="No independent review act was recorded for this claim.">
              {reviews.length ? reviews.map((act) => <ArgumentRecord key={act.act_id} act={act} actorName={displayText(profileIndex.get(act.actor_id)?.name, readableCode(act.actor_id))} spanIndex={spanIndex} />) : null}
            </Stage>

            <Stage icon={<BranchesOutlined />} eyebrow="Routed response" title="Original proposer answers" emptyText="No proposer response or revision was recorded for this claim.">
              {responses.length ? responses.map((act) => <ArgumentRecord key={act.act_id} act={act} actorName={displayText(profileIndex.get(act.actor_id)?.name, act.actor_id)} spanIndex={spanIndex} />) : null}
              {revisions.length ? revisions.map((revision) => <RevisionRecord key={revision.revision_id} revision={revision} profile={profileIndex.get(revision.agent_id)} spanIndex={spanIndex} />) : null}
            </Stage>

            <Stage icon={<AuditOutlined />} eyebrow="Blind adjudication" title="Primary and Review Judges" emptyText="No paired judge verdict was recorded for this claim.">
              {verdicts.length ? verdicts.map((verdict) => <VerdictRecord key={verdict.verdict_id} verdict={verdict} spanIndex={spanIndex} />) : null}
            </Stage>
          </article>
        </div>
      )}
    </section>
  );
}
