# Hybrid Evidence Debate Domain Model

Status: working model for grilling; not an accepted architecture.

## 1. Goal

The final Coordinator must combine two capabilities without pretending that one is the other:

1. independently executed LLM agents contribute genuinely different analysis, criticism, and revisions;
2. EvidenceCore makes every factual move source-bound, versioned, auditable, and eligible for deterministic checks.

The system is not a chat room with citations attached, and EvidenceCore is not a replacement for multi-agent reasoning.

## 2. Bounded Contexts

### 2.1 EvidenceCore

EvidenceCore owns objects about the observed world:

| Entity | Meaning | Existing support |
| --- | --- | --- |
| `AcquisitionObservation` | One agent/task/provider discovery of a source | Implemented |
| `SourceDocument` / canonical source | One resolved source identity | Implemented through normalized/canonical evidence |
| `EvidenceSpan` | Addressable excerpt in a source | Implemented |
| `ClaimProposal` | Specialist-authored candidate assertion | Implemented |
| `Claim` | Canonical claim with scope, stance, confidence, and span links | Implemented |
| `EvidenceRelation` | Support, contradict, qualify, or duplicate relation | Partly implemented |
| `SufficiencyAssessment` | Evidence quality/gap assessment for a claim | Implemented |
| `EvidenceSnapshot` | Immutable/version-addressable evidence view | Implemented through Blackboard snapshots/version |

EvidenceCore never treats an LLM statement as evidence. Agents may only reference or propose evidence objects; they do not directly mutate canonical state.

### 2.2 Deliberation Protocol

The deliberation context owns objects about what agents argued:

| Entity | Required fields |
| --- | --- |
| `DebateSession` | session ID, topic, claim scope, evidence version, protocol, budgets, status |
| `InvestigationBrief` | original topic, target, factual/discourse questions, scope, claim modes, role obligations |
| `DebateAgentProfile` | agent ID, analytical lens, protocol capabilities, model route, evidence-view policy |
| `EvidenceView` | snapshot version plus the filtered IDs visible to one agent |
| `Position` | agent, claim, initial stance, confidence, assumptions, cited span IDs, sealed timestamp |
| `ArgumentAct` | actor, act type, target claim/act, assertion, cited span IDs, reason codes, round |
| `Revision` | previous position, new position, triggering acts/spans, change type, reason |
| `EvidenceRequest` | claim, support/refute/clarify purpose, missing dimension, desired source, budget |
| `ClaimVerdict` | decision, rubric dimensions, decisive acts/spans, unresolved attacks, required wording |
| `DissentRecord` | minority position, whether empirical/normative, evidence status, preservation policy |
| `ProtocolFailure` | invalid reference, non-response, duplicate act, timeout, order inconsistency, budget stop |
| `ArgumentLedger` | append-only record of all the above, keyed to an EvidenceSnapshot version |

### 2.3 Final Artifact

The Coordinator artifact joins the two contexts without collapsing them:

```text
Evidence provenance: source -> span -> claim
Argument provenance: agent -> act -> challenge/revision -> verdict
Final insight: audited claim IDs + evidence IDs + argument/verdict IDs + wording policy
```

## 3. Claim Modes

One rubric cannot adjudicate every public-opinion statement.

| Claim mode | Example | Appropriate test |
| --- | --- | --- |
| `empirical_fact` | "The listed price changed from X to Y" | direct source-span entailment, freshness, authority |
| `causal_or_forecast` | "The price change caused negative adoption" | multiple evidence types, alternative explanations, calibrated uncertainty |
| `discourse_observation` | "Price complaints dominate the sampled posts" | sample boundary, dedup/amplification control, stance coverage |
| `normative_interpretation` | "The policy is unfair" | stakeholder/value framing and evidence context; preserve legitimate dissent |

`reject` is valid for unsupported empirical claims. A normative disagreement may remain `unresolved` or `plural` rather than false.

## 4. Role Dimensions

Analytical lens and protocol function are orthogonal.

Analytical lenses may include factual/technical, public/stakeholder, media/narrative, commercial/strategic, and policy/ethics. They determine what an agent attends to.

Protocol functions may include proposer, challenger, methodologist, retrieval requester, reviser, and judge. They determine what acts an agent may perform in a phase.

An agent can be a `policy/ethics` lens during the sealed opening and later receive a `challenger` function for a selected claim. A permanent "supporter" persona is not required.

## 5. Candidate Lifecycle

```text
Acquire evidence
  -> EvidenceCore reduces to snapshot vN
  -> select agent profiles and filtered evidence views
  -> sealed independent opening positions
  -> publish positions to ArgumentLedger
  -> route non-duplicate material challenges
  -> agents rebut, revise, concede, abstain, or request evidence
  -> typed evidence requests re-enter bounded Query/Media acquisition
  -> EvidenceCore creates snapshot vN+1
  -> affected agents re-evaluate only changed claims
  -> anonymous rubric-bound judge or jury decides per claim
  -> preserve accepted, weakened, rejected, unresolved, and plural outcomes
  -> synthesize only from verdict-bearing claims
```

The selected architecture may omit or reorganize stages, but it must state the loss explicitly.

## 6. State Transitions

### Position

```text
sealed -> published -> challenged -> upheld | revised | conceded | abstained
```

### Claim

```text
proposed -> evidence_eligible -> uncontested | contested
contested -> needs_evidence -> re_evaluated
uncontested | re_evaluated -> accepted | weakened | rejected | unresolved | plural
```

Consensus is not a terminal state. Termination is based on claim coverage, material unresolved challenges, evidence-request status, and budget.

## 7. Non-Negotiable Invariants

1. Every agent forms its opening position before seeing peer positions.
2. Every factual `support`, `challenge`, or `rebut` references valid span IDs from a declared EvidenceSnapshot, or explicitly declares `missing_evidence`.
3. Debate agents cannot directly write canonical EvidenceCore objects.
4. A moderator cannot impersonate all debaters during cross-examination.
5. Every revision preserves the prior position and its trigger.
6. Unsupported evidence references fail ingest and become protocol diagnostics.
7. Agent agreement never upgrades source independence or evidence quality.
8. Deterministic eligibility checks run before LLM judgment.
9. Judges do not receive author/model identity and must tolerate `unresolved`/`needs_search`.
10. The final artifact preserves material dissent and budget-limited gaps.

## 8. Working Glossary

| Term | Definition |
| --- | --- |
| Agent independence | Isolation of initial context/answer, distinct objective, and optionally distinct model route; not merely a different name |
| Analytical lens | The domain perspective that controls what evidence/questions an agent prioritizes |
| Protocol function | The allowed action role in a debate phase, such as challenger or judge |
| Evidence | An externally acquired, addressable source span; never an agent's prose |
| Position | One agent's stance on one claim at one evidence version |
| Argument act | A typed communicative move that supports, attacks, qualifies, revises, or requests evidence |
| Argument Ledger | Append-only provenance of positions, acts, revisions, verdicts, and protocol failures |
| Evidence version | Immutable snapshot ID against which an argument was formed |
| Material challenge | A non-duplicate attack that could change claim eligibility, confidence, wording, or retrieval |
| Revision provenance | The exact previous position, observed challenge/evidence, and reason for a change |
| Dissent | A material minority position retained without being forced into consensus |
| Verdict | Per-claim protocol outcome, not a claim that the underlying source is objectively true |
| Final artifact | Versioned join of evidence provenance, deliberation provenance, verdicts, and report policy |

## 9. Unresolved Domain Questions

Resolved: the visible collaboration uses two chambers. Independent analytical agents form a Perspective Chamber; independent skeptic/methodologist agents form an Evidence Review Chamber. Original agents answer routed challenges themselves.

Resolved: four perspective agents are selected deterministically by `analysis_type` from a versioned role catalog. Runtime LLM role invention is not allowed.

Resolved: the protocol supports role-specific heterogeneous model profiles, but the current implementation uses one available model API. It records context, objective, and model-family independence separately and must not claim current model diversity.

Resolved: Evidence Review uses sparse material-claim routing. EvidenceCore selects high-impact/high-risk claims; Skeptic and Methodologist review them independently; only the original proposer may answer or revise.

Resolved: adjudication uses a deterministic eligibility gate plus isolated Primary and Review Judges with shuffled argument order. Conflicting judgments remain unresolved, weakened, or trigger retrieval instead of being forced through majority voting.

Resolved: the current implementation runs one mandatory debate cycle plus at most one typed targeted-retrieval cycle, reviews at most six material claims, caps new debate-layer calls at approximately 18, and re-evaluates only affected agents/claims.

Resolved: sealed openings use a shared evidence core plus deterministic role-specific slices. Every view remains tied to one snapshot and may be expanded only through typed, traced requests.

Resolved: final synthesis separates Audited Findings, Contested Findings, Perspective Tensions, Rejected Claims, and Evidence Gaps, each with different downstream rendering rules.

Resolved: the four-slide defense presents Coordinator Architecture, Debate Protocol, and Evidence Architecture before using one real post-implementation claim as a compact end-to-end integration trace. The claim no longer organises the first three pages, and the deck does not present refactor history.

Resolved: validation is one full engineering-acceptance run plus focused contract tests. No comparative effectiveness claim is permitted.

Resolved: the existing Proof interface gains a claim-centric Debate Inspector backed directly by the versioned artifact; no theatrical live-chat UI is used.

Provisionally resolved: use `DeepSeek API pricing` for the one full acceptance run, subject only to an explicit preflight failure and a new user decision.

Resolved: the frontend accepts/displays the broad topic `DeepSeek API pricing`, while the Planner produces and all internal stages execute a structured factual-plus-discourse `InvestigationBrief`; both are stored in the artifact.

Resolved: the derived InvestigationBrief appears read-only in Proof/Debate Inspector; the broad topic entry remains unchanged and no advanced editor is added in the current scope.

No unresolved domain-model question requires user input. Runtime acceptance will select the real claim trajectory for the final slides.
