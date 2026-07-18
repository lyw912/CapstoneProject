# ADR-001: Evidence-Bound Independent Multi-Agent Deliberation

Status: Accepted - implementation in progress

Date: 2026-07-15

## Context

The active Coordinator replaced the legacy prompt-level deliberation path with a Query/Media fusion graph and EvidenceCore. This improved source identity, provenance, claim binding, sufficiency routing, and output auditability, but removed independently executed deliberation agents from the active runtime.

The required final product must retain genuine LLM multi-agent collaboration and use EvidenceCore to make that collaboration reliable. The final defense deck will present the implemented final architecture directly, not the refactor history.

## Decision Drivers

- independent agents must actually execute, receive challenges, and revise or dissent;
- factual argument acts must bind to canonical evidence spans;
- public-opinion analysis must retain plural perspectives and value disagreements;
- communication must resist conformity, correlated errors, and moderator impersonation;
- judging must be bounded, bias-aware, and able to abstain/request evidence;
- the protocol must fit the current LangGraph/Fusion runtime and external artifact contract;
- the implementation must be observable, testable, and explainable in four slides;
- API calls, latency, and failure modes must be measured rather than hand-waved.

## Already Accepted Constraints

1. EvidenceCore is retained as the canonical evidence owner and single-writer reducer.
2. The legacy moderator-simulated cross-examination is not sufficient as the final protocol.
3. Agents form sealed opening positions before peer exposure.
4. Agent messages and evidence objects remain separate bounded contexts.
5. The final system preserves revision provenance and material dissent.
6. Consensus is not the correctness or stopping criterion.

## Decisions Resolved During Grilling

### G1 - Primary Collaboration Topology

Decision: use **Dual-Chamber Deliberation** as the primary topology.

- The `Perspective Chamber` runs multiple independently executed analytical agents against a frozen EvidenceCore snapshot.
- The `Evidence Review Chamber` runs independent skeptical and methodological review over material claims and arguments.
- Original perspective agents receive routed challenges and must personally rebut, revise, concede, abstain, or request evidence; a moderator cannot answer on their behalf.
- Per-claim judging occurs only after review/revision, and the final artifact preserves both evidence provenance and deliberation provenance.

This selects the two-level organizational structure only. It does not yet decide how analytical lenses are selected, how models are assigned, how challenges are routed, or whether adjudication uses one judge or a jury.

### G2 - Perspective Chamber Role Selection

Decision: run four perspective agents selected deterministically from a versioned analytical-role catalog.

- `analysis_type` selects the role set for `event`, `technology`, `policy`, `brand`, `person`, or `general` topics.
- Each `DebateAgentProfile` has a stable role ID, analytical mandate, required evidence dimensions, prohibited inference patterns, protocol capabilities, and output contract.
- The role-set version and selection reason are recorded in the debate trace.
- An LLM does not invent runtime roles. Unknown or ambiguous topics use the explicit `general` profile rather than ad hoc personas.
- The existing perspective templates are a starting taxonomy only; they must be converted from prompt strings into validated runtime profiles and active independent graph nodes.

### G3 - Model Independence And Current Deployment

Decision: design for a configurable heterogeneous model pool, but implement and present the current deployment with one available model API.

- Every role has an explicit model profile so different providers/model families can be assigned later without changing the deliberation protocol.
- The current implementation maps all perspective, review, and adjudication roles to the available Query-model API.
- Current agent independence comes from separate invocations, sealed context, distinct analytical objectives, filtered evidence views, and private position state; it does not include model-family diversity.
- The artifact records independence dimensions separately: `context_isolated`, `objective_distinct`, and `model_family_distinct`.
- Provider absence may trigger a declared same-model fallback, never a hidden substitution.
- The final defense deck describes the executed same-model configuration. Configurable heterogeneous routing may appear only as an extension point, not as an implemented multi-model claim.

### G4 - Evidence Review Routing

Decision: use sparse, material-claim routing instead of all-to-all cross-examination or exhaustive review.

- EvidenceCore deterministically ranks claims for review using impact and risk signals, including contradiction, one-sided or single-source support, UGC-only support, amplification/copy risk, causal or population-level extrapolation, and high-impact evidence gaps.
- The Skeptic independently searches the available snapshot for counter-evidence, alternative explanations, and unsupported assumptions.
- The Methodologist independently checks source independence, sampling boundaries, temporal scope, metric use, and inference type.
- Challenges are routed only to the original perspective agent responsible for the claim. That agent must rebut, revise, concede, abstain, or submit a typed evidence request itself.
- Low-risk claims still pass deterministic source-span eligibility checks but do not consume a full LLM review cycle.
- Routing scores, selected claims, skipped claims, and reasons are recorded so sparse review remains inspectable rather than arbitrary.

### G5 - Claim Adjudication

Decision: use deterministic evidence eligibility followed by paired blind LLM adjudication without forced consensus.

1. EvidenceCore rejects invalid references, missing required source spans, and mechanically detectable scope violations before LLM judgment.
2. A Primary Judge evaluates anonymized per-claim argument subgraphs against a claim-mode-specific rubric.
3. A Review Judge runs in an isolated context with argument order shuffled/reversed and independently checks the verdict and decisive evidence.
4. Compatible verdicts are merged with both decision traces. Incompatible verdicts become `unresolved`, `needs_search`, or `weaken` according to the evidence state; no third majority vote forces a winner.
5. Judge inputs exclude role/model identity, self-reported prestige, and irrelevant transcript order. Multiple material claims may be batched while preserving per-claim verdicts.

The current deployment uses the same model API for both judges, so this mitigates context/order bias but does not claim model-family-independent adjudication.

### G6 - Debate And Retrieval Budget

Decision: run one mandatory dual-chamber debate cycle plus at most one targeted evidence-retrieval cycle.

- The mandatory cycle contains four sealed perspective openings, independent Skeptic and Methodologist reviews, routed proposer responses/revisions, and paired blind adjudication.
- At most six material claims enter LLM evidence review in one run.
- `needs_search` decisions may create one bounded typed Query/Media retrieval batch. EvidenceCore then creates a new snapshot, and only agents/claims affected by changed evidence are re-evaluated.
- The debate layer has a configurable hard cap of approximately 18 new LLM calls and its own deadline. Existing Query/Media subgraph calls are measured separately but contribute to the run-level budget summary.
- No second retrieval/debate cycle is allowed in the current implementation. Budget/deadline termination preserves unresolved claims, pending requests, and diagnostics in the artifact.

### G7 - Perspective Evidence Views

Decision: give every perspective agent a shared evidence core plus a deterministic role-specific evidence slice.

- The shared core contains the query/scope, claim-mode definitions, canonical source index, highest-priority factual spans, sample boundaries, global coverage gaps, and quality/freshness warnings.
- Each role slice adds lens-relevant top-k spans, claims, section dossiers, and mandatory evidence dimensions from its `DebateAgentProfile`.
- Views are derived from one frozen EvidenceCore snapshot. The artifact records snapshot version, visible object IDs, view-policy version, and truncation/selection reasons for every agent.
- Agents may submit typed requests for other source/span IDs or new evidence, but cannot silently access untracked context or introduce factual evidence from model memory.
- Judges receive complete per-material-claim evidence/argument subgraphs, not one perspective's filtered view.

### G8 - Final Synthesis And Dissent Policy

Decision: use a layered final artifact that distinguishes usable findings, unresolved empirical disputes, legitimate perspective tensions, rejected claims, and evidence gaps.

- `Audited Findings` contains accepted and weakened claims with wording policy, citations, counter-evidence, and verdict provenance.
- `Contested Findings` contains unresolved empirical disputes with competing positions, decisive spans, and the reason adjudication remains inconclusive.
- `Perspective Tensions` contains normative, stakeholder, or value-framework disagreements that should not be forced into true/false semantics.
- `Rejected Claims` remain accessible in Proof/Audit trace but cannot enter report assertions.
- `Evidence Gaps` records typed pending requests, budget/deadline exhaustion, and what evidence could change the outcome.

ReportEngine consumes these categories under different rendering rules rather than flattening all outputs into consensus/dissent strings.

### G9 - Four-Slide Defense Narrative

Presentation status: superseded by ADR-002 G15. The real claim remains a required implementation trace, but the accepted four-page information architecture is now Coordinator Architecture -> Debate Protocol -> Evidence Architecture -> One Claim, End to End.

Decision: use one real claim trajectory as the visual spine while revealing the complete final architecture.

1. A real EvidenceCore snapshot feeds four sealed, independently executed Perspective Agents through shared-core plus role-specific views.
2. One material claim exposes independent Skeptic/Methodologist attacks and the original proposer's evidence-bound rebuttal or revision.
3. A typed evidence request shows the single bounded retrieval loop and affected-claim transition from snapshot `vN` to `vN+1`.
4. Paired blind adjudication produces a layered artifact containing audited findings, contested findings, perspective tensions, rejected claims, and evidence gaps.

The slides describe the implemented final system directly. Ownership remains visible, but migration history and old-versus-new framing are excluded. The anchor claim must come from an inspected post-implementation trace rather than a fabricated storyboard.

### G10 - Validation Strength

Decision: perform engineering acceptance rather than a comparative or formal effectiveness experiment.

- Contract tests verify independent agent execution, sealed evidence views, argument reference validation, challenge routing, proposer revision, one retrieval cycle, paired adjudication, layered output, failure handling, and budget termination.
- One representative live end-to-end run must execute the complete configured path and produce an inspectable artifact/trace suitable for the final slide claim trajectory.
- Acceptance proves that the designed protocol executes and preserves its contracts. It does not prove higher accuracy, groundedness, usefulness, or efficiency than single-agent or non-interactive baselines.
- The final deck may say `implemented`, `executed`, `bounded`, `versioned`, and `inspectable` when supported by the run. It may not say `improves`, `outperforms`, or equivalent comparative language.

### G11 - Debate Proof Surface

Decision: extend the existing Proof interface with a compact Debate Inspector.

- The inspector groups data by claim rather than rendering a chat transcript.
- It exposes four sealed opening positions, selected Skeptic/Methodologist challenges, proposer rebuttals/revisions/evidence requests, cited source/span links, paired judge verdicts, and final output category.
- It includes loading, partial-agent failure, invalid-reference, budget-stop, no-material-claim, and judge-disagreement states.
- It reads the versioned Coordinator artifact and never invents a separate frontend interpretation of the debate.
- It does not use live chat bubbles, simulated typing, or theatrical debate animation.
- A cropped real inspector state may support the final defense, but presentation-native diagrams remain responsible for explaining the mechanism.

### G12 - Provisional Acceptance Topic

Decision: provisionally use `DeepSeek API pricing` for the single complete acceptance run and final claim trajectory.

- The run must execute from scratch through the implemented dual-chamber protocol; the pre-existing pricing artifact cannot serve as evidence for new behavior.
- The case should contain both official/primary pricing evidence and sampled public-discourse evidence so factual and discourse claim modes can be distinguished.
- The topic is provisional because the user said "choose A first." It may be reconsidered only if a preflight proves that required sources are unavailable, no material claim can exercise review/revision, or a mandatory protocol stage cannot execute.
- Any topic change requires an explicit new decision; the implementation must not silently select a more convenient demo.

### G13 - Display Query Versus Execution Brief

Decision: keep a broad topic-form frontend entry while executing a structured two-part investigation brief.

- The visible/input query remains `DeepSeek API pricing` for a simple demonstration and reusable topic-oriented product interface.
- The Planner deterministically produces an `InvestigationBrief` containing the original query, target entity, analysis type, factual pricing-change question, bounded public-discourse question, claim modes, time/sample boundaries, and role evidence obligations.
- The acceptance brief must answer the equivalent of: `How did DeepSeek API pricing change, and what does the available public discourse support about user reaction?`
- Perspective, review, retrieval, judging, synthesis, and ReportEngine stages consume the structured brief, not only the three-word topic.
- The artifact preserves both `original_query` and the complete derived brief so execution intent is inspectable and reproducible.

### G14 - InvestigationBrief Visibility

Decision: keep the topic entry simple and expose the derived InvestigationBrief read-only in Proof/Debate Inspector.

- The run entry continues to show only the broad topic.
- Proof displays original topic, factual question, public-discourse question, time/sample boundary, selected perspective roles, and evidence obligations.
- The current scope does not add a pre-run advanced editor. This keeps the acceptance run reproducible and avoids making role/question editing an untested product surface.

## Candidate Decisions

See [hybrid-evidence-debate-options.md](../architecture/hybrid-evidence-debate-options.md).

The live candidates are:

- Fixed Analytical Council;
- Claim Court;
- Dual-Chamber Deliberation;
- Sparse Argument-Graph Debate;
- Asynchronous Blackboard Swarm;
- composable hybrids of these patterns.

Independent Panel + Jury and Moderator-Simulated Cross-Examination remain comparison baselines, not intended final choices.

## Unresolved Decisions

No unresolved product-level decisions block implementation. The exact claim, spans, revisions, and verdicts used in the final slides are runtime-derived selections from the inspected acceptance artifact, not preselected design decisions.

## Consequences

The active Fusion graph requires a two-chamber deliberation subgraph after EvidenceCore reduction and before final synthesis, typed argument contracts and repositories, bounded targeted retrieval, paired adjudication, artifact/projection evolution, focused contract tests, Proof UI support, one live acceptance run, and a final four-slide rebuild from that run.
