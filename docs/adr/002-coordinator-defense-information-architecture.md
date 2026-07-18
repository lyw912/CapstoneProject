# ADR-002: Coordinator Defense Information Architecture

Status: Accepted

Date: 2026-07-16

## Context

ADR-001 G9 selected one real claim trajectory as the visual spine of all four Coordinator slides. The resulting deck is locally accurate but structurally misleading: Slide 1 compresses the reusable subsystem into a flat runtime chain, while Slides 2-4 continue one DeepSeek pricing case. An examiner can follow the case without being able to reconstruct the Coordinator's architecture, organisation, state ownership, or reusable design.

The presentation scope is only the candidate-owned Coordinator. QueryEngine, MediaEngine, Proof, and ReportEngine may appear only as external contracts at its boundary.

## Decision Drivers

- Give the audience a correct mental model of the complete Coordinator before showing detail.
- Separate reusable architecture from one runtime instance.
- Give Debate and Evidence each enough space to explain their distinctive controls.
- Preserve the truthful implementation boundary without overloading the visible deck: show isolated contexts and versioned state, do not imply multi-model execution, and make no effectiveness claim.
- Fit four English slides and a 3-5 minute defense segment with projection-safe typography.
- Use short titles that identify the page object rather than sentence-length conclusions.

## Options

### A. Case Spine Across Four Slides

Sequence: protocol summary -> sealed openings -> claim revision -> verdict and Inspector.

Strength: concrete and memorable.

Failure: the example becomes the apparent project, while Coordinator hierarchy and reusable Evidence design remain implicit.

### B. Architecture Before Instance

Sequence: Coordinator architecture -> Debate protocol -> Evidence architecture -> one end-to-end trace.

Strength: establishes the subsystem, then explains its two main design contributions, then proves that they integrate in a real run.

Cost: the real claim receives one page instead of three, so the trace must be aggressively compressed.

### C. Three Controls Before Instance

Sequence: ownership boundary -> independence control -> evidence control -> release control with a small example on every page.

Strength: assertion-led and compact.

Failure: risks three abstract control pages and still makes the organisational hierarchy hard to reconstruct.

## Recommended Decision

Select Option B and supersede ADR-001 G9 as a presentation decision. ADR-001's product architecture decisions remain unchanged.

G15 accepted this recommendation on 2026-07-16.

### Slide 1 - `Coordinator Architecture`

Show one candidate-owned boundary with four readable layers:

1. `Fusion Control Graph`: plan, fan-out, reduce, deliberate, audit, finalise, plus the one bounded follow-up edge.
2. `Dual-Chamber Deliberation`: Perspective Chamber, Evidence Review Chamber, proposer response, and paired adjudication.
3. `Governed State`: EvidenceCore owns versioned source/span/claim truth state; ArgumentLedger owns positions, attacks, revisions, verdicts, and failures.
4. `Coordinator Artifact`: one versioned release object joins evidence, argument, audit, diagnostics, and downstream projections.

Query/Media typed contributions enter at the left boundary. Proof/ReportEngine consume the artifact at the right boundary. Their internals are not shown.

### Slide 2 - `Debate Protocol`

Explain reusable collaboration controls rather than the pricing case:

- sealed independent openings;
- sparse material-claim review by Skeptic and Methodologist;
- original-proposer accountability for rebuttal, revision, concession, abstention, or evidence request;
- paired blind, order-reversed adjudication with deterministic conservative merge;
- isolated contexts shown as the visible independence mechanism, without model-routing or numeric runtime configuration.

### Slide 3 - `Evidence & Argument`

Explain the two coordinated state records and their contract:

- multiple discoveries may resolve to one canonical source while their acquisition paths remain preserved;
- canonical sources expose exact, addressable excerpts rather than whole-document citations;
- claims connect to supporting and contradicting excerpts in an EvidenceGraph;
- quality, freshness, source diversity, copy risk, and sufficiency remain inspectable graph attributes;
- one reducer owns canonical writes and every accepted change advances the EvidenceCore version;
- ArgumentLedger preserves the attributable position -> challenge -> revision -> verdict chain without treating agent prose as evidence;
- every ledger act references claim/excerpt IDs and an evidence version; typed evidence requests create a new EvidenceCore version and affected-claim reassessment.

### Slide 4 - `One Claim, End to End`

Compress the real pricing trace into one integration proof:

`topic -> initial evidence snapshot -> four sealed openings -> two attacks -> proposer revision -> targeted evidence follow-up -> updated evidence snapshot -> paired verdicts -> audited finding -> Proof / ReportEngine`

Use only the shortest before/after wording needed to make the revision visible. Do not use an Inspector screenshot; the artifact-backed trace itself is the implementation evidence.

## Presentation Glossary

| Term | Meaning in this deck |
| --- | --- |
| Coordinator architecture | Stable internal ownership, layers, contracts, and control relationships; not a chronological case flow |
| Control graph | Parent LangGraph that schedules stages and owns the only bounded return edge |
| Debate protocol | Reusable rules governing independent positions, review, response, and adjudication |
| Evidence & argument state | EvidenceCore owns canonical evidence; ArgumentLedger owns attributable debate history; typed references and version transitions connect them without collapsing them |
| Trace | One concrete runtime instance used to prove integration after the reusable design is understood |
| External contract | Typed input or output crossing the Coordinator boundary without exposing another team member's subsystem internals |

## Consequences

- The existing Slide 2 lens map and Slide 3 sentence-surgery composition cannot remain as full pages; selected fragments may be reused only inside the final trace.
- The current Slide 1 flat protocol chain must be replaced by a layered ownership diagram with a thin runtime path overlay.
- Sentence-length assertion headlines move into speaker narration or small takeaway lines. Page titles become short object names.
- The real artifact-backed trace remains essential implementation evidence. The Inspector screenshot is omitted from the final four pages.

## Decision Record

G15: Option B formally supersedes ADR-001 G9 for the defense deck. The real claim remains the final integration trace, but it no longer organises Slides 1-3.

## G16 - Slide 1 Organising Grammar

Status: accepted - Option A

### A. Layered Architecture With A Runtime Overlay

Place one large Coordinator boundary between narrow external input/output contract rails. Inside it, use four horizontal responsibility bands:

`Fusion Control Graph -> Dual-Chamber Deliberation -> Governed State -> Coordinator Artifact`

The Governed State band contains two clearly separate owners: `EvidenceCore` and `ArgumentLedger`. A thin numbered path overlays the bands to show contributions, EvidenceViews, argument acts, the bounded follow-up edge, and artifact release.

Strength: the audience can reconstruct both the stable subsystem and its motion. It also uses the HKU body's wide, shallow canvas efficiently.

### B. Nested Ownership Map

Place EvidenceCore and ArgumentLedger at the centre, surround them with the Debate chambers, and wrap the entire group with the Fusion control boundary.

Strength: ownership and containment are visually strong.

Risk: runtime direction, the only legal return edge, and final artifact release require crossing arrows and become harder to narrate.

### C. Eight-Node Runtime Swimlane

Draw the exact parent LangGraph stages from planning to finalisation, with state stores below the lane.

Strength: closest to code execution order.

Failure: repeats the current flat-pipeline problem and makes every stage appear equally important.

Recommended answer: A. Use architecture as the dominant grammar and runtime as a secondary overlay. Do not make the eight graph nodes the primary visual objects.

Decision: use Option A. Slide 1 is a layered architecture diagram with one secondary runtime overlay; the exact eight-node execution lane is not the dominant visual.

## G17 - Slide 2 Organising Principle

Status: accepted - agent topology carries plain-language safeguards

The implemented Debate controls are fixed by code and tests:

1. `Independence`: four role-selected agents form positions concurrently in sealed contexts before publication.
2. `Materiality`: deterministic risk signals select at most six material claims for independent Skeptic and Methodologist review.
3. `Accountability`: each challenge returns to the position's original proposer, who must rebut, revise, concede, abstain, or request evidence; prior positions and triggers remain recorded.
4. `Conservative adjudication`: anonymous Primary and Review Judges receive original/reversed argument order; incompatible verdicts become weaken, needs-search, or unresolved rather than forced consensus.

Canonical source identity, exact excerpts, support/contradiction relations, quality attributes, and versioned writes are explained on Slide 3. Slide 2 must not re-teach them.

### A. Four Protocol Safeguards

Organise the page around `Independence -> Materiality -> Accountability -> Conservative Adjudication`. Use a light sequence line only to establish order.

Strength: explains why this is a designed debate protocol rather than role-play or an agent roster.

### B. Agent Organisation Chart

Organise the page around four perspective roles, two reviewers, the proposer, and two judges.

Strength: makes participants obvious.

Failure: agent count appears to be the contribution; the reusable safeguards and failure controls become annotations.

### C. Round-By-Round Timeline

Organise the page as opening -> review -> response -> adjudication.

Strength: easy to narrate.

Failure: duplicates the runtime path already overlaid on Slide 1 and risks another generic flow diagram.

Recommended answer: A. Present the four safeguards as the main design contribution, with role names and order as supporting labels.

Decision: combine B's agent topology with A's design meaning. The main visual is not a roster or four detached principle cards. It is one relationship map:

- `Perspective Chamber`: four role-selected agents, with the selected claim's original proposer highlighted inside the chamber;
- `Evidence Review Chamber`: independent Skeptic and Methodologist nodes;
- a routed return edge from the reviewers to that original proposer;
- `Paired Adjudication`: Primary and Review Judges receiving anonymised original/reversed argument order.

Place four short safeguards on the relationships they govern:

1. `Form views independently`
2. `Challenge key claims`
3. `Defend or revise`
4. `Disagreement may remain`

The original proposer is one of the four Perspective agents, not a fifth role. Slide 2 contains no detached bottom constraint rail, but this does not reduce its logical density. `Isolated contexts` remains visible because it explains independent formation; same-model deployment, claim/call caps, and cycle counts are omitted from the slides. Plain language means shorter wording and clearer hierarchy, not less causal or architectural content.

## G18 - Slide 3 Visual Carrier

Status: accepted under G19 - EvidenceCore and ArgumentLedger are both required

The EvidenceGraph-only proposal is rejected because it explains evidence construction but not the Coordinator's key dual-state design. The page must show both canonical evidence state and attributable argument state, while keeping their different functions unmistakable.

### Revised Main Visual

Use two linked but visibly different records:

```text
EVIDENCECORE / EVIDENCEGRAPH              ARGUMENTLEDGER
what the sources support                  how the claim was debated

discoveries -> source -> excerpt          position -> challenge
                         |                           -> revision
                  supports / contradicts             -> verdict
                         v
                       claim
```

EvidenceCore receives more visual weight because this remains the Evidence page. It shows canonical source identity, exact excerpts, support/contradiction relations, quality/freshness/sufficiency attributes, and a single-writer version marker.

ArgumentLedger shows only the compact attributable chain `position -> challenge -> revision -> verdict`, with append-only history.

Three cross-record rules make the architecture useful:

1. Every argument references a `claim ID + excerpt IDs + evidence version`.
2. Debate can append argument history but cannot rewrite canonical evidence.
3. Missing evidence creates a typed request; the single writer creates `EvidenceCore vN+1`, then only affected claims are reconsidered.

No Agent roster is repeated on this page. The ledger is shown as a state record, not as another Debate diagram.

Recommended answer: use this dual-state design so Slide 3 explains both the evidence graph and the audit trail that depends on it.

## G19 - Four-Slide Coherence Contract

Status: accepted with G20 presentation refinements

The deck must be designed as one closed-loop explanation, not four individually optimised pages:

1. `Coordinator Architecture` establishes ownership, layers, the two state records, the bounded control loop, and the release artifact.
2. `Debate Protocol` zooms into the deliberation layer and explains how independent positions become targeted challenges, author-owned revisions, and non-forced paired judgment.
3. `Evidence & Argument State` zooms into the state layer and explains how EvidenceGraph and ArgumentLedger remain separate, reference-bound, versioned, and connected by typed evidence requests.
4. `One Claim, End to End` zooms back out and moves one real claim through the exact architecture, Debate protocol, dual-state transition, adjudication, and artifact release.

Every page must introduce objects reused by the next page, and Slide 4 must visibly recombine the objects introduced on Slides 1-3. Concise titles and labels must reduce reading load without deleting implementation truth, constraints, or causal relationships.

## G20 - Runtime Detail Abstraction

Status: accepted

- `v337` and `v453` are internal EvidenceBlackboard event-sequence versions, not model versions, source counts, or research rounds.
- Replace them in visible slide language with `Initial Evidence Snapshot` and `Updated Evidence Snapshot`; the oral explanation may state that both are immutable versioned states.
- Keep `isolated contexts` on Slide 2 because it explains how positions remain independent before publication.
- Omit visible same-model routing, maximum material-claim count, evidence-cycle count, and LLM-call cap. They remain implementation diagnostics and viva backup answers, not primary explanatory content.

Recommended answer: accept these abstractions. They preserve the architecture while removing details that invite the wrong questions.

Decision: locked. Visible slides use `Initial Evidence Snapshot` and `Updated Evidence Snapshot`, retain `isolated contexts`, omit model configuration and numeric runtime parameters, and make no multi-model claim.

## G21 - Slide 3 Visual Weight

Status: accepted - Option A

### A. EvidenceCore 60 / ArgumentLedger 40

EvidenceCore is the larger graph region and carries canonicalisation, exact excerpts, support/contradiction, quality, sufficiency, and version ownership. ArgumentLedger is a substantial but narrower append-only region that carries position, challenge, revision, verdict, and reference tokens.

Strength: preserves the page's Evidence focus while still making the dual-state architecture and audit trail fully understandable.

### B. Equal 50 / 50 Regions

Both records receive equal geometry.

Strength: emphasises architectural symmetry.

Risk: visually suggests that agent argument history is co-equal with canonical evidence, weakening the ownership hierarchy.

Recommended answer: A. Both records remain necessary, but EvidenceCore should be visually dominant because ArgumentLedger depends on EvidenceGraph IDs and versions, not the reverse.

Decision: EvidenceCore receives approximately 60 percent of the main visual area and ArgumentLedger approximately 40 percent. Both remain structurally complete.

## G22 - Slide 4 Implementation Proof

Status: accepted - Option B

The existing full Debate Inspector screenshot is not projection-safe when reduced. Its useful claim details become unreadable, and its header exposes model/call parameters intentionally omitted by G20.

### A. Trace 80 / Targeted Real UI Crop 20

Keep the five-checkpoint end-to-end trace as the dominant visual. Capture a new real Inspector crop showing only the selected claim's revision, paired verdicts, and final output category. Exclude the global status/configuration header.

Strength: proves implementation without allowing the UI to replace the architecture story.

### B. Pure Presentation Trace

Use no product screenshot.

Strength: maximum projection clarity.

Risk: loses a compact visual proof that the trace exists in the implemented Proof surface.

### C. Large Inspector Split

Give approximately half the page to the full Inspector.

Failure: unreadable text, unwanted runtime parameters, and the frontend appears to be the contribution.

Recommended answer: A. Use a newly captured, claim-specific crop at approximately 20 percent of the page; never reuse the current full screenshot unchanged.

Decision: use Option B. Slide 4 is a pure presentation-native trace with no Inspector screenshot. Implementation evidence comes from the exact artifact-backed claim wording, challenges, revision, semantic snapshot transition, paired verdicts, final classification, and release boundary.
