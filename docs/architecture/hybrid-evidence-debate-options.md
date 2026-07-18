# Hybrid Multi-Agent + EvidenceCore Architecture Options

Status: divergence set for grilling. No option is selected.

Grilling update: Option C, Dual-Chamber Deliberation, is selected as the primary topology. Its internal role selection, communication pattern, model allocation, and adjudication mechanism remain open; composable variants are still under comparison.

## 1. Comparison Criteria

Each option is compared on:

- genuine independence and visible multi-agent collaboration;
- fit for public-opinion analysis rather than binary fact checking;
- strength of EvidenceCore integration;
- ability to preserve dissent and belief revision;
- resistance to conformity and judge bias;
- latency/API cost on the current deployment profile;
- implementation fit with the active Fusion graph;
- traceability, testability, and four-slide explanatory power.

## 2. Option A - Fixed Analytical Council

```text
Evidence snapshot
  -> four sealed lens agents in parallel
  -> each agent independently reviews selected peer claims
  -> original agents rebut/revise
  -> blinded judge
```

The four agents retain recognisable public-opinion lenses such as factual/technical, stakeholder/public, media/narrative, and policy/ethics.

Strengths: clearest continuation of the previous four-perspective story; easy to show in a defense; holistic analysis; fixed bounded cost.

Weaknesses: persona prompts may create stylistic rather than epistemic diversity; all-to-all review grows quickly; fixed lenses can be irrelevant to some topics; one judge remains a bottleneck.

Implementation shape: four parallel deliberator subgraphs after `evidence_reduce`, selective peer review, one revision round, judge node, then existing synthesis.

## 3. Option B - Claim Court

```text
Material claim
  -> evidence-bound proponent
  -> evidence-bound opponent
  -> methodologist
  -> blinded judge
  -> accept | weaken | reject | needs_search | unresolved
```

Agents are assigned per claim instead of keeping stable topic personas. Proponent and opponent must cite spans; the methodologist checks sampling, source independence, inference scope, and missing evidence.

Strengths: strongest claim-level adversarial testing; natural fit with EvidenceGraph and typed follow-up retrieval; easy to test per claim; clear revision/verdict provenance.

Weaknesses: can flatten rich stakeholder perspectives into artificial for/against positions; expensive if applied to every claim; repeated claim courts can lose whole-topic synthesis; visual story risks looking like fact checking only.

Implementation shape: route only top material/contested claims into courts; parallel courts; shared evidence requests return through the existing reducer.

## 4. Option C - Dual-Chamber Deliberation

```text
Evidence snapshot
  -> Perspective Chamber: independent domain lenses propose interpretations
  -> Evidence Chamber: skeptic + methodologist attack claims and assumptions
  -> original agents revise
  -> judge/jury records claim verdicts and preserved dissent
```

The first chamber maximizes coverage and viewpoint diversity. The second maximizes adversarial verification. Analytical lens and protocol function remain visibly separate.

Strengths: best conceptual fusion of public-opinion plurality and EvidenceCore rigor; avoids forcing every lens to be a permanent supporter/opponent; excellent defense narrative; supports factual and normative claim modes.

Weaknesses: highest fixed orchestration complexity; more LLM calls; requires careful routing so critics do not review every low-value claim; may be too much for a short live run without adaptive pruning.

Implementation shape: 3-4 sealed perspective agents, one skeptical critic and one methodological critic (possibly parallel), targeted revision, then a judge or small jury.

## 5. Option D - Sparse Argument-Graph Debate

```text
Independent positions
  -> Argument Ledger support/attack graph
  -> router selects novel/material unresolved attacks
  -> only affected agents receive challenges
  -> judge consumes claim subgraphs, not a full transcript
```

Communication topology is selected dynamically from the argument graph rather than all-to-all or fixed rounds.

Strengths: strongest research/engineering novelty; avoids redundant token exchange; preserves minority arguments; scales better with claim count; makes EvidenceCore and deliberation graph jointly inspectable.

Weaknesses: hardest router and stopping policy; a bad materiality score can suppress a decisive minority challenge; less immediately intuitive to examiners; more implementation and evaluation work.

Implementation shape: add `ArgumentLedger`, novelty/materiality scoring, targeted challenge tasks, affected-claim re-evaluation, and graph-derived stopping.

## 6. Option E - Asynchronous Blackboard Swarm

```text
EvidenceCore + ArgumentLedger
  <-> agents subscribe to gaps/claims
  <-> agents propose, challenge, retrieve, and revise asynchronously
  -> stop when no material work remains or budget expires
```

Agents choose work from a shared queue rather than following fixed debate phases.

Strengths: flexible, extensible, and closest to a society-of-agents architecture; handles partial failures; agents can specialize deeply; good for long-running research.

Weaknesses: hardest to guarantee equal participation, independence, deterministic replay, and bounded cost; susceptible to loops and premature self-selection; much harder to explain in four slides; excessive for the current capstone runtime unless tightly constrained.

Implementation shape: typed subscriptions, leases, task queue, idempotent acts, semantic stopping, and stronger persistent storage than the current in-memory run repositories.

## 7. Option F - Independent Panel + Jury Baseline

```text
Evidence snapshot
  -> independent agents produce sealed claim portfolios
  -> no peer messages
  -> blinded jury aggregates and preserves dissent
```

This is an ensemble baseline, not a full debate.

Strengths: maximum first-round independence; low coordination failure; bounded latency; useful experimental baseline against interactive variants; straightforward to implement.

Weaknesses: no rebuttal, correction, or belief revision; weak visible collaboration; jury must infer conflicts without agents answering challenges; does not satisfy the desired debate highlight as the final architecture.

Implementation shape: parallel role agents, normalized claim portfolios, order-shuffled jury, artifact projection.

## 8. Option G - Moderator-Simulated Cross-Examination

```text
Four independent openings
  -> one moderator writes all peer responses
  -> one arbitrator summarizes
```

This is the legacy pattern and a rejected baseline.

Strengths: cheap and simple; produces debate-shaped text.

Weaknesses: debaters do not actually challenge or revise; moderator can invent positions/evidence; weak independence after round one; no source-span protocol; transcript can look collaborative while being centrally generated.

## 9. Comparison Matrix

Scale: 1 weak/low, 5 strong/high. For Cost and Complexity, 5 means expensive/complex.

| Option | Real interaction | Public-opinion fit | Evidence rigor | Dissent/revision | Cost | Complexity | Defense clarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A Fixed Council | 4 | 5 | 4 | 4 | 3 | 3 | 5 |
| B Claim Court | 5 | 3 | 5 | 5 | 4 | 3 | 4 |
| C Dual Chamber | 5 | 5 | 5 | 5 | 5 | 4 | 5 |
| D Sparse Argument Graph | 5 | 5 | 5 | 5 | 3 | 5 | 3 |
| E Blackboard Swarm | 5 | 4 | 4 | 5 | 5 | 5 | 2 |
| F Independent Panel | 1 | 4 | 4 | 2 | 2 | 2 | 3 |
| G Moderator Simulation | 2 | 4 | 1 | 2 | 2 | 1 | 3 |

## 10. Composable Variants To Explore

The options are not mutually exclusive components:

- A + B: stable lens agents, but only material disputed claims enter a claim court.
- C + D: dual chambers with sparse challenge routing rather than full broadcast.
- D + F: sealed independent panel provides the baseline positions, then only graph-selected disputes become interactive.
- A/C + heterogeneous jury: preserve visible expert collaboration while reducing single-judge dependence.
- Any option + deterministic evidence gate: missing/invalid span references are rejected before LLM adjudication.

No combination is accepted until the grilling decisions resolve visible identity, model heterogeneity, judge topology, and runtime budget.
