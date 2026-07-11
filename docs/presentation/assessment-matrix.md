# Assessment Matrix

This matrix maps project goals to implemented evidence, acceptance proof, and operational controls.

## Functional Assessment

| Goal | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Run integrated public-opinion analysis from a topic | Implemented | `/api/coordinator/run`, Signal Studio Home | Uses the configured provider profile. |
| Retrieve and score evidence | Implemented | QueryEngine graph and Proof page | Search provider quality affects output. |
| Compare stance and divergence | Implemented | QueryEngine stance output, Coordinator divergence matrix | Visualized in Proof. |
| Integrate media-oriented research | Implemented, evaluation pending | MediaEngine subgraph, `MediaContribution`, `SectionDossier`, Coordinator artifact | Active path preserves source-bound dossiers/assets; server-side quality and latency comparison has not yet been run. |
| Perform structured claim review | Implemented, evaluation pending | Parent fusion graph, Evidence Blackboard, `AuditDecision` | Proposals, counter-evidence, methodology checks, and judge decisions are evidence-ID-bound; this is not free-form agent debate. |
| Separate facts, opinions, and bias signals | Implemented, evaluation pending | EvidenceCore quality/claim/audit kernels | Output quality still depends on retrieval and must be measured on the server test set. |
| Generate reports | Implemented | ReportEngine graph and `/api/report/*` | Uses the Coordinator artifact contract in the final path. |
| Export HTML, Markdown, PDF | Implemented | ReportEngine endpoints | PDF requires system dependencies. |
| Provide monitoring and traceability | Implemented | Monitor view, Coordinator trace, LangSmith API | Local trace is always available; LangSmith is configurable. |
| Evaluate provider choices | Existing limited evidence | `api_evaluation/` | Existing provider benchmark is not evidence for the new fusion architecture; fusion experiments are deferred. |

## Architecture Assessment

| Quality | Status | Evidence |
| --- | --- | --- |
| Modularity | Strong | Separate engines and graph builders. |
| Traceability | Strong | Coordinator trace, task status, artifact metadata, configurable LangSmith. |
| Maintainability | Good | Typed graph state, centralized settings, documented API. |
| Operability | Good | Setup/runbook/troubleshooting, runtime APIs. |
| Test coverage | Contract-focused | Fusion contract tests plus existing backend regression tests; live provider and end-to-end server tests remain pending. |
| Deployment readiness | Defined | Dockerfile, Compose config, environment template, dependency setup, runbook. |

## Operational Controls

| Operational Area | Control | Evidence | Extension Path |
| --- | --- | --- | --- |
| Provider latency | Configurable providers, timeout settings, benchmark harness | `.env.example`, `config.py`, `api_evaluation/` | Add per-engine automatic provider rotation. |
| Task continuity | Latest Coordinator artifact and generated report outputs persist to disk | `AgentCoordinator/cache/`, `output/` | Persist active task registry in a database. |
| API contract alignment | Markdown reference and OpenAPI YAML cover the final runtime APIs | `docs/reference/api.md`, `docs/reference/openapi.yaml` | Add route-level contract tests in CI. |
| PDF export stack | Docker and platform dependency setup define the WeasyPrint/Pango path | `Dockerfile`, `docs/operations/setup.md` | Add image-based export smoke checks in CI. |
| Artifact lifecycle | Runtime asset boundaries and generated paths are documented | `docs/reference/runtime-assets.md` | Add scheduled retention policy. |
| Shared deployment | Security controls identify protected routes and secret handling | `docs/quality/security-and-safety.md` | Add authentication and role checks for shared environments. |

## Evidence Index

| Evidence Type | Location |
| --- | --- |
| Architecture docs | `docs/architecture/` |
| Component docs | `docs/components/` |
| API contracts | `docs/reference/api.md`, `docs/reference/openapi.yaml` |
| Runtime guides | `docs/operations/` |
| Tests | `tests/` |
| Provider evaluation | `api_evaluation/` and [API Evaluation](../quality/api-evaluation.md) |
| Provider benchmark summary | [Evidence Dashboard](../quality/evidence-dashboard.md) |
| Diagrams | `docs/assets/diagrams/` |

## Overall Assessment

The repository now contains an integrated Query/Media execution architecture rather than only disconnected specialist code. The strongest engineering areas are the explicit contribution contracts, single-writer evidence provenance, stable external artifact, and structured ReportEngine IR. A high mark should not be claimed from implementation alone: live server experiments, latency/cost instrumentation, ablation against Query-only/Media-only/previous-intelligence paths, and multiple-topic qualitative error analysis remain required evidence.
