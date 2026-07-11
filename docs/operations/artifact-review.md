# Artifact Review

Use this page to review the project through its checked-in evidence: UI screenshots, architecture diagrams, cached Coordinator artifacts, generated report examples, API contracts, and provider evaluation results.

## Review Modes

| Mode | Material | What To Inspect |
| --- | --- | --- |
| Documentation review | `README.md`, `docs/` | Product scope, architecture, operations, quality evidence, and defense narrative. |
| UI review | `docs/assets/screenshots/` | Signal Studio home, readout, proof, report editor, monitor, settings, and revision views. |
| Artifact review | `AgentCoordinator/cache/coordinator_output_latest.json` | Stable analysis schema consumed by Signal Studio and ReportEngine. |
| Report review | Curated static report examples and generated `output/` files | HTML, Markdown, and PDF report rendering quality. |
| Contract review | `docs/reference/api.md`, `docs/reference/openapi.yaml` | Coordinator, ReportEngine, runtime, config, and observability APIs. |
| Evaluation review | `api_evaluation/results_full_r3/summary.csv` | Provider score, success rate, latency, and recommended runtime profile. |

## Cached Coordinator Artifact

Signal Studio and ReportEngine share the latest Coordinator artifact:

```text
AgentCoordinator/cache/coordinator_output_latest.json
```

When the Flask runtime is running, load it through:

```powershell
curl.exe http://127.0.0.1:5000/api/coordinator/latest
```

The Readout, Proof, and Monitor views use the same artifact shape documented in [Coordinator Output Schema](../reference/coordinator-output-schema.md).

## Report Examples

Curated generated report examples live under:

```text
static/v2_report_example/
```

Use these files to inspect report layout, structured sections, table rendering, chart rendering, and export formatting. The ReportEngine implementation and export endpoints are documented in [ReportEngine](../components/report-engine.md), [Report IR](../reference/report-ir.md), and [API Reference](../reference/api.md).

## Review Sequence

| Step | Action | Expected Evidence |
| --- | --- | --- |
| 1 | Read [Project Brief](../overview/project-brief.md). | Product scope and end-to-end workflow are clear. |
| 2 | Open [System Architecture](../architecture/system-architecture.md). | Layered architecture and data handoff are visible. |
| 3 | Open [Screenshots](../overview/screenshots.md). | Final Signal Studio workflow is visible across all main views. |
| 4 | Inspect `coordinator_output_latest.json`. | The analysis artifact follows the documented schema. |
| 5 | Open report examples or generated `output/` artifacts. | Report rendering and export formats are reviewable. |
| 6 | Read [API Evaluation](../quality/api-evaluation.md). | Provider choices are tied to benchmark evidence. |
| 7 | Use [Acceptance Walkthrough](../presentation/acceptance-walkthrough.md). | The defense sequence connects UI, APIs, agents, reports, and tests. |

## Related Documents

- [Setup](setup.md)
- [Runbook](runbook.md)
- [Evidence Dashboard](../quality/evidence-dashboard.md)
- [Defense Brief](../presentation/defense-brief.md)
