# Report IR

ReportEngine uses an intermediate representation to validate, repair, and render generated reports.

## Version

| Constant | Value |
| --- | --- |
| `IR_VERSION` | `1.0` |

## Chapter Schema

Each generated chapter JSON includes:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `chapterId` | string | yes | Stable chapter identifier. |
| `title` | string | yes | Chapter title. |
| `anchor` | string | yes | Anchor id. |
| `order` | number | yes | Chapter order. |
| `summary` | string | no | Chapter summary. |
| `blocks` | array | yes | Structured content blocks. |
| `xrefs` | object | no | Cross-references. |
| `widgets` | array | no | Widget identifiers. |
| `footnotes` | array | no | Footnote objects. |
| `errors` | array | no | Generation or validation diagnostics. |
| `metadata` | object | no | Extra chapter metadata. |

## Supported Block Types

| Block Type | Purpose |
| --- | --- |
| `heading` | Section heading with level and anchor. |
| `paragraph` | Inline text runs with marks. |
| `list` | Ordered, bullet, or task list. |
| `table` | Structured rows and cells. |
| `swotTable` | SWOT analysis block. |
| `pestTable` | PEST analysis block. |
| `blockquote` | Quoted or emphasized block content. |
| `engineQuote` | Controlled quote from QueryAgent or MediaAgent. |
| `hr` | Horizontal rule. |
| `code` | Code block. |
| `math` | LaTeX math block. |
| `figure` | Image figure with caption. |
| `callout` | Info/warning/success/danger callout. |
| `kpiGrid` | KPI card grid. |
| `widget` | Chart or custom widget payload. |
| `toc` | Table of contents block. |

## Inline Marks

| Mark | Purpose |
| --- | --- |
| `bold`, `italic`, `underline`, `strike` | Basic formatting. |
| `code` | Inline code. |
| `link` | Hyperlink. |
| `color`, `font`, `highlight` | Styling. |
| `subscript`, `superscript` | Scientific/technical notation. |
| `math` | Inline math. |

## Engine Quote Rules

`engineQuote` has controlled titles:

| `engine` | Required `title` |
| --- | --- |
| `media` | `Multimodal Agent` |
| `query` | `Query Agent` |

Tests under `tests/test_report_engine_sanitization.py` validate these rules and sanitization behavior.

## Renderers

| Renderer | Path | Output |
| --- | --- | --- |
| HTML | `ReportEngine/renderers/html_renderer.py` | Browser-ready report HTML. |
| Markdown | `ReportEngine/renderers/markdown_renderer.py` | Markdown export. |
| PDF | `ReportEngine/renderers/pdf_renderer.py` | PDF bytes/files. |
| Chart/Math helpers | `ReportEngine/renderers/chart_to_svg.py`, `math_to_svg.py` | Embedded visual rendering support. |

## Editing Model

Signal Studio does not edit raw Markdown or raw report HTML as the source of truth. The Edit view maps editable Document IR blocks into TipTap content, keeps complex blocks such as charts and specialized tables as locked previews, then re-renders the updated IR through ReportEngine. HTML, Markdown, and PDF exports are generated from Document IR.

## Error Handling

| Issue | Handling |
| --- | --- |
| Malformed LLM JSON | JSON parser/repair utilities attempt recovery. |
| Invalid blocks | IR validator reports diagnostics. |
| Chart issues | Chart validator and repair service attempt fixes. |
| PDF runtime stack guidance | PDF endpoint reports the WeasyPrint/Pango setup path when the runtime stack needs attention. |

## Related Documents

- [ReportEngine](../components/report-engine.md)
- [API Reference](api.md)
- [Testing](../quality/testing.md)
