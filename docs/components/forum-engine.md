# ForumEngine

ForumEngine provides legacy log monitoring and host-speech utilities. It is not part of the final Signal Studio analysis path, but its APIs and tests remain in the repository.

## Implementation

| Path | Purpose |
| --- | --- |
| `ForumEngine/monitor.py` | Log monitor, parsing, JSON extraction, start/stop helpers. |
| `ForumEngine/llm_host.py` | Forum host LLM wrapper. |
| `tests/test_monitor.py` | Parser regression coverage. |
| `tests/forum_log_test_data.py` | Old/new log format fixtures and real examples. |
| `logs/forum.log` | Runtime log file when monitor is active. |

## Signal Studio Behavior

`initialize_system_components()` stops ForumEngine monitor and marks it as stopped. This keeps Signal Studio as the primary operator path while preserving ForumEngine APIs for compatibility review.

## Legacy API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/forum/start` | Start ForumEngine monitoring. |
| `GET /api/forum/stop` | Stop ForumEngine monitoring. |
| `GET /api/forum/log` | Read and parse `logs/forum.log`. |
| `POST /api/forum/log/history` | Read log lines from a byte position. |
| `GET /api/output/forum` | Read forum output through the legacy process output API. |

## Parser Responsibilities

| Responsibility | Notes |
| --- | --- |
| Detect target summary lines | Supports older `[HH:MM:SS]` and newer loguru formats. |
| Extract JSON fragments | Handles single-line and multi-line JSON-like outputs. |
| Filter low-value logs | Avoids search-node noise, parser errors, and irrelevant logs. |
| Preserve useful content | Keeps summary/reflection content for forum-style review. |

## Related Documents

- [Runtime Flow](../architecture/runtime-flow.md)
- [Testing](../quality/testing.md)
- [API Reference](../reference/api.md)
