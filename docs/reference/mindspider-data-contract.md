# MindSpider Data Contract

The supplied server-data snapshot is `mindspider_capstone_20260711_205632.sql.gz` (254,850,502 bytes, timestamped 2026-07-11 21:17:33). It was inspected as a read-only gzip stream; it was not extracted or imported into the local workstation. The header contains MediaCrawler platform tables and populated rows, including `bilibili_video`.

## Query Mapping

`QueryEngine/tools/mindspider_search.py` reads the following content tables. The mapping follows the repository schema/model definitions and the actual search implementation.

| Platform | Table | Content | Public URL | Published Time |
| --- | --- | --- | --- | --- |
| Xiaohongshu | `xhs_note` | `title`, `desc` | `note_url` | `time` Unix timestamp |
| Douyin | `douyin_aweme` | `title`, `desc` | `aweme_url` | `create_time` Unix timestamp |
| Kuaishou | `kuaishou_video` | `title`, `desc` | `video_url` | `create_time` Unix timestamp |
| Bilibili | `bilibili_video` | `title`, `desc` | `video_url` | `create_time` Unix timestamp |
| Weibo | `weibo_note` | `content` | `note_url` | `create_time` Unix timestamp |
| Tieba | `tieba_note` | `title`, `desc` | `note_url` | `publish_time` datetime |
| Zhihu | `zhihu_content` | `title`, `content_text` | `content_url` | `created_time` datetime |

Searches also preserve `source_keyword` and the source table. Rows without a public URL receive a deterministic URI:

```text
mindspider://<platform>/<table>/<sha1(platform, table, content, publish_time)>
```

This prevents unrelated URL-less posts from the same table being collapsed into one canonical source.

## Evidence Mapping

```text
MindSpider row
-> QueryEngine SourceItem
-> EvidenceCandidate (canonical source identity)
-> AcquisitionObservation (task + query + provider + retrieval time)
-> EvidenceCore quality / source span / claim audit
```

A source can be found by multiple subqueries, tasks, or agents. Those discoveries remain separate `AcquisitionObservation` records; `NormalizedItem.retrieval_query` exists only as a compatibility field and is not the authoritative provenance record.

## Runtime Controls

| Setting | Default | Effect |
| --- | --- | --- |
| `COORDINATOR_ENABLE_MINDSPIDER_DB` | `false` | Enables read-only QueryEngine searches over existing crawl tables. |
| `COORDINATOR_ALLOW_MINDSPIDER_CRAWL_TRIGGER` | `false` | Separately permits stale/missing-data enrichment to start BroadTopicExtraction. |

The analysis endpoint must not implicitly start a crawler merely because a table is empty or stale. Crawl scheduling, credentials, rate limits, and collection compliance remain operational concerns outside the evidence-fusion request.

## Limitations

- Keyword `LIKE` retrieval is not a substitute for a full-text or vector index.
- Crawl timestamps, publication timestamps, and query retrieval timestamps have different meanings and must not be merged.
- Platform samples are observable collected data, not a representative population survey.
- The dump demonstrates available data, not freshness guarantees for future runs.
