import argparse
import csv
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


ROOT = Path(__file__).resolve().parent
DEFAULT_TIMEOUT = 90


@dataclass
class ProviderResult:
    provider: str
    provider_type: str
    engine: str
    case_id: str
    ok: bool
    latency_seconds: float
    output_text: str
    normalized: Dict[str, Any]
    metrics: Dict[str, float]
    error: str = ""


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_api_key(provider: Dict[str, Any]) -> Optional[str]:
    key = provider.get("api_key")
    if key:
        return key
    env_name = provider.get("api_key_env")
    if env_name:
        return os.getenv(env_name)
    return None


def enabled(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if item.get("enabled")]


def call_openai_compatible(provider: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    api_key = get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"Missing API key for {provider['name']}")

    base_url = str(provider["base_url"]).rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": case["system"]},
            {"role": "user", "content": case["user"]},
        ],
        "temperature": case.get("temperature", 0.2),
        "max_tokens": case.get("max_tokens", 1000),
    }
    started = time.perf_counter()
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=provider.get("timeout", DEFAULT_TIMEOUT),
    )
    latency = time.perf_counter() - started
    raise_for_status_with_body(response)
    data = response.json()
    text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage") or {}
    return {"latency": latency, "text": text, "raw": data, "usage": usage}


def call_tavily(provider: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    api_key = get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"Missing API key for {provider['name']}")

    started = time.perf_counter()
    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Content-Type": "application/json"},
        json={
            "api_key": api_key,
            "query": case["query"],
            "max_results": case.get("max_results", 8),
            "search_depth": provider.get("search_depth", "basic"),
            "include_answer": True,
            "include_raw_content": False,
            "topic": "general",
        },
        timeout=provider.get("timeout", 45),
    )
    latency = time.perf_counter() - started
    raise_for_status_with_body(response)
    data = response.json()
    return {"latency": latency, "raw": data, "normalized": normalize_tavily(data)}


def call_anspire(provider: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    api_key = get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"Missing API key for {provider['name']}")

    started = time.perf_counter()
    response = requests.get(
        provider.get("base_url", "https://plugin.anspire.cn/api/ntsearch/search"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        },
        params={
            "query": case["query"],
            "top_k": case.get("max_results", 8),
            "Insite": "",
            "FromTime": "",
            "ToTime": "",
        },
        timeout=provider.get("timeout", 45),
    )
    latency = time.perf_counter() - started
    raise_for_status_with_body(response)
    data = response.json()
    return {"latency": latency, "raw": data, "normalized": normalize_anspire(data)}


def call_bocha(provider: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    api_key = get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"Missing API key for {provider['name']}")

    started = time.perf_counter()
    response = requests.post(
        provider.get("base_url", "https://api.bocha.cn/v1/ai-search"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        },
        json={
            "query": case["query"],
            "count": case.get("max_results", 8),
            "answer": True,
            "stream": False,
        },
        timeout=provider.get("timeout", 45),
    )
    latency = time.perf_counter() - started
    raise_for_status_with_body(response)
    data = response.json()
    return {"latency": latency, "raw": data, "normalized": normalize_bocha(data)}


def raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        if len(body) > 500:
            body = body[:500] + "..."
        raise requests.HTTPError(
            f"{exc}; response_body={body or '<empty>'}",
            response=response,
        ) from exc


def normalize_tavily(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer": data.get("answer") or "",
        "results": [
            {
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "snippet": item.get("content") or "",
                "date": item.get("published_date") or "",
                "score": item.get("score"),
            }
            for item in data.get("results", [])
        ],
        "images": data.get("images") or [],
    }


def normalize_anspire(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer": "",
        "results": [
            {
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "snippet": item.get("content") or "",
                "date": item.get("date") or "",
                "score": item.get("score"),
            }
            for item in data.get("results", [])
        ],
        "images": [],
    }


def normalize_bocha(data: Dict[str, Any]) -> Dict[str, Any]:
    answer = ""
    results = []
    images = []
    cards = []
    for msg in data.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = parse_jsonish(msg.get("content"))
        msg_type = msg.get("type")
        content_type = msg.get("content_type")
        if msg_type == "answer":
            answer = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        elif msg_type == "source" and content_type == "webpage" and isinstance(content, dict):
            for item in content.get("value", []):
                results.append(
                    {
                        "title": item.get("name") or "",
                        "url": item.get("url") or "",
                        "snippet": item.get("snippet") or "",
                        "date": item.get("dateLastCrawled") or "",
                        "score": None,
                    }
                )
        elif msg_type == "source" and content_type == "image":
            images.append(content)
        elif msg_type == "source":
            cards.append({"type": content_type, "content": content})
    return {"answer": answer, "results": results, "images": images, "cards": cards}


def parse_jsonish(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def extract_json_object(text: str) -> Optional[Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_positions = [pos for pos in [cleaned.find("{"), cleaned.find("[")] if pos >= 0]
    if not start_positions:
        return None
    start = min(start_positions)
    for end in range(len(cleaned), start, -1):
        candidate = cleaned[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def score_latency(seconds: float, fast: float = 3.0, slow: float = 30.0) -> float:
    if seconds <= fast:
        return 5.0
    if seconds >= slow:
        return 1.0
    return round(5.0 - 4.0 * ((seconds - fast) / (slow - fast)), 2)


def estimate_llm_cost(provider: Dict[str, Any], usage: Dict[str, Any]) -> float:
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    input_price = float(provider.get("input_price_per_1m") or 0)
    output_price = float(provider.get("output_price_per_1m") or 0)
    return (prompt_tokens / 1_000_000 * input_price) + (completion_tokens / 1_000_000 * output_price)


def score_cost(cost: float) -> float:
    if cost <= 0:
        return 3.0
    if cost <= 0.001:
        return 5.0
    if cost <= 0.005:
        return 4.0
    if cost <= 0.02:
        return 3.0
    if cost <= 0.08:
        return 2.0
    return 1.0


def auto_score_llm(provider: Dict[str, Any], case: Dict[str, Any], text: str, latency: float, usage: Dict[str, Any], ok: bool) -> Dict[str, float]:
    if not ok:
        return zero_llm_metrics()

    parsed = extract_json_object(text)
    wants_json = "JSON" in case["user"] or "json" in case["user"]
    expected_fields = case.get("expected_json_fields") or []
    format_score = score_json_format(parsed, wants_json, expected_fields)
    required_terms_score = score_required_terms(text, case.get("required_terms") or [])
    forbidden_penalty = count_forbidden_patterns(text, case.get("forbidden_patterns") or [])
    min_chars = int(case.get("min_chars") or 200)
    max_chars = int(case.get("max_chars") or 0)
    length_score = score_length(text, min_chars=min_chars, max_chars=max_chars)
    language_score = 5.0
    if case.get("language") == "zh" and not contains_cjk(text):
        language_score = 2.0
    grounding_score = max(0.0, 4.5 - forbidden_penalty * 1.5)
    cost = estimate_llm_cost(provider, usage)
    return {
        "task_success": round((length_score + required_terms_score) / 2, 2),
        "format_compliance": format_score,
        "grounding": grounding_score,
        "reasoning_quality": round((required_terms_score + grounding_score) / 2, 2),
        "language_quality": language_score,
        "latency": score_latency(latency),
        "cost": score_cost(cost),
        "stability": 3.0,
    }


def score_json_format(parsed: Optional[Any], wants_json: bool, expected_fields: List[str]) -> float:
    if not wants_json and not expected_fields:
        return 5.0
    if parsed is None:
        return 1.0
    if not expected_fields:
        return 5.0
    if not isinstance(parsed, dict):
        return 2.0
    present = sum(1 for field in expected_fields if field in parsed)
    return round(1.0 + 4.0 * present / len(expected_fields), 2)


def score_required_terms(text: str, required_terms: List[str]) -> float:
    if not required_terms:
        return 4.0
    lowered = text.lower()
    matched = sum(1 for term in required_terms if str(term).lower() in lowered)
    return round(5.0 * matched / len(required_terms), 2)


def count_forbidden_patterns(text: str, patterns: List[str]) -> int:
    default_patterns = ["我查到", "根据公开资料", "据网上信息", "作为AI", "无法访问互联网"]
    all_patterns = list(patterns) + default_patterns
    return sum(1 for pattern in all_patterns if re.search(pattern, text))


def score_length(text: str, min_chars: int = 200, max_chars: int = 0) -> float:
    length = len(text.strip())
    if length == 0:
        return 0.0
    score = 5.0 if length >= min_chars else round(5.0 * length / max(1, min_chars), 2)
    if max_chars and length > max_chars:
        over_ratio = (length - max_chars) / max_chars
        score = max(1.0, score - min(3.0, over_ratio * 5.0))
    return round(score, 2)


def zero_llm_metrics() -> Dict[str, float]:
    return {
        "task_success": 0.0,
        "format_compliance": 0.0,
        "grounding": 0.0,
        "reasoning_quality": 0.0,
        "language_quality": 0.0,
        "latency": 0.0,
        "cost": 0.0,
        "stability": 0.0,
    }


def auto_score_search(case: Dict[str, Any], normalized: Dict[str, Any], latency: float, ok: bool) -> Dict[str, float]:
    if not ok:
        return zero_search_metrics()

    results = normalized.get("results") or []
    text_blob = " ".join(
        f"{item.get('title', '')} {item.get('snippet', '')} {item.get('url', '')}" for item in results
    ).lower()
    expected_terms = [str(term).lower() for term in case.get("expected_terms", [])]
    matched = sum(1 for term in expected_terms if term in text_blob)
    answer_blob = str(normalized.get("answer") or "").lower()
    answer_matched = sum(1 for term in expected_terms if term in answer_blob)
    relevance = 5.0 if not expected_terms else round(5.0 * max(matched, answer_matched) / len(expected_terms), 2)
    coverage = min(5.0, round(len(results) / max(1, case.get("max_results", 8)) * 5.0, 2))
    date_count = sum(1 for item in results if item.get("date"))
    metadata = min(5.0, 2.5 + (date_count / max(1, len(results)) * 2.5)) if results else 0.0
    domains = {domain_from_url(item.get("url", "")) for item in results if item.get("url")}
    source_quality = min(5.0, 2.0 + len(domains) * 0.45) if domains else 1.0
    freshness = 3.0 + min(2.0, date_count * 0.5)
    has_answer_or_cards = bool(normalized.get("answer")) or bool(normalized.get("cards"))
    parseability = 5.0 if isinstance(normalized, dict) and "results" in normalized else 2.0
    metadata = min(5.0, metadata + (0.5 if has_answer_or_cards else 0.0))
    return {
        "relevance": relevance,
        "freshness": min(5.0, freshness),
        "source_quality": source_quality,
        "coverage": coverage,
        "metadata_quality": round(metadata, 2),
        "latency": score_latency(latency, fast=2.0, slow=20.0),
        "cost": 3.0,
        "parseability": parseability,
    }


def zero_search_metrics() -> Dict[str, float]:
    return {
        "relevance": 0.0,
        "freshness": 0.0,
        "source_quality": 0.0,
        "coverage": 0.0,
        "metadata_quality": 0.0,
        "latency": 0.0,
        "cost": 0.0,
        "parseability": 0.0,
    }


def domain_from_url(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1).lower() if match else ""


def weighted_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    return round(sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items()) / total_weight, 3)


def result_to_dict(result: ProviderResult) -> Dict[str, Any]:
    return {
        "provider": result.provider,
        "provider_type": result.provider_type,
        "engine": result.engine,
        "case_id": result.case_id,
        "ok": result.ok,
        "latency_seconds": result.latency_seconds,
        "output_text": result.output_text,
        "normalized": result.normalized,
        "metrics": result.metrics,
        "error": result.error,
    }


def progress_line(message: str) -> None:
    print(message, flush=True)


def short_error(exc: Exception, limit: int = 220) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def run_llm_cases(providers: List[Dict[str, Any]], cases: List[Dict[str, Any]], repetitions: int) -> List[ProviderResult]:
    results = []
    total = len(providers) * len(cases) * repetitions
    current = 0
    for provider in providers:
        for case in cases:
            for repetition in range(repetitions):
                current += 1
                case_id = case["id"] if repetitions == 1 else f"{case['id']}#r{repetition + 1}"
                progress_line(f"[LLM {current}/{total}] {provider['name']} -> {case_id}")
                try:
                    call = call_openai_compatible(provider, case)
                    metrics = auto_score_llm(
                        provider,
                        case,
                        call["text"],
                        call["latency"],
                        call.get("usage") or {},
                        ok=True,
                    )
                    for engine in case["engines"]:
                        results.append(
                            ProviderResult(
                                provider=provider["name"],
                                provider_type="llm",
                                engine=engine,
                                case_id=case_id,
                                ok=True,
                                latency_seconds=call["latency"],
                                output_text=call["text"],
                                normalized={"usage": call.get("usage") or {}},
                                metrics=metrics,
                            )
                        )
                    progress_line(f"  OK {provider['name']} -> {case_id} ({call['latency']:.2f}s)")
                except Exception as exc:
                    progress_line(f"  FAIL {provider['name']} -> {case_id}: {short_error(exc)}")
                    for engine in case["engines"]:
                        results.append(
                            ProviderResult(
                                provider=provider["name"],
                                provider_type="llm",
                                engine=engine,
                                case_id=case_id,
                                ok=False,
                                latency_seconds=0.0,
                                output_text="",
                                normalized={},
                                metrics=zero_llm_metrics(),
                                error=str(exc),
                            )
                        )
    return results


def run_search_cases(providers: List[Dict[str, Any]], cases: List[Dict[str, Any]], repetitions: int) -> List[ProviderResult]:
    callers = {"tavily": call_tavily, "anspire": call_anspire, "bocha": call_bocha}
    results = []
    runnable_providers = [provider for provider in providers if callers.get(provider.get("type"))]
    total = len(runnable_providers) * len(cases) * repetitions
    current = 0
    for provider in runnable_providers:
        provider_type = provider.get("type")
        caller = callers.get(provider_type)
        if not caller:
            continue
        for case in cases:
            for repetition in range(repetitions):
                current += 1
                case_id = case["id"] if repetitions == 1 else f"{case['id']}#r{repetition + 1}"
                progress_line(f"[SEARCH {current}/{total}] {provider['name']} -> {case_id}")
                try:
                    call = caller(provider, case)
                    metrics = auto_score_search(case, call["normalized"], call["latency"], ok=True)
                    for engine in case["engines"]:
                        results.append(
                            ProviderResult(
                                provider=provider["name"],
                                provider_type="search",
                                engine=engine,
                                case_id=case_id,
                                ok=True,
                                latency_seconds=call["latency"],
                                output_text=json.dumps(call["normalized"], ensure_ascii=False),
                                normalized=call["normalized"],
                                metrics=metrics,
                            )
                        )
                    progress_line(f"  OK {provider['name']} -> {case_id} ({call['latency']:.2f}s)")
                except Exception as exc:
                    progress_line(f"  FAIL {provider['name']} -> {case_id}: {short_error(exc)}")
                    for engine in case["engines"]:
                        results.append(
                            ProviderResult(
                                provider=provider["name"],
                                provider_type="search",
                                engine=engine,
                                case_id=case_id,
                                ok=False,
                                latency_seconds=0.0,
                                output_text="",
                                normalized={},
                                metrics=zero_search_metrics(),
                                error=str(exc),
                            )
                        )
    return results


def summarize(results: List[ProviderResult], profiles: Dict[str, Any]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[ProviderResult]] = {}
    for result in results:
        grouped.setdefault((result.engine, result.provider_type, result.provider), []).append(result)

    rows = []
    for (engine, provider_type, provider), items in sorted(grouped.items()):
        profile = profiles["engines"].get(engine, {})
        weight_key = f"{provider_type}_weights"
        weights = profile.get(weight_key, {})
        per_case_scores = [weighted_score(item.metrics, weights) for item in items]
        positive_latencies = [item.latency_seconds for item in items if item.latency_seconds > 0]
        avg_latency = round(statistics.mean(positive_latencies), 3) if positive_latencies else 0.0
        rows.append(
            {
                "engine": engine,
                "provider_type": provider_type,
                "provider": provider,
                "score": round(statistics.mean(per_case_scores), 3) if per_case_scores else 0.0,
                "success_rate": round(sum(1 for item in items if item.ok) / len(items), 3),
                "avg_latency_seconds": avg_latency,
                "cases": len(items),
            }
        )
    return rows


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["engine", "provider_type", "provider", "score", "success_rate", "avg_latency_seconds", "cases"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manual_review_csv(path: Path, results: List[ProviderResult], profiles: Dict[str, Any]) -> None:
    fields = [
        "provider_type",
        "engine",
        "provider",
        "case_id",
        "auto_score",
        "success",
        "latency_seconds",
        "needs_manual_score",
        "manual_task_success_0_5",
        "manual_grounding_0_5",
        "manual_format_0_5",
        "manual_notes",
        "error",
        "output_excerpt",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for result in results:
            weights = profiles["engines"].get(result.engine, {}).get(f"{result.provider_type}_weights", {})
            writer.writerow(
                {
                    "provider_type": result.provider_type,
                    "engine": result.engine,
                    "provider": result.provider,
                    "case_id": result.case_id,
                    "auto_score": weighted_score(result.metrics, weights),
                    "success": result.ok,
                    "latency_seconds": round(result.latency_seconds, 3),
                    "needs_manual_score": "yes" if result.ok else "failed_call",
                    "manual_task_success_0_5": "",
                    "manual_grounding_0_5": "",
                    "manual_format_0_5": "",
                    "manual_notes": "",
                    "error": result.error,
                    "output_excerpt": compact_excerpt(result.output_text),
                }
            )


def compact_excerpt(text: str, limit: int = 600) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


def write_recommendations(path: Path, rows: List[Dict[str, Any]]) -> None:
    by_group: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault((row["engine"], row["provider_type"]), []).append(row)

    lines = ["# API Evaluation Recommendations", ""]
    for (engine, provider_type), group in sorted(by_group.items()):
        ranked = sorted(group, key=lambda row: (row["score"], row["success_rate"], -row["avg_latency_seconds"]), reverse=True)
        lines.append(f"## {engine} / {provider_type}")
        lines.append("")
        lines.append("| Rank | Provider | Score | Success Rate | Avg Latency |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for idx, row in enumerate(ranked, start=1):
            lines.append(
                f"| {idx} | {row['provider']} | {row['score']} | {row['success_rate']} | {row['avg_latency_seconds']}s |"
            )
        lines.append("")
        if ranked:
            viable = [row for row in ranked if float(row.get("success_rate", 0.0)) > 0]
            if viable:
                best = viable[0]
                lines.append(f"Recommended default: `{best['provider']}`.")
            else:
                lines.append("No viable provider in this group. All calls failed; fix credentials/config before selecting.")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate candidate APIs for each engine profile.")
    parser.add_argument("--providers", type=Path, default=ROOT / "providers.local.json")
    parser.add_argument("--profiles", type=Path, default=ROOT / "engine_profiles.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "test_cases.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild summary files from an existing raw_results.jsonl without calling APIs.")
    args = parser.parse_args()

    if not args.providers.exists():
        raise SystemExit(f"Provider config not found: {args.providers}. Copy providers.example.json first.")

    providers_config = load_json(args.providers)
    profiles = load_json(args.profiles)
    cases = load_json(args.cases)
    args.out.mkdir(parents=True, exist_ok=True)

    results: List[ProviderResult] = []
    if args.summarize_only:
        raw_path = args.out / "raw_results.jsonl"
        if not raw_path.exists():
            raise SystemExit(f"Cannot summarize only; missing {raw_path}")
        results = load_results_jsonl(raw_path)
        raw_rows = [result_to_dict(result) for result in results]
        summary_rows = summarize(results, profiles)
        write_summary_csv(args.out / "summary.csv", summary_rows)
        write_manual_review_csv(args.out / "manual_review.csv", results, profiles)
        write_recommendations(args.out / "recommendations.md", summary_rows)
        print(f"Rebuilt summary files from {raw_path}")
        return 0

    if not args.skip_llm:
        results.extend(
            run_llm_cases(
                enabled(providers_config.get("llm_providers", [])),
                cases.get("llm_cases", []),
                args.repetitions,
            )
        )
    if not args.skip_search:
        results.extend(
            run_search_cases(
                enabled(providers_config.get("search_providers", [])),
                cases.get("search_cases", []),
                args.repetitions,
            )
        )

    raw_rows = [result_to_dict(result) for result in results]
    write_jsonl(args.out / "raw_results.jsonl", raw_rows)
    summary_rows = summarize(results, profiles)
    write_summary_csv(args.out / "summary.csv", summary_rows)
    write_manual_review_csv(args.out / "manual_review.csv", results, profiles)
    write_recommendations(args.out / "recommendations.md", summary_rows)

    print(f"Wrote {len(raw_rows)} raw results to {args.out / 'raw_results.jsonl'}")
    print(f"Wrote summary to {args.out / 'summary.csv'}")
    print(f"Wrote manual review sheet to {args.out / 'manual_review.csv'}")
    print(f"Wrote recommendations to {args.out / 'recommendations.md'}")
    return 0


def load_results_jsonl(path: Path) -> List[ProviderResult]:
    results = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            results.append(
                ProviderResult(
                    provider=row["provider"],
                    provider_type=row["provider_type"],
                    engine=row["engine"],
                    case_id=row["case_id"],
                    ok=bool(row["ok"]),
                    latency_seconds=float(row.get("latency_seconds") or 0.0),
                    output_text=row.get("output_text") or "",
                    normalized=row.get("normalized") or {},
                    metrics=row.get("metrics") or {},
                    error=row.get("error") or "",
                )
            )
    return results


if __name__ == "__main__":
    raise SystemExit(main())
