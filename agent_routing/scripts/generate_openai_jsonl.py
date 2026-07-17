"""Generate JSONL responses from OpenAI using existing chat-prompt JSONL.

Input rows must contain:
  {"example_id": ..., "prompt": [{"role": "system", "content": ...}, ...]}

Output rows are compatible with the local DeepSeek importer:
  {"example_id": ..., "prompt": [...], "response": "..."}
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Set

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error in {path}:{i}: {e}") from e
    return rows


def _completed_example_ids(path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("response"):
                done.add(str(row.get("example_id")))
    return done


def _iter_rows(rows: List[Dict[str, Any]], limit: int, resume: bool, output_file: str) -> Iterable[Dict[str, Any]]:
    done = _completed_example_ids(output_file) if resume else set()
    yielded = 0
    for row in rows:
        if str(row.get("example_id")) in done:
            continue
        yield row
        yielded += 1
        if limit > 0 and yielded >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="Generate at most N rows; 0 means all rows.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Skip example_ids with non-empty responses in output.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install `openai` to use this script: pip install openai") from e

    rows = _read_jsonl(args.input_file)
    todo = list(_iter_rows(rows, limit=args.limit, resume=args.resume, output_file=args.output_file))
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    client = OpenAI(timeout=args.timeout)
    mode = "a" if args.resume else "w"

    with open(args.output_file, mode, encoding="utf-8") as out:
        for idx, ex in enumerate(todo, start=1):
            messages = ex.get("prompt")
            if not isinstance(messages, list):
                out.write(json.dumps({
                    "example_id": ex.get("example_id"),
                    "prompt": messages,
                    "response": "",
                    "error": "missing_or_invalid_prompt",
                }, ensure_ascii=False) + "\n")
                out.flush()
                continue

            last_err = ""
            for attempt in range(args.max_retries + 1):
                try:
                    request = {
                        "model": args.model,
                        "messages": messages,
                        "temperature": float(args.temperature),
                        "max_completion_tokens": int(args.max_tokens),
                    }
                    try:
                        resp = client.chat.completions.create(**request)
                    except Exception as e:
                        if "max_completion_tokens" not in str(e):
                            raise
                        request.pop("max_completion_tokens", None)
                        request["max_tokens"] = int(args.max_tokens)
                        resp = client.chat.completions.create(**request)
                    text = (resp.choices[0].message.content or "").strip()
                    out.write(json.dumps({
                        "example_id": ex.get("example_id"),
                        "prompt": messages,
                        "response": text,
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"[{idx}/{len(todo)}] example_id={ex.get('example_id')} done", flush=True)
                    break
                except Exception as e:
                    last_err = str(e)
                    if attempt < args.max_retries:
                        time.sleep(min(2 ** attempt, 8))
                    else:
                        out.write(json.dumps({
                            "example_id": ex.get("example_id"),
                            "prompt": messages,
                            "response": "",
                            "error": last_err,
                        }, ensure_ascii=False) + "\n")
                        out.flush()
                        print(f"[{idx}/{len(todo)}] example_id={ex.get('example_id')} error={last_err}", flush=True)


if __name__ == "__main__":
    main()
