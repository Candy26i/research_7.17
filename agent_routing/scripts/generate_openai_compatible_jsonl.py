"""Generate JSONL responses through any OpenAI-compatible chat endpoint."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_url", default="http://localhost:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max_tokens", type=int, default=2200)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    def run(row):
        resp = requests.post(
            args.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {args.api_key}"},
            json={
                "model": args.model,
                "messages": row["prompt"],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
            timeout=args.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return {"example_id": row["example_id"], "response": text}

    output = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run, row) for row in rows]
        for future in as_completed(futures):
            output.append(future.result())

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"generated {len(output)} responses -> {args.output}")


if __name__ == "__main__":
    main()
