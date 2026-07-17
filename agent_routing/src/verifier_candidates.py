"""Dependency-light safety checks for verifier candidate artifacts."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List


MIN_VERIFIER_CANDIDATE_COVERAGE = 0.95


def _question_hash(question: str) -> str:
    normalized = re.sub(r"\s+", " ", str(question).strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def load_validated_prediction_map(path: str, rows: List[Any]) -> Dict[int, str]:
    """Load predictions and bind them to rows using ID plus question hash."""
    if not path:
        return {}

    records: Dict[int, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    for record in source_rows:
        eid = record.get("example_id")
        if eid is None:
            continue
        try:
            eid_int = int(eid)
        except Exception:
            continue
        pred = str(
            record.get("pred", record.get("prediction", record.get("answer", ""))) or ""
        ).strip()
        qhash = str(record.get("question_hash") or "").strip()
        normalized = {"pred": pred, "question_hash": qhash}
        previous = records.get(eid_int)
        if previous is not None and previous != normalized:
            raise ValueError(
                f"Conflicting manager predictions for example_id={eid_int} in {path}"
            )
        records[eid_int] = normalized

    valid: Dict[int, str] = {}
    missing_hashes: List[int] = []
    hash_mismatches: List[int] = []
    for row in rows:
        eid = int(row.example_id)
        record = records.get(eid)
        if record is None:
            continue
        supplied_hash = record["question_hash"]
        if not supplied_hash:
            missing_hashes.append(eid)
            continue
        if supplied_hash != _question_hash(row.question):
            hash_mismatches.append(eid)
            continue
        pred = record["pred"]
        if pred in row.choices:
            valid[eid] = pred

    if missing_hashes:
        preview = ", ".join(str(x) for x in missing_hashes[:5])
        raise ValueError(
            "Manager predictions are missing question_hash for example_id(s) "
            f"{preview}. Regenerate them with export_base_predictions."
        )
    if hash_mismatches:
        preview = ", ".join(str(x) for x in hash_mismatches[:5])
        raise ValueError(
            "Manager prediction question_hash mismatch for example_id(s) "
            f"{preview}. Refusing to bind predictions from a different split/cache."
        )
    return valid


def require_verifier_candidate_coverage(
    rows: List[Any],
    candidate_map: Dict[int, str],
    minimum: float = MIN_VERIFIER_CANDIDATE_COVERAGE,
) -> Dict[str, Any]:
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum verifier candidate coverage must be between 0 and 1")
    n_rows = len(rows)
    n_valid = sum(
        candidate_map.get(int(row.example_id), "") in row.choices
        for row in rows
    )
    coverage = n_valid / max(1, n_rows)
    if coverage < minimum:
        raise ValueError(
            "Verifier synthesis requires real, parseable manager predictions for at "
            f"least {minimum:.0%} of sampled rows; found {n_valid}/{n_rows} "
            f"({coverage:.1%}). Run export_base_predictions for the same split, seed, "
            "and sample size, then retry. Ground truth and random choices must not be "
            "used as main-experiment substitutes."
        )
    return {
        "n_candidate_requested": n_rows,
        "n_candidate_valid": n_valid,
        "candidate_coverage": coverage,
    }


def validate_candidate_bound_prompt_rows(prompt_rows: List[Dict[str, Any]]) -> None:
    """Require each verifier prompt to contain its declared valid candidate."""
    if not prompt_rows:
        raise ValueError("Verifier prompt JSONL is empty")

    invalid_ids: List[Any] = []
    invalid_hashes: List[Any] = []
    invalid_agent_kinds: List[Any] = []
    invalid_roles: List[Any] = []
    missing_markers: List[Any] = []
    seen_ids = set()
    duplicate_ids = set()
    for src in prompt_rows:
        eid = src.get("example_id")
        try:
            eid_int = int(eid)
        except Exception:
            invalid_ids.append(eid)
            continue
        if eid_int in seen_ids:
            duplicate_ids.add(eid_int)
        seen_ids.add(eid_int)
        if src.get("agent_kind") != "verifier":
            invalid_agent_kinds.append(eid_int)
        if str(src.get("question_hash") or "") != _question_hash(src.get("question") or ""):
            invalid_hashes.append(eid_int)

        candidate = str(src.get("candidate_answer") or "").strip()
        choices = dict(src.get("choices") or {})
        if candidate not in choices:
            invalid_ids.append(eid_int)
            continue
        marker = f"CANDIDATE ANSWER TO AUDIT: {candidate}"
        messages = src.get("prompt") or []
        roles = [str(message.get("role") or "") for message in messages]
        if roles != ["system", "user"]:
            invalid_roles.append(eid_int)
        if not any(
            message.get("role") == "user"
            and marker in str(message.get("content") or "")
            for message in messages
        ):
            missing_markers.append(eid_int)

    if (
        invalid_ids
        or invalid_hashes
        or invalid_agent_kinds
        or invalid_roles
        or duplicate_ids
        or missing_markers
    ):
        raise ValueError(
            "Refusing to import verifier responses without candidate-bound prompts: "
            f"invalid_candidates_or_ids={len(invalid_ids)}, "
            f"invalid_question_hashes={len(invalid_hashes)}, "
            f"invalid_agent_kinds={len(invalid_agent_kinds)}, "
            f"invalid_roles={len(invalid_roles)}, "
            f"duplicate_ids={len(duplicate_ids)}, "
            f"missing_candidate_markers={len(missing_markers)}. Re-export verifier "
            "prompts from matching base_predictions.jsonl."
        )
