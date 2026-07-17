"""Diagnose a CGC (Design A) run from its train_raw_trace.jsonl.

Usage:
    python scripts/diagnose_cgc_trace.py outputs/manager/<ID>/grpo_cgc/train_raw_trace.jsonl

Distinguishes the three candidate collapse mechanisms:
  A. Off-arm attempt tokens receiving no-tool outcomes (pairing self-cancel):
     -> look at mean reward / advantage proxy of off-arm attempters vs
        on-arm attempters vs direct answers in the early window.
  B. Sentinel-induced format failures on the off arm:
     -> off-arm parse-failure and clamp-penalty rates vs on-arm.
  C. LLD / no-headroom:
     -> online dP (correct|used_tools - correct|no_tools) per window; if ~0
        while A and B look clean, the dataset has no routing signal or the
        collapse is token-level.
"""
import json
import sys
from collections import defaultdict


def main(path: str, window: int = 200) -> None:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    cgc = [r for r in recs if r.get("reward_mode") == "cgc"]
    print(f"total records: {len(recs)}   cgc records: {len(cgc)}")
    if not cgc:
        print("!! no reward_mode=='cgc' records — CGC mode did not engage. "
              "Check the [MANAGER_GRPO] CGC mode ON line in the launch log.")
        return

    n_off = sum(1 for r in cgc if r.get("off_arm"))
    n_flat = sum(1 for r in cgc if r.get("flattened"))
    print(f"off_arm ratio: {n_off/len(cgc):.3f}   (expect ~0.5; ~0 means arm "
          f"assignment never fired)")
    print(f"flattened ratio: {n_flat/len(cgc):.3f}")

    def rate(xs):
        xs = list(xs)
        return (sum(xs) / len(xs)) if xs else float("nan")

    print(f"\n{'win':>4} {'n':>5} | {'attempt%':>8} {'off%':>5} | "
          f"{'acc_tool':>8} {'acc_notool':>10} {'dP':>7} | "
          f"{'parse_fail on/off':>18} {'clamp on/off':>14} | "
          f"{'meanR dir/on/off':>20}")
    for w0 in range(0, len(cgc), window):
        win = cgc[w0:w0 + window]
        # attempted = emitted any tool call (executed or blocked)
        att = [r for r in win if (r.get("tool_calls", 0) or 0) > 0
               or (r.get("blocked_tool_calls", 0) or 0) > 0]
        on_att = [r for r in att if not r.get("off_arm")]
        off_att = [r for r in att if r.get("off_arm")]
        used = [r for r in win if r.get("used_tools")]
        nott = [r for r in win if not r.get("used_tools")]
        direct = [r for r in win if not (r.get("tool_calls", 0) or 0)
                  and not (r.get("blocked_tool_calls", 0) or 0)]

        acc_t = rate(bool(r.get("correct")) for r in used)
        acc_n = rate(bool(r.get("correct")) for r in nott)
        pf_on = rate(r.get("pred") is None for r in on_att)
        pf_off = rate(r.get("pred") is None for r in off_att)
        # clamp fingerprint: reward <= -0.15 means the -0.2 format penalty hit
        cl_on = rate(float(r.get("reward", 0)) <= -0.15 for r in on_att)
        cl_off = rate(float(r.get("reward", 0)) <= -0.15 for r in off_att)
        mr_dir = rate(float(r.get("reward", 0)) for r in direct)
        mr_on = rate(float(r.get("reward", 0)) for r in on_att)
        mr_off = rate(float(r.get("reward", 0)) for r in off_att)

        print(f"{w0//window:>4} {len(win):>5} | {len(att)/len(win):>8.2f} "
              f"{rate(bool(r.get('off_arm')) for r in win):>5.2f} | "
              f"{acc_t:>8.3f} {acc_n:>10.3f} {acc_t-acc_n:>7.3f} | "
              f"{pf_on:>8.2f}/{pf_off:<8.2f} {cl_on:>6.2f}/{cl_off:<6.2f} | "
              f"{mr_dir:>6.3f}/{mr_on:<6.3f}/{mr_off:<6.3f}")

    print(
        "\nHow to read:\n"
        "  Mechanism B (sentinel OOD): parse_fail/clamp OFF >> ON in early "
        "windows -> off arm is a format-failure factory; fix the sentinel/"
        "format handling first.\n"
        "  Mechanism A (pairing self-cancel): meanR off-att < meanR direct "
        "while parse_fail is similar across arms -> off-arm attempt tokens "
        "are being punished for no-tool outcomes; exclude off-arm rollouts "
        "from the loss (baseline-only).\n"
        "  Mechanism C (no signal / LLD): dP ~ 0 in all windows with A and B "
        "clean -> no routing headroom on this dataset, or token-level "
        "collapse; move to tool-token gradient masking / different dataset."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
