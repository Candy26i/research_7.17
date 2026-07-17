# research_0703 — Learning When to Commit

Training Qwen3.5-4B/9B Managers that learn the **delegate-or-commit** decision:
at every step, either call one of three frozen specialist sub-agents
(Extractor / Reasoner / Verifier) or commit to an answer.
The stopping policy is trained with GRPO under an **incentive-compatible
anytime reward**; transition rewards and summed draft bonuses remain ablations.

- **System and pipeline**: [`agent_routing/README.md`](agent_routing/README.md)
- **Canonical in-domain commands**:
  [`agent_routing/IN_DOMAIN_EXPERIMENTS.md`](agent_routing/IN_DOMAIN_EXPERIMENTS.md)
- **Additional experiment notes**:
  [`agent_routing/EXPERIMENTS.md`](agent_routing/EXPERIMENTS.md)

Snapshot lineage: forked from `research_6.8` (rule_applier era); synced to the
2026-07-03 codebase. Do not reuse artifacts from the old snapshot without
checking the migration and split policies.
