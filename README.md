# research_0703 — Learning When to Commit

Training an 8B orchestrator that learns the **delegate-or-commit** decision:
at every step, either delegate a cognitive subtask to one of three frozen
specialist advisors (extractor / reasoner / verifier) or commit to an answer.
The stopping policy is trained with GRPO under an **incentive-compatible
anytime reward** — the two natural alternatives (transition rewards, summed
draft bonuses) are provably exploitable and are kept only as ablation arms.

- **System and pipeline**: [`agent_routing/README.md`](agent_routing/README.md)
- **Full experiment plan (data budgets, hyperparameters, every command)**:
  [`agent_routing/EXPERIMENTS.md`](agent_routing/EXPERIMENTS.md)

Snapshot lineage: forked from `research_6.8` (rule_applier era); synced to the
2026-07-03 codebase. See the migration table at the top of EXPERIMENTS.md
before reusing any artifacts produced by the old snapshot.
