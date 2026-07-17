# Agent Routing — Learning When to Commit

This repository trains a Qwen3.5-9B manager to decide whether to commit to its
current answer or consult one of three frozen sub-agents:

```text
delegate(extractor) | delegate(reasoner) | delegate(verifier) | COMMIT
```

The main experiment is strictly **in-domain**. MedQA, LegalBench large5,
MMLU-Pro, and GPQA each train their own sub-agent LoRAs, cold-start manager, and
GRPO manager. No benchmark reuses a manager trained on another benchmark.

The canonical, copy-paste execution guide is
**[IN_DOMAIN_EXPERIMENTS.md](IN_DOMAIN_EXPERIMENTS.md)**. It specifies frozen
splits, exact question budgets, every command for all four benchmarks,
synthetic-data audits, GPU layout, baselines, ablations, and reporting rules.
`EXPERIMENTS.md` is an archived transfer-oriented plan and must not define new
main-paper runs.

## Current evaluation design

| Benchmark | In-domain training pool | Dev | Held-out pool | Final evaluation |
|---|---:|---:|---:|---:|
| MedQA | 3,000 | 300 | 500 | 300 |
| LegalBench large5 | about 2,134 | 200 balanced | 500 balanced | 300 balanced, 60/task |
| MMLU-Pro custom partition | 4,000 | 300 | 500 | 300 |
| GPQA Extended minus all Diamond | 318 + 30 dev | 30 | all Diamond 198 | all 198 |

MMLU-Pro results must be labeled as a custom in-domain split, not standard
leaderboard performance. GPQA-Diamond is never used for sub-agent synthesis,
cold-start, GRPO, prompt selection, or reward tuning.

## Architecture and invariants

```text
                         Manager (Qwen3.5-9B)
                   draft -> delegate or commit
                      /          |          \
             Extractor       Reasoner       Verifier
            factual signal   neutral plan   audits current draft
```

- Sub-agents emit structured signals, never final answers.
- Synthetic data is schema-checked, choice-leakage audited, deduplicated, and
  selected on one shared question set across all three sub-agent roles.
- Verifier candidates come from actual GT-blind base-manager predictions.
- Cold-start defaults to `base_stepwise`: the base manager is queried before
  tool use and after every sub-agent result, including the final result.
- Ground truth may supervise the terminal answer and select a cost-aware
  oracle action, but is never copied into a normal `DRAFT_ANSWER`.
- Sub-agent SFT, cold-start, GRPO, dev, and held-out question hashes are disjoint.
- GPQA training is Extended minus every Diamond question, regardless of how
  many Diamond questions are reported.

## Thinking and compute controls

The main ADC Manager is explicitly non-thinking, emits no free-form reasoning,
and uses a 1,024-token GRPO trajectory cap. Evaluation permits 256 generated
tokens per decision and 1,024 across at most four Manager turns. The Manager
routes, stops, and emits an answer while the sub-agents do the long-form
reasoning. The paper comparison is framework-internal: learned routing versus
forced `none`, individual roles, pairs, and all three roles. MAS evaluation
reports Manager and per-sub-agent prompt/completion/total tokens, their
system-wide sums, cache hits, cap hits, and usage conditioned on tool-call count.
The role caps default to Extractor 512, Reasoner 1,024, and Verifier 768 and can
be overridden with `SUBAGENT_<ROLE>_MAX_NEW_TOKENS` environment variables.

## Reward

The main manager uses the bounded anytime ADC objective:

```text
R = final_bonus * 1[final answer correct]
  + draft_bonus * mean(correctness of answer statements)
  - missing_draft_penalty * missing draft count
  - cost_per_tool * number of delegations
```

`transition` and `sum` remain ablation-only reward variants because they can
encourage sandbagging or unnecessary calls.

## Installation and GPU layout

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt accelerate deepspeed requests

conda create -n vllm_env python=3.11 -y
conda activate vllm_env
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

Qwen3.5 is loaded through `Qwen3_5ForCausalLM`, so text-only SFT/GRPO does not
retain the vision encoder. Its LoRA targets include both standard attention
projections and the Gated DeltaNet projections (`in_proj_qkv`, `in_proj_z`,
`in_proj_b`, `in_proj_a`, `out_proj`). Replacing only the model ID in an older
Qwen3 training environment is not supported.

For Qwen3.5-9B:

```text
GPU 0    vLLM base model + extractor/reasoner/verifier LoRAs
GPU 1-3  manager GRPO with Accelerate/DeepSpeed
GPU 1    final manager evaluation, calling sub-agents remotely on GPU 0
```

Start the shared base/sub-agent server after training a benchmark's three
adapters:

```bash
conda activate vllm_env
bash scripts/start_subagent_server.sh Qwen/Qwen3.5-9B <RUN_ID> outputs 8000 32768
curl -f http://localhost:8000/health
```

The requested main GRPO profile keeps eight rollouts per question:

```bash
--mgr_bs 8 --mgr_num_generations 8 --mgr_generation_batch_size 8 \
--mgr_max_completion_length 1024 \
--mgr_temperature 1.0 --mgr_top_p 0.95 --mgr_top_k 20 --mgr_min_p 0.0 \
--mgr_learning_rate 1e-6 --mgr_grpo_beta 0.001
```

With three training GPUs this is a global completion batch of 24 and three
questions per update; completions are generated eight at a time to reduce peak
memory without changing the rollout group. It is aligned with recent
agentic/MAS RL practice:
[AgentFlow](https://arxiv.org/abs/2510.05592) uses global batch 32 with eight
rollouts per sample and a 2,048-token planner cap; [Dr. MAS](https://arxiv.org/abs/2602.08847)
uses group size 8, batch 32, and a 4,096-token response cap for its Qwen3-4B/8B
math setting. Those agents generate long mathematical reasoning, so their cap is
not a reason to make this routing-only Manager verbose. Always run a short 9B
memory smoke test before the full job.

## Lightweight validation

These checks do not download models or datasets:

```bash
python -m py_compile \
  src/manager/evolve.py src/pipeline/cli.py src/pipeline/stages.py \
  src/subagents/synthesize.py scripts/*.py tests/*.py
python -m unittest discover -s tests -v
bash -n scripts/start_subagent_server.sh scripts/train_manager_grpo_multigpu.sh
```

Full GPU integration and benchmark runs require the model weights, accepted
dataset licenses, and the cluster environments described in
`IN_DOMAIN_EXPERIMENTS.md`.
