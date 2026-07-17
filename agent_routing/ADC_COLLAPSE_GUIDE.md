# ADC 工具坍塌:成因分析、修复方案与后续实验监控指南

> 适用版本:2026-07-16 的 reward/training 修复(`reward.py` / `prompt.py` / `grpo_train.py` / `stages.py` / `cli.py`,测试见 `tests/test_adc_reward.py`)。
> 背景现象:**binary reward 下 manager 正常学会 tool call;开启 `--mgr_adc_mode` 后 tool rate 坍塌到 ~0**,尽管 trace 数据显示 corrections > corruptions(工具的真实边际价值为正)。

---

## 1. 坍塌为什么会发生

坍塌不是单一 bug,是**四个反工具压力叠加**,并且其中最主要的一个会随训练进行自我强化。按贡献大小排序:

### 1.1 主因:`scale_rewards="group"` 把微小的 cost 差放大成满幅负梯度

GRPO 的组内 advantage 归一化:`A_i = (r_i − mean) / std`。

两种 reward 模式在"全对组"(同一道题的 8 条 rollout 全部答对)里的行为有**质的差别**:

| 模式 | 全对组内 reward | std | advantage |
|---|---|---|---|
| binary | 全部 1.0,完全相等 | 0 | **0**(组不产生梯度) |
| ADC(旧) | 1.2 / 1.15 / 1.1…(随 k 变化) | ~0.02 | 调工具的 ≈ **−1.6** |

也就是说:旧 ADC 配置下,**每一道模型已经会做的题,都在用与"答错"同量级的梯度教它"别调工具"**。而且训练越往后、准确率越高,全对组占比越大,这个反工具梯度越强——一个随模型变强而收紧的棘轮,终点就是 k→0。

名义上 cost_per_tool=0.05 的节俭压力,经过 std 归一化的"汇率"后实际是 30 倍以上。这就是"明明 corrections > corruptions、工具有用,还是塌了"的核心原因:工具的收益要越过的有效门槛不是 5pp,而是 50pp+。

### 1.2 奖励面本身系统性偏向 k=0

旧默认参数(final=1.0, draft=0.2, cost=0.05)下的奖励面:

| 轨迹 | reward |
|---|---|
| 直答,对 | **1.2**(final 计入 entries,k=0 白拿满额 draft bonus) |
| k 个工具,全对 | 1.2 − 0.05k(上限严格低于直答) |
| 1 工具,首 draft 错 → final 对(教科书式纠错) | **1.05**(被 anytime 平均稀释) |
| 直答,错 | 0 |

两个结构性问题:
- **k=0 补贴**:`entries` 包含 final,直答答对时 avg 自动 = 1,免费拿满 0.2。注意这个设计不能简单删掉——把 final 从 entries 剔除会让"调一个平凡工具刷正确 draft"变成正收益("sum" 漏洞复活)。
- **稀释税**:"首 draft 错、被工具纠正"正是 ADC 想教的行为,但这类轨迹的 avg 必然 < 1,反而比"全程都对"多扣分。**任何对 draft 内容付费的项,必然在(a)奖励多余调用、(b)惩罚纠错、(c)补贴 k=0 三者中至少占一样**(不可能三角)。

### 1.3 截断/回滚轨迹吃 −1.2 格式锤,且最深奖励谷只在工具侧

TRL 会把超出 completion 预算的 tool result **回滚**(见 `grpo_train.py` 中的注释),轨迹以 dangling tool call 收尾 → 无 final answer + 最后消息带 tool_calls → 双重格式违规 → 旧代码砸 `−(final_bonus + draft_bonus) = −1.2`,叠加 cost 后总分可到 **−1.65**。

关键不对称:**直答轨迹几乎不可能被截断**(只有几十个 token)。整个奖励空间的最低谷只有调工具才够得着;每次预算事件都是一记大额反工具梯度,再经过组内 std 放大,一个 −1.5 的离群值能把同组所有工具使用者一起拖下水。

### 1.4 格式锤定点打击与 deliberation 绑定的行为

- 旧防刷屏正则 `ANSWER_[A-Za-z0-9]` 会把 final turn 里的 `DRAFT_ANSWER_B` 也计入(子串匹配)。SFT 学出"先 draft 再 answer"习惯的模型,RL 初期在 final turn 惯性带一行 draft 就吃 −1.2 —— 被重罚的恰好是和工具使用绑定的行为模式。
- `extract_answer_sequence` 旧实现把中间轮的裸 `ANSWER_` 也计入 entry/draft:既能绕过 missing_draft_penalty(不写 DRAFT 格式也不罚),又能在看完 tool result 后"复读"正确答案稀释早期错误 draft(单次约 +0.017,但在 std 放大下足以被学出来)。

---

## 2. 修复方案(已全部实现)

| # | 问题 | 修复 | 位置 / flag | 新默认 |
|---|---|---|---|---|
| 1 | 组内 std 放大 | advantage 不再除以组内 std(Dr. GRPO 风格),cost 恢复名义汇率 | `--mgr_scale_rewards` | `none`(可选 `batch`/`group`) |
| 2a | 截断轨迹进 loss | `mask_truncated_completions=True`(带 TRL 签名检查) | `grpo_train.py` GRPOConfig | 自动开启 |
| 2b | 截断轨迹吃格式锤 | 识别 dangling tool call(最后消息带 tool_calls 且无 final)→ 豁免格式罚,trace 记 `truncated` | `reward.py` | 自动 |
| 3 | draft 内容奖励的三角扭曲 | draft_bonus 降一个数量级,draft 习惯改由 missing_draft_penalty(纯格式约束)维持 | `--mgr_adc_draft_bonus` | 0.2 → **0.02** |
| 4a | cost 高于真实边际价值 | 默认下调,并要求按 §4 校准 | `--mgr_adc_cost_per_tool` | 0.05 → **0.02** |
| 4b | 技能未成形先被收费 | cost 从 0 线性退火到目标值 | `--mgr_adc_cost_warmup_steps` | **100**(0 = 关闭) |
| 5a | 格式锤随 k 加深 | 违规时 `reward = min(r, 0) − format_penalty`,谷底不再叠加 | `--mgr_adc_format_penalty` | **0.2** |
| 5b | 正则误伤 final turn 的 draft 行 | `(?<!DRAFT_)ANSWER_[A-Za-z0-9]` | `reward.py` | 自动 |
| 5c | 裸 ANSWER 伪造 draft / 复读稀释 | 中间轮只认 `DRAFT_ANSWER_`,final 只从最后一轮取 | `prompt.py` | 自动 |

### 修复后的 reward 公式

```
R = 1.0 · 1[final 正确]
  + 0.02 · (全部 answer 声明中正确的比例)        # 近似只是 tie-breaker
  − 0.1  · max(0, 工具数 − draft 数)             # 纯格式约束
  − cost(t) · 工具数                              # cost(t) = 0.02 · min(1, step/100)
格式违规(策略选择的,非截断): R = min(R, 0) − 0.2
截断(dangling tool call):     不施加格式罚,mask 出 loss
```

**决策边界**:调第 k 个工具 ⟺ 它带来的 ΔP(答对) > cost/final_bonus = **2pp**。这条边界现在真的以名义值生效(不再被 std 放大),它本身就是"需要时用、不需要时不用"的定义。

### 为什么 draft 降到 0.02 不会丢掉 draft 质量

draft 的真实价值是**环境介导**的:`verifier_tool` 消费 `current_draft`,draft 诚实 → verifier 审得准 → final 更容易对 → final bonus 的梯度已经传回 draft 质量。这条通道天然免疫 sandbagging 和 farming,不需要显式内容奖励。显式项保留 0.02 只是为了论文里 anytime/transition/sum 三个 variant 的对照结构完整。

---

## 3. 修复后各典型轨迹的 reward(warmup 结束后,cost=0.02)

| 轨迹 | 旧 reward | 新 reward |
|---|---|---|
| 直答,对 | 1.2 | 1.02 |
| k=1 全对 | 1.15 | 1.00 |
| k=1 首 draft 错→final 对 | 1.05 | 1.01(稀释税从 0.15 降到 0.01) |
| k=3 全对 | 1.05 | 0.96 |
| 直答,错 | 0 | 0 |
| k=2 截断(dangling call) | ≈ −1.29 | ≈ −0.02,且 mask 出 loss |
| final turn 刷屏(格式违规) | −1.2 起步、随 k 加深 | −0.2 固定 |
| final turn 带 draft 行 + answer | −1.2(误伤) | 正常计分 |

新旧对比的要点:**"对 vs 错"的信号(≈1.0)相对"工具多 vs 少"的信号(0.02/个)恢复了 50:1 的设计比例**,且这个比例不再被归一化扭曲。

---

## 4. cost_per_tool 的数据驱动校准(每换一个数据集/底模都要做)

不要拍脑袋。对每条调用了工具的轨迹,`1[final 正确] − 1[首 draft 正确]` 就是该轨迹实现了的 ΔP,汇总即 (corrections − corruptions)。校准规则:

```
边际价值 v = (Σ corrections − Σ corruptions) / Σ tool_calls × final_bonus
cost_per_tool ∈ [v/3, v/2]
```

从 `train_raw_trace.jsonl`(或 eval trace)直接算:

```python
import json

recs = [json.loads(l) for l in open("outputs/manager/<run>/train_raw_trace.jsonl", encoding="utf-8")]
adc  = [r for r in recs if r.get("reward_mode", "").startswith("adc") and r.get("tool_calls", 0) > 0]
corr = sum(r.get("corrections", 0) for r in adc)
corrup = sum(r.get("corruptions", 0) for r in adc)
calls = sum(r["tool_calls"] for r in adc)
v = (corr - corrup) / max(calls, 1)
print(f"corrections={corr} corruptions={corrup} calls={calls}")
print(f"边际价值 v={v:.4f}  建议 cost_per_tool ∈ [{v/3:.3f}, {v/2:.3f}]")
```

- v ≤ 0:先别开 cost(`--mgr_adc_cost_per_tool 0`),问题在 subagent 质量,不在路由。
- cost > v:即使没有任何放大 bug,坍塌到 k=0 也是这个 reward 下的"理性解"。

---

## 5. 后续实验监控指南

### 5.1 必看的五个指标(全部可从 `train_raw_trace.jsonl` 计算)

| 指标 | 怎么算 | 健康区间 |
|---|---|---|
| **tool rate(总体)** | mean(tool_calls > 0),按 step 窗口滑动 | warmup 结束(默认第 100 步)后不归零、不 100% |
| **tool rate(按难度分桶)** ★ | 难度代理 = 该题组内正确率;分 易(>0.8)/中/难(<0.4) 三桶 | 易题桶缓慢下降、难题桶持平或上升,**两条曲线张开** |
| **净纠错率** | (corrections − corruptions) / 窗口内轨迹数 | 持续 > 0 且不趋零(趋零 = 工具技能在萎缩) |
| **truncated 比例** | mean(truncated),trace 新增字段 | < 5%;偏高说明 max_completion_length 不够 |
| **reward 分量** | draft_reward / tool_cost / missing_drafts 分别的均值 | missing_drafts 均值应快速降到 ~0(draft 格式已学会) |

★ 是唯一能区分"健康自适应"和"病理坍塌"的指标:**整体 tool rate 下降本身不是坏消息**,要看它降在哪个桶。

### 5.2 症状 → 诊断 → 调整对照表

| 症状 | 可能原因 | 调整动作(按顺序试) |
|---|---|---|
| warmup 后 tool rate 仍快速滑向 0,难题桶也在掉 | cost 仍高于真实边际价值 | ① 用 §4 重新校准 cost;② 延长 warmup(200-300);③ 确认没显式传 `--mgr_scale_rewards group` |
| tool rate 稳定但准确率低于 binary baseline | 探索不足,多工具技能没长出来 | ① `--mgr_clip_epsilon_high 0.28`(DAPO);② 加 `--mgr_exploration_hint`;③ warmup 期间拉长 |
| 易题桶 tool rate 迟迟不降(节俭学得太慢) | `none` 下节俭梯度只有名义值,信号弱 | ① 先等——这是良性失败,只费推理不掉分;② `--mgr_scale_rewards batch`(中间档);③ 训练后期把 cost 升到 0.03-0.04(仍须 < v/2) |
| truncated 比例 > 5-10% | completion 预算不够 3 个 tool reply | 提高 `--mgr_max_completion_length`(脚本建议 4096),或压 SUBAGENT_*_MAX_NEW_TOKENS |
| corrections ≈ corruptions ≈ 0 | 模型不写 draft,或 verifier 没拿到 current_draft | 查 missing_drafts 均值;查 verifier 调用参数;draft 格式罚(0.1)是否被显式关掉 |
| reward 均值高但 pred 解析失败率升高 | 新的格式漂移(clamp 后违规成本变低) | 观察 fail_buffer 里 valid_format=False 的样本;必要时 `--mgr_adc_format_penalty 0.3-0.5` |

### 5.3 判定"坍塌复发"的硬标准

同时满足以下两条即为复发,停下来重新校准而不是继续加 step:

1. 难题桶(组内正确率 < 0.4)的 tool rate 连续 ~50 步单调下降且 < 10%;
2. 同窗口内整体准确率没有相应上升。

只满足 1 不满足 2 的情况理论上不该出现(难题不用工具、准确率还涨,说明难度代理失效,换代理重算)。

---

## 6. 已知残余权衡(设计上接受的,不是遗漏)

1. **k=0 补贴仍在,但只剩 0.02**。final 计入 entries 是防 "sum" farming 的关键(§1.2),不能删;draft_bonus 降到 0.02 后补贴与 cost 同量级,可忽略。
2. **warmup 期间(前 100 步)没有节俭压力**,模型可能短暂过度调用——这是有意的:先长技能,后收费。
3. **`none` 下节俭学习比旧配置慢得多**。这是坍塌修复的直接代价,失败方向从"毁掉能力"变成"多花推理钱",可用 §5.2 第三行的手段渐进加压。
4. **binary 模式也受 `scale_rewards` 默认值变化影响**(混合组的 advantage 尺度变了;全对组两种设置都是零梯度)。要和历史 binary run 严格可比,给旧 arm 显式传 `--mgr_scale_rewards group`。
5. **EXPERIMENTS.md 里带显式旧值的命令**(`--mgr_adc_cost_per_tool 0.05 --mgr_adc_draft_bonus 0.2`)会覆盖新默认,复现旧(坍塌)配置——做消融时这正好是对照 arm,做主实验时记得删掉这些 flag。

## 7. 消融建议(如果要把坍塌机制写进论文)

坍塌本身是一个干净的发现:**process reward + 组内 std 归一化 ⇒ 稀疏成本项被放大 ⇒ 工具使用坍塌**。四个 arm 即可支撑:

| arm | flag | 预期 |
|---|---|---|
| 修复全开(主) | 默认值 | tool rate 存活,难度分桶张开 |
| 只回滚 scaling | `--mgr_scale_rewards group` | 复现坍塌 → 证明主因 |
| 只关截断保护 | 需临时改码(mask + 豁免) | tool rate 受截断率调制的衰减 |
| 旧参数全回滚 | 显式传旧 flag + group | 完整复现原始坍塌曲线 |

回归测试:`python -m pytest tests/test_adc_reward.py -v`(9 项,覆盖全部修复行为与 anytime/transition/sum 的激励相容性质)。
