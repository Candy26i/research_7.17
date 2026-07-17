# ADC 残余漏洞分析:双通道 draft、draft 时机、两个未验证实现假设

> 姊妹文档:[ADC_COLLAPSE_GUIDE.md](ADC_COLLAPSE_GUIDE.md)(坍塌成因与已完成的修复)。
> 本文覆盖修复之后**仍然存在**的三个漏洞:每个漏洞是什么、数学上为什么威胁的不是训练而是测量、怎么改代码、怎么监控、以及"什么信号出现才值得打补丁"的决策规则。
>
> 总体定位:当前 reward 的所有可证明作弊收益已被 draft_bonus = 0.02 封顶(见 GUIDE §2),训练坍塌级别的漏洞已关闭。本文三个漏洞的共同特征是:**GRPO 优化的目标函数和论文想测量的量之间存在缝隙**——上界分析保证前者干净,保证不了后者。

---

## 漏洞 1:draft 文本行与 verifier 参数是两条独立通道,诚实性只有 0.02 在管

### 1.1 问题是什么

manager 的"当前想法"有两个出口,走的是完全不同的路径:

| 通道 | 谁写 | 谁读 | 诚实激励来源 | 激励强度 |
|---|---|---|---|---|
| `DRAFT_ANSWER_X` 文本行 | assistant 轮的 content | **reward**(`extract_answer_sequence` → anytime 平均、corrections/corruptions 统计) | draft_bonus | **0.02**(几乎为零) |
| `current_draft="X"` 参数 | verifier_tool 的 tool call arguments | **verifier**(决定它审哪个假设) | verifier 审得准 → final 对 → final_bonus | **≈1.0**(环境介导,强) |

两条通道之间没有任何一致性约束。数学上完全允许这样的策略存在:**参数通道传真实想法(因为这直接影响能不能答对),文本行通道写敷衍常数(因为只有 0.02 的差别)**。GRPO 是随机爬山,只要这个策略不比诚实策略差超过噪声水平,它就可能被漂移到。

**为什么这不威胁训练**:文本行敷衍的组内 reward 损失 ≤ draft_bonus·(avg 差) ≤ 0.02,方向仍是"诚实略优"(anytime 平均下故意写错严格降分),不存在正向作弊收益,不会引发任何行为坍塌。

**为什么这严重威胁论文**:你的三个核心测量全部只读文本行通道——

- `corrections` / `corruptions`(W→C / C→W 转移)→ 支撑"deliberation 有净收益"的主张;
- `y_hat_seq` → anytime accuracy 曲线;
- `compute_deliberation_stats` 的全部输出。

如果文本行漂移成敷衍输出(比如恒写 `DRAFT_ANSWER_A`,或恒抄最终答案),这些统计不会报错、不会变 NaN,只会**安静地变成噪声**:corrections 归零或虚高,anytime 曲线失去含义,而你从 wandb 上看训练一切正常。这是最危险的失败模式——不 crash 的测量失效。

### 1.2 怎么修改

三个方案,按侵入性从低到高。**建议先只上 A(纯观测),B/C 等监控信号触发再上**(决策规则见 §1.4)。

**方案 A:一致性观测(零风险,建议立即做)**

reward 函数已经拿到完整 completion,能同时看到两条通道。在 `reward.py` 的 `_extract_completion_stats`(或旁边加一个小函数)里提取每个 verifier 调用轮的两样东西,写进 trace:

```python
def _draft_channel_consistency(completion, choice_keys):
    """对每个调 verifier_tool 的 assistant 轮,比较该轮的 DRAFT_ANSWER_ 行
    与 tool call 的 current_draft 参数。返回 (n_verifier_calls, n_consistent,
    n_missing_arg)。非 list completion 返回 (0, 0, 0)。"""
    n_calls = n_consistent = n_missing = 0
    if not isinstance(completion, list):
        return 0, 0, 0
    for msg in completion:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            fn = (tc.get("function", {}) or {})
            if fn.get("name") != "verifier_tool":
                continue
            n_calls += 1
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            arg_draft = str(args.get("current_draft", "")).strip().upper()
            line_draft = parse_draft_answer(_msg_text(msg.get("content")), choice_keys)
            if not arg_draft:
                n_missing += 1
            elif line_draft is not None and arg_draft == str(line_draft).upper():
                n_consistent += 1
    return n_calls, n_consistent, n_missing
```

trace 新增字段:`verifier_calls`、`draft_arg_consistent`、`draft_arg_missing`。

**方案 B:一致性软惩罚(监控触发后再上)**

在 ADC 分支加一项:`R -= consistency_penalty · (verifier_calls − draft_arg_consistent)`,建议 0.05(高于 0.02 的敷衍收益上界,低于 missing_draft_penalty)。注意:只罚 verifier 轮,extractor/reasoner 没有参数通道,罚不了也不需要罚。

风险:如果 SFT 模型本来一致率就低(格式还没学会),RL 初期这是又一个与工具绑定的惩罚——**上这个补丁前必须先看方案 A 的基线一致率**,低于 80% 就先修 SFT 数据而不是加罚。

**方案 C:结构性消灭双通道(最彻底,改接口)**

让环境代填 `current_draft`:manager 只写 `DRAFT_ANSWER_X` 文本行,`stages.py`/GRPO 环境在派发 verifier 调用时自动从该轮文本解析 draft 填入参数,tool schema 里删掉 `current_draft`。通道合一之后,文本行直接吃到环境介导的 ≈1.0 激励,漏洞从根上消失。

代价:改 `build_manager_tool_schemas`、SFT 数据里的 verifier 调用格式、eval 通路,三处要同步;历史 checkpoint 的行为不兼容。**适合下一个大版本做,不适合热修。**

### 1.3 怎么监控

| 指标 | 计算 | 健康线 | 报警含义 |
|---|---|---|---|
| **一致率** | Σdraft_arg_consistent / Σverifier_calls,按 step 窗口 | ≥ 0.95 且不随训练下降 | 下降 = 文本行或参数在漂移,corrections 统计开始失真 |
| **参数缺失率** | Σdraft_arg_missing / Σverifier_calls | ≈ 0 | 上升 = 模型学会省略参数(verifier 退化成盲审) |
| **draft 熵** | 文本行 draft 的选项分布熵,按窗口 | 接近答案分布的熵 | 熵骤降(恒写同一字母)= 敷衍模式的直接签名 |

其中 draft 熵是最灵敏的:敷衍常数策略在一致率还没掉的时候就会先把熵打下去。

### 1.4 决策规则

```
一致率 ≥95% 且 draft 熵正常     → 什么都不做,测量可信
一致率 85-95% 或 熵缓慢下降     → 上方案 B(0.05 软惩罚)
一致率 <85% 或 参数缺失率 >10%  → 检查 SFT 数据质量;论文里 corrections
                                   统计改用参数通道重算(需方案 A 的字段)
长期方案                         → 下个版本上 C
```

---

## 漏洞 2:draft 放置时机不受约束,"承诺"语义可被 post-hoc 满足

### 2.1 问题是什么

当前 `missing_drafts = max(0, k − #drafts)` 是**全局计数**:只要求 draft 总数 ≥ 工具总数,不要求 draft 出现在调工具的那一轮。于是存在一个合法策略:

```
轮1: [tool call, 无 draft]        ← 本该在这里承诺
轮2: [tool call, 无 draft]
轮3(见完所有工具结果): DRAFT_ANSWER_B
                        DRAFT_ANSWER_B   ← 每轮只计最后一个,需分两轮;
最终轮: ANSWER_B                            或最终轮带一行(现在正则已允许)
```

只要凑够数量,missing = 0,不吃罚。

**数学收益**:post-hoc draft 是见过工具结果的"知情承诺",正确率 q_late ≥ q_early(诚实早期 draft),anytime 平均项因此多赚 ≤ draft_bonus·(q_late − q_early)/2 ≤ **0.01**。训练上无关紧要。

**测量危害**:ADC 的整个叙事建立在"draft = 调工具**前**的诚实 best-guess,工具结果改变它 = correction"之上。post-hoc draft 恒等于最终答案 → W→C 转移消失 → **corrections 被系统性低估**,"工具没用"的假象;同时 draft accuracy 被虚高,anytime 曲线整体上移。和漏洞 1 一样:不 crash,只是安静地测歪。

另外它还与漏洞 1 复合:draft 推迟到工具轮之后,verifier 调用时文本行还不存在,一致性检查(方案 A)会把这种情况计为"该轮无 draft 行"——所以两个漏洞的监控要一起看才能分清是谁在作祟。

### 2.2 怎么修改

**逐轮配对惩罚**(把全局计数改成逐轮约束),这是语义上唯一正确的形式:

```python
# prompt.py 新增:
def count_unpaired_tool_turns(completion, choice_keys) -> Tuple[int, int]:
    """返回 (带 tool call 的 assistant 轮数, 其中缺 DRAFT_ANSWER_ 行的轮数)。"""
    n_tool_turns = n_unpaired = 0
    if not isinstance(completion, list):
        return 0, 0
    for msg in completion:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if not msg.get("tool_calls"):
            continue
        n_tool_turns += 1
        if parse_draft_answer(_msg_text_of(msg), choice_keys) is None:
            n_unpaired += 1
    return n_tool_turns, n_unpaired

# reward.py ADC 分支:
#   missing_drafts = n_unpaired   (替换 max(0, k − len(drafts)))
```

要点:

- **配对单位是"轮"不是"调用"**:一轮里调两个工具配一个 draft 是合理的(prompt 也是这么教的),按 tool call 数罚会误伤。
- **entries / anytime 平均的口径不变**,只改 missing 的算法——这样 "sum"/"transition" 消融 arm 的对照结构不受影响(它们的 miss_penalty 分支同步换新算法即可,语义一致)。
- **加 flag 保兼容**:`--mgr_adc_per_turn_pairing`(默认 True),需要复现旧行为的 ablation 时可关。
- 罚不罚 post-hoc draft 本身?不罚。final 轮带 draft 行在正则修复后是合法的,它进 entries 稀释/增益 ≤0.01,无所谓;逐轮配对已经保证工具轮**必须**有事前 draft,post-hoc 的只是额外条目,危害归零。
- 测试:`test_adc_reward.py` 加两条——(a) 工具轮无 draft、事后补两行 → 仍吃 0.1×2 罚;(b) 一轮两个 tool call + 一个 draft → 不罚。

这个补丁**建议直接上**,不用等监控触发:改动小、语义收紧方向唯一、不存在漏洞 1 那种"SFT 基线可能本来就差"的前置条件——SFT 数据里 draft 本来就是和 tool call 同轮的(prompt 明确要求),合规轨迹完全不受影响。

### 2.3 怎么监控

| 指标 | 计算 | 健康线 | 报警含义 |
|---|---|---|---|
| **配对率** | 1 − Σn_unpaired / Σn_tool_turns(trace 加这两个字段) | 打补丁后应 →1;打之前 ≥0.9 | 打之前就低 = SFT 阶段 draft 习惯没学牢 |
| **draft-final 相等率** | mean(所有 draft == final),从 y_hat_seq 算,现有字段就够 | 应随难度分层:易题高、难题低 | 全局 →1 = post-hoc 化的签名(承诺失去信息量) |
| **首 draft 时机分布** | y_hat_seq 首项所在轮 / 工具轮位置(需 trace 加 `first_draft_turn`) | 集中在第一个工具轮 | 后移 = 正在利用时机漏洞 |

draft-final 相等率是不需要任何新字段、现在就能从历史 trace 算的:如果你手头旧 run 的这个值已经异常高(>0.95 且不分难度),说明时机漏洞在旧配置下就已经被利用,corrections>corruptions 的旧结论也要重新审视。

---

## 漏洞 3:两个未验证的实现假设(loss 长度归一化;截断样本与组均值)

### 3.1 问题是什么

这两个不是 reward 公式的漏洞,是**我们对 TRL 行为的假设**,没在训练环境里验证过。假设错了,效果是"修复打了折扣",不是新的作弊面。

**3a. loss 侧仍是逐序列长度归一化(Dr. GRPO 只采用了一半)。**
`scale_rewards="none"` 修的是 advantage 的 std 除法(Dr. GRPO 修正 #1);但 TRL 默认的 `loss_type` 仍把每条序列的 token loss 除以自身长度(Dr. GRPO 修正 #2 针对的就是它)。后果:工具轨迹(数千 token,含 tool result)拿到正 advantage 时,**每个策略 token 分到的梯度被长度稀释**;直答轨迹(几十 token)的每 token 梯度浓。这是一个温和但恒定的"学直答快、学用工具慢"的不对称——不会造成坍塌(advantage 方向没变),但会拖慢难题桶 tool rate 的恢复速度。

**3b. 被 mask 的截断样本是否仍进组均值。**
`mask_truncated_completions=True` 把截断样本的 token 从 loss 里剔除,但其 reward 是否仍参与同组其他样本的 advantage 基线(组均值)计算,取决于 TRL 实现。若参与:一个 r≈−0.02 的截断样本会把全对组的均值从 1.01 拉到 ≈0.88,同组每个正常样本的 advantage 被抬高 ≈0.13——方向是良性的(等于给"没被截断"发奖金),但引入了一个和策略无关的噪声源:**组的 advantage 尺度随截断事件波动**。截断率 <5% 时可忽略;截断率高时它会放大 §3a 之外的另一种方差。

### 3.2 怎么修改 / 验证

**先验证,再决定改不改。**两条都是训练服务器上一分钟能确认的事:

```bash
# 3a: 确认 loss_type 的可选值与默认值
python -c "from trl import GRPOConfig; import inspect; \
  p = inspect.signature(GRPOConfig.__init__).parameters['loss_type']; \
  print(p.default)"

# 3b: 看 mask 与组均值的实现(搜 advantage 计算处)
python - <<'EOF'
import inspect, trl.trainer.grpo_trainer as g
src = inspect.getsource(g)
i = src.find("mask_truncated")
print(src[max(0,i-1500):i+1500])
EOF
```

**3a 的修改**:确认支持后,给 `grpo_train.py` 加 `loss_type` 直通参数(默认保持 TRL 默认,加 `--mgr_loss_type dr_grpo` 作为消融 arm)。**不建议默认打开**:它和 `scale_rewards` 是两个独立变量,一起默认改会让"哪个开关救了 tool rate"说不清;先做单开关消融(GUIDE §7 的 arm 表加一行),显著就转正。

**3b 的修改**:若确认截断 reward 进组均值且截断率压不下来(>5%),两个选项:
- 首选**治本**:提高 `--mgr_max_completion_length` / 压 SUBAGENT_*_MAX_NEW_TOKENS,把截断率打到 <2%,问题自然消失;
- 备选:reward 侧把截断样本的 reward 改记为 `None`(TRL 新版支持 reward 函数返回 None 表示"该样本不参与该 reward 统计")——**先验证安装版本的 None 语义再动**,否则可能变成 NaN 传播。

### 3.3 怎么监控

| 指标 | 计算 | 说明 |
|---|---|---|
| **截断率** | mean(truncated),trace 现有字段 | <2% 时 3b 整体可忽略;>5% 先修预算 |
| **组内 reward 极差 vs 截断共现** | 有截断样本的组 vs 无截断的组,各自的组内 advantage 方差 | 差异显著 = 3b 的噪声在起作用 |
| **每 token 梯度代理**(3a,粗) | wandb 上按轨迹长度分桶的 completion 占比变化速度:短轨迹行为收敛快于长轨迹很多 | 只是旁证;严格归因靠 loss_type 消融 arm |

---

## 汇总:优先级与行动清单

| 行动 | 针对 | 时机 | 改动量 |
|---|---|---|---|
| 逐轮配对 missing 罚(§2.2)+ 两条测试 | 漏洞 2 | **立即**,无前置条件 | reward.py + prompt.py,小 |
| 一致性/时机观测字段:`verifier_calls`、`draft_arg_consistent`、`draft_arg_missing`、`first_draft_turn`、`n_tool_turns`、`n_unpaired`(§1.2A, §2.3) | 漏洞 1+2 | **立即**,纯观测零风险 | reward.py trace,小 |
| 用旧 trace 先算 draft-final 相等率(§2.3) | 漏洞 2 追溯 | 立即,不用重训 | 一段脚本 |
| 服务器上验证 3a/3b 两个假设(§3.2) | 漏洞 3 | 下次登训练机时 | 两条命令 |
| 一致性软惩罚 0.05(§1.2B) | 漏洞 1 | 仅当一致率落到 85-95% | reward.py,小 |
| `--mgr_loss_type dr_grpo` 消融 arm | 漏洞 3a | 主实验稳定后 | grpo_train.py 直通,小 |
| 通道合一(§1.2C) | 漏洞 1 根治 | 下个大版本 | schema+SFT+eval,大 |

三个漏洞的共同教训,值得写进论文的 discussion:**process reward 的激励相容性证明只覆盖"reward 读得到的通道"**;当测量通道(文本行)和行为通道(工具参数)分离、或约束是全局的而语义是逐步的,策略可以在不损失 reward 的情况下让测量失真。监控的作用不是防坍塌,是守住测量的有效性。
