# 设计 A:CGC — Counterfactual Group Composition(反事实组构成)

> 实现日期:2026-07-17。相关文档:[ADC_COLLAPSE_GUIDE.md](ADC_COLLAPSE_GUIDE.md)(第一轮修复)、[ADC_RESIDUAL_HOLES.md](ADC_RESIDUAL_HOLES.md)(残余漏洞)。
> 一句话:**harness 在每个 GRPO 组内做随机对照试验——一半 rollout 工具被禁用——让组均值本身携带无工具反事实基线;reward 退回 binary + 微小 cost。路由信号来自组的构成,不来自 reward 的形状。**

启动(在原 GRPO 命令上替换 reward 相关 flag):

```bash
--mgr_cgc_mode        # 其余全用默认值即可;不要再传 --mgr_adc_mode
```

---

## 1. 为什么采用设计 A:完整证据链

这个设计不是拍脑袋,是被四轮实验逐步逼出来的。按时间线:

| # | 实验/事件 | 结论 |
|---|---|---|
| 1 | MedQA binary:acc +10,tool_rate 健康 | binary 是能工作的地基 |
| 2 | GPQA ADC(原版):坍塌 | 发现 std 放大、截断锤、k=0 补贴等 reward 层漏洞 |
| 3 | 修复全部 reward 层漏洞(scale_rewards=none、截断豁免、clamp、退火)后重跑 GPQA ADC | **前 10 步仍然坍塌到 0**(此时 warmup 使 cost≈0)——坍塌与 cost 无关,reward 层修复已到极限 |
| 4 | GPQA binary:tool_rate 1.0,acc 0.29(无工具时 0.25) | binary 不塌但过度调用;且测得工具在 GPQA 的总边际价值仅 ≈ +4pp |

从 3、4 推出两个第一性原理层面的诊断:

**诊断一(统计功效):** 想学的量是反事实边际收益 ΔP = P(对|调用) − P(对|不调用)。从 Bernoulli 奖励里分辨 Δ 需要 ~Var/Δ² 个配对样本:MedQA(Δ≈0.10)约 20 个,GRPO 喂得起;GPQA(Δ≈0.04)约 125 个,喂不起。**"能否学会路由"由 Δ²/Var 决定,reward 设计只能改常数,改不了标度。**

**诊断二(梯度结构,LLD):** 所有工具轨迹共享几乎相同的 tool-call 语法 token(低熵瓶颈);在大多数 rollout 都答错的数据上,微负 advantage 反复叠加在这一撮共享 token 上,而直答轨迹没有对应瓶颈。概率一旦被压低,探索消失,单向棘轮几步内完成——这解释了"10 步、暴力、与 reward 参数无关"。(文献印证:arXiv 2512.04220 "Lazy Likelihood Displacement";2605.26037 "peak-then-collapse"。)

**由此得出设计原则:** 每一个塞进 reward 的信息项都同时成为可被梯度流经、可被策略博弈、可被病理放大的激励项(ADC 两版的全部失败都是这个模式)。而 harness 决定的**组构成是外生的**——不在策略动作空间里,策略无法对它博弈,梯度不流经它。所以:**激励走 reward(保持 binary 的简单与抗坍塌),信息走结构(harness 构造反事实)。**

## 2. 机制:三件事各自解决一个诊断

```
每组 8 条 rollout:
  4 条 on-arm  —— 工具正常执行
  4 条 off-arm —— 环境把一切工具调用回以 tools_unavailable 哨兵,模型被迫直答

reward(cgc 模式):
  R = 1[答对] − cost(t)·k_eff − 0.05·(缺 draft 的工具轮数)     [on-arm]
  R = 1[答对]                                                  [off-arm,全豁免]
  cost(t) = 0.01 · min(1, step/100)    格式违规: min(R,0) − 0.2    截断豁免同 ADC

组置平(novar):组内正确性无方差(全对/全错)→ 全组 reward 置为组均值 → advantage 全 0
```

- **配对 arm → 解决诊断一**:组均值 ≈ (p̂₀+p̂₁)/2,工具轨迹的期望 advantage ∝ ΔP/2。反事实每组必在场,配对差分替代自然混合,方差大幅下降;
- **组置平 → 解决诊断二的燃料**:全错/全对组里唯一的组内差异是 cost/罚(全在工具侧),正是恒定反工具滴灌;置平后这些组闭嘴(DAPO dynamic sampling 的软实现,不动 TRL);混合组内 cost 仍作为节俭 tiebreaker 存活;
- **off-arm 本身 → 削弱 LLD**:每组只有一半 rollout 含 tool-call token,共享瓶颈上的负梯度质量直接减半,且直答能力被 off-arm 持续训练(防止这次 GPQA 那种 acc 掉到 0.25 的连带损伤)。

**draft 彻底退出 reward**(bonus=0),只保留 on-arm 的逐轮存在性约束(0.05/缺失轮,同时落实了 RESIDUAL_HOLES §2 的逐轮配对修法——post-hoc 补 draft 不再能免罚,测试锁定)。draft 变成纯外生遥测通道:corrections/corruptions、首 draft 正确率照常记录,且新增一个免费检验——**on-arm 首 draft 准确率 ≈ off-arm 直答准确率**(两者都估计 p₀),吻合即证明 draft 诚实。

## 3. 实现清单

| 组件 | 位置 | 内容 |
|---|---|---|
| arm 分配 | `grpo_train.py` `_cgc_configure`/`_cgc_sample_off_arm` + `ManagerToolEnvironment.reset` | 每次 reset 采样;fraction=0.5 时确定性交替(组连续创建时恰好 4/4),否则种子 RNG |
| 工具拦截 | `ManagerToolEnvironment` 三个工具方法 | off-arm 一律返回 `TOOLS_UNAVAILABLE_MSG` 哨兵(JSON,内含"立即直答"的指令) |
| arm 识别 | `reward.py` `_count_blocked_tool_msgs` | reward 侧凭哨兵识别,无需位置假设;`k_eff` = 实际执行的工具数(排除被拦截的) |
| CGC reward | `reward.py` cgc 分支 | 见 §2 公式;模式优先级 cgc > adc > ccr > binary |
| 组置平 | `reward.py` reward_fn 末尾 | 按 example_id 分组(对 batch 排列鲁棒),novar 判据 |
| 逐轮配对 | `prompt.py` `count_unpaired_tool_turns` | 带 tool_calls 的 assistant 轮缺 DRAFT 行才计罚 |
| 遥测 | trace 新字段 | `k_eff`、`blocked_tool_calls`、`off_arm`、`used_tools`、`n_unpaired_tool_turns`、`first_draft`、`first_draft_correct`、`corrections`、`corruptions`、`truncated`、`flattened`、`cost_per_tool_effective` |
| flags | `--mgr_cgc_mode/off_arm_fraction/cost_per_tool/missing_draft_penalty/cost_warmup_steps/flatten` | 默认:0.5 / 0.01 / 0.05 / 100 / novar |
| 守卫 | `train_manager_grpo` | cgc 要求 binding_mode=environment(argument 模式无 per-rollout 拦截点,直接 raise);与 adc 同开时警告并优先 cgc |
| 测试 | `tests/test_cgc_reward.py` | 11 项,全部通过(总套件 40/40) |

## 4. 可证伪的预测与监控

| 数据集 | 预测 | 证伪含义 |
|---|---|---|
| MedQA | acc 保持 ≥ +10;tool_rate 从"高"收缩到集中于难题 | 若 acc 掉:off-arm 比例伤害了工具技能训练,调低 off_arm_fraction |
| GPQA | tool_rate 稳定在低位但**不归零**;acc ≥ 0.29(不再出现 0.25 的连带损伤) | 若仍 10 步归零:证伪"组构成"假说,矛头指向 LLD 的 token 层,下一步是 tool-call 语法 token 负梯度 mask(改 loss,不改 harness) |

监控(全部来自 trace 新字段):
1. **在线 ΔP̂**:按窗口统计 mean(correct | used_tools) − mean(correct | ~used_tools)——设计 A 的核心读数,它同时就是论文的"边际收益测量";
2. **flattened 率**:被置平的组占比。GPQA 上预期高(大量无信号组),MedQA 上应适中;→1 说明数据集难度失配;
3. **draft 诚实性**:on-arm first_draft_correct 均值 vs off-arm correct 均值,差 >5pp 报警;
4. **n_unpaired_tool_turns**:应快速 →0(draft 习惯);
5. 老三样:分难度桶 tool_rate、corrections−corruptions、truncated 率。

## 5. 定位与后续

- **ADC 不删**:anytime/transition/sum 保留为论文的激励相容性消融 arm;CGC 是主线训练法。
- **GPQA 的战场定位**:工具总边际价值 ≈ +4pp,理性路由器学到的就是"少用"。主实验讲"学会用"应放 MedQA;GPQA 讲"框架正确地学会不滥用"(对照 binary 的 tool_rate 1.0 过度调用/cognitive offloading)。跑正式实验前建议先用 k-sweep 测 oracle routing headroom。
- **通往设计 B/C**:CGC 的配对遥测(p̂₀, p̂₁)就是设计 B(校准阈值路由、λ 扫 Pareto 曲线)的监督数据;draft 序列就是设计 C(外生 budget 的 anytime 训练)的 anytime 输出合同。设计 A 不只是修坍塌——它是整个"测量边际收益 → 用边际收益路由"框架的数据采集层。

## 6. 论文表述(供直接取用)

贡献命名:**Counterfactual Group Composition** —— 在 group-relative RL 中,由 agent harness 对每个组实施随机工具可用性干预,使组相对基线成为无工具反事实基线,advantage 由此成为工具边际价值(ATE/2)的无混淆估计。核心二分:endogenous shaping(reward 内的信息项,可被策略博弈、被归一化放大,实验证明坍塌)vs exogenous composition(harness 决定的组结构,策略不可触及,梯度不流经)。不改 GRPO 数学、不改 loss、不改 reward 语义——学习信号的正确注入点不在目标函数里,在实验设计里。
