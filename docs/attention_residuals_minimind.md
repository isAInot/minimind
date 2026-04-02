# MiniMind Attention Residuals 改造说明

## 1. 这次改了什么

这次把 Kimi `Attention Residuals` 的核心思路接进了 MiniMind，并且做成了可切换模式，而不是直接替换原始实现。

当前支持三种残差模式：

- `standard`: 原始 MiniMind 残差，不改行为，用来做 baseline
- `attnres_full`: 完整版 Attention Residuals，沿深度维度对历史残差分支做 softmax 聚合
- `attnres_block`: 工程缩放版，块内保留标准残差，只在 block 之间做 Attention Residuals

这套实现同时覆盖：

- dense FFN
- MoE FFN

MoE 的 router、top-k expert、aux loss 都保持原逻辑不变，只改 residual merge 方式。

---

## 2. 论文核心思想如何映射到 MiniMind

标准 pre-norm Transformer 的残差写法可以看成：

```text
h_{l+1} = h_l + f_l(h_l)
```

它的本质是“把所有历史分支都等权累加”。

Attention Residuals 的思路是：

1. 不再默认所有历史分支权重都相同
2. 每个残差融合点都用一个可学习的 query
3. 对历史 residual branches 做沿深度维度的注意力聚合

本项目里采用的近似公式是：

```text
V = [v_0, v_1, ..., v_n]
K = RMSNorm(V)
alpha = softmax(K · q)
h = sum_i alpha_i * v_i
```

其中：

- `v_i` 是历史 residual branches
- `q` 是该融合点自己的可学习 query 向量
- `q` 使用全零初始化，因此初始阶段会退化成近似均匀加权，训练更平稳

---

## 3. MiniMind 里 residual branch 是怎么定义的

为了让 dense 和 MoE 共用同一套逻辑，这里把 residual branch 定义为子层输出本身，而不是完整 hidden state。

### 3.1 Attention branch

```text
attn_branch = SelfAttention(RMSNorm(x))
```

### 3.2 Dense FFN branch

```text
ffn_branch = MLP(RMSNorm(h))
```

### 3.3 MoE branch

```text
moe_branch = MoE(RMSNorm(h))
```

这里的 `moe_branch` 指的是：

- router 完成打分
- top-k expert 完成路由
- expert 输出完成加权聚合
- 返回最终的 MoE 子层输出

也就是说：

- 不把 router logits 单独纳入 residual mixer
- 不把 top-k 权重单独纳入 residual mixer
- `aux_loss` 保留原逻辑

---

## 4. 三种模式分别怎么工作

## 4.1 `standard`

保持原始 MiniMind 行为：

```text
x = x + attn(...)
x = x + mlp(...)
```

这条路径的目的主要是：

- 保留 baseline
- 兼容旧权重命名
- 方便后续做对照实验

## 4.2 `attnres_full`

这是最接近论文核心思想的版本。

执行流程：

1. 初始 residual source 只有 embedding hidden
2. 在每一层 attention 前，先对当前所有 residual sources 做一次 mixer
3. 用 mixer 的输出作为 attention 子层输入
4. attention 输出作为一个新的 residual branch 追加到 history
5. 在 MLP/MoE 前，再对更新后的 residual sources 做一次 mixer
6. MLP/MoE 输出再作为一个新的 residual branch 追加到 history
7. 最终输出层再做一次 final mixer，而不是直接只拿最后一层 hidden

可以把它理解成：

- 每层有两个残差融合点
- 每个融合点都“看见”当前已有的全部历史分支

## 4.3 `attnres_block`

这是为了降低实现和实验成本的工程版。

执行流程：

1. 把多个 residual branches 视作一个 block
2. block 内部仍然使用标准残差
3. 每个 block 结束时，收集 block output 作为 block-level state
4. 下一个 block 开始前，用 block mixer 对历史 block states 做聚合
5. 最终输出层再做一次 final mixer

默认 `attnres_block_size=4`，含义是：

- 一个 block 约包含 4 个 residual branches
- 对 MiniMind 这种每层有 attention branch + FFN/MoE branch 的结构来说，约等于 2 个 transformer layers

---

## 5. 代码入口和职责

## 5.1 模型主干

主改动文件：

- `model/model_minimind.py`

新增的关键配置：

- `residual_mode`
- `attnres_block_size`
- `attnres_use_output_norm`
- `attnres_collect_stats`

新增的关键模块：

- `ResidualMixer`

它内部包含：

- 一个 query 向量 `query`
- 一个 `RMSNorm` 作为 key normalization

## 5.2 训练工具层

主改动文件：

- `trainer/trainer_utils.py`

这里统一收口了几件事：

- `add_residual_args`: 给训练/推理脚本统一加 CLI
- `build_lm_config`: 从 args 构建新 config
- `build_optimizer`: 给 post-train 生成 AttnRes 参数组
- `get_weight_path / resolve_weight_path`: 统一新旧权重命名
- `ensure_supported_rollout_engine`: 禁止 AttnRes 走 `sglang` rollout

## 5.3 已接入的新 CLI 脚本

训练侧：

- `trainer/train_pretrain.py`
- `trainer/train_full_sft.py`
- `trainer/train_dpo.py`
- `trainer/train_grpo.py`
- `trainer/train_ppo.py`

推理/评测侧：

- `eval_llm.py`
- `scripts/serve_openai_api.py`
- `scripts/eval_toolcall.py`

导出侧：

- `scripts/convert_model.py`

---

## 6. checkpoint 命名规则

为了避免 AttnRes 权重和旧权重冲突，新增了 residual suffix。

命名规则：

```text
{weight}_{hidden_size}{_moe?}{residual_suffix}.pth
```

示例：

- dense baseline: `full_sft_768.pth`
- moe baseline: `full_sft_768_moe.pth`
- dense full attnres: `full_sft_768_attnres_full.pth`
- moe full attnres: `full_sft_768_moe_attnres_full.pth`
- dense block attnres: `full_sft_768_attnres_block_b4.pth`

兼容规则：

- `standard` 模式会优先找新命名
- 如果没有，再回退到旧命名
- `attnres_*` 模式不会回退到旧权重

原因很简单：

- 新结构不保证和旧标准残差权重一一兼容
- AttnRes 最稳妥的做法是从对应阶段重新训练

---

## 7. post-train 为什么要拆参数组

这次只在 post-train 阶段引入 AttnRes 参数组学习率缩放。

当前实现：

- baseline/backbone 参数：原学习率
- `ResidualMixer` 参数：`learning_rate * attnres_lr_scale`
- `ResidualMixer` 参数组的 `weight_decay=0`

原因：

- AttnRes query 和 key_norm 是新增参数
- 如果后训练时全部参数用同一学习率，新增模块容易学得太慢
- 把新模块单独提速，更适合 SFT / DPO / PPO / GRPO 阶段快速适配

当前没有把这个策略强行推广到 pretrain，原因是：

- 先减少变量
- 让 pretrain 和 baseline 更好做 apples-to-apples 对照

---

## 8. dense 和 MoE 的差异点

相同点：

- residual mixer 逻辑完全一致
- checkpoint 命名规则一致
- CLI 配置一致
- post-train 参数组策略一致

不同点：

- dense 的 branch 是普通 MLP 输出
- MoE 的 branch 是 routed expert 聚合后的最终输出
- MoE 仍然额外带 `aux_loss`

实现上你可以把它记成一句话：

> Attention Residuals 改的是“怎么合并分支”，MoE 改的是“怎么产生分支”。

两者在这个项目里是正交关系。

---

## 9. 和论文不完全一致的地方

这部分非常重要，面试时最好主动说明。

## 9.1 保留了兼容式 baseline

论文/官方实现重点不在保留原始结构兼容，但这个项目保留了 `standard` 模式。

原因：

- 方便对照实验
- 方便面试讲 baseline vs. variant
- 方便保留原始 MiniMind 路径

## 9.2 `attnres_block` 是工程缩放版

这里的 block 版不是逐行照搬大规模系统实现，而是基于 MiniMind 的轻量化近似。

特点：

- block 内部仍是标准残差
- block 之间才使用 attention over depth

这样更适合：

- 教学项目
- 小模型实验
- CPU/单卡环境下的快速验证

## 9.3 没有实现外部推理后端适配

当前明确限制：

- `sglang` rollout 不支持 AttnRes
- Qwen 兼容导出不支持 AttnRes

原因：

- 它们都假设了比较固定的标准 Transformer 残差结构
- 这次优先保证项目内训练/推理闭环

如果面试时被问“为什么不做”，可以回答：

> 这轮目标是验证算法改动是否有效，而不是优先做生态兼容；先把 backbone 改通并完成训练闭环，再考虑外部 serving/export 适配。

---

## 10. 推荐实验命令模板

下面只给命令模板，不承诺你当前机器可直接跑通训练。

## 10.1 dense baseline pretrain

```bash
cd trainer
python train_pretrain.py \
  --use_moe 0 \
  --residual_mode standard \
  --max_seq_len 340
```

## 10.2 dense full AttnRes pretrain

```bash
cd trainer
python train_pretrain.py \
  --use_moe 0 \
  --residual_mode attnres_full \
  --attnres_block_size 4 \
  --attnres_collect_stats 1 \
  --max_seq_len 340
```

## 10.3 moe baseline pretrain

```bash
cd trainer
python train_pretrain.py \
  --use_moe 1 \
  --residual_mode standard \
  --max_seq_len 340
```

## 10.4 moe full AttnRes pretrain

```bash
cd trainer
python train_pretrain.py \
  --use_moe 1 \
  --residual_mode attnres_full \
  --attnres_collect_stats 1 \
  --max_seq_len 340
```

SFT / DPO / PPO / GRPO 也是一样的切法，只需要把：

- `--residual_mode`
- `--attnres_block_size`
- `--attnres_use_output_norm`
- `--attnres_collect_stats`
- `--attnres_lr_scale`

带上即可。

---

## 11. GPU 机器上的建议验证顺序

建议按这个顺序，不要一上来同时跑太多变量。

### 第一阶段：结构正确性

先验证：

- dense `standard / attnres_full / attnres_block`
- moe `standard / attnres_full / attnres_block`

每种模式都至少确认：

- 可以实例化
- 可以 forward
- 可以 backward
- 可以 generate
- checkpoint 可保存/加载

### 第二阶段：pretrain 对照

固定数据、batch、token budget，只比较：

- dense baseline vs dense attnres_full
- moe baseline vs moe attnres_full

关键看：

- train loss
- val ppl
- 收敛前期稳定性

### 第三阶段：post-train 对照

推荐先做：

- SFT
- DPO

再看时间和算力决定是否补：

- PPO
- GRPO

### 第四阶段：统计与可视化

重点画这些：

- residual entropy
- average selected depth
- 各层 hidden RMS
- gradient norm

如果 AttnRes 有收益，这几张图通常比单纯的最终 loss 更有说服力。

---

## 12. 当前这轮你需要知道的限制

这轮实现是“先改代码，再换机器测试”，所以有几个现实约束：

- 当前机器没有 CUDA
- 当前 shell 环境里也未确认装有 `torch`
- 所以这轮更偏结构落地和静态校验

已经适合做的事：

- 阅读代码
- 检查接口
- 交叉核对论文和实现
- 准备后续实验脚本

换到新机器后优先做：

1. 小模型 CPU/GPU forward 冒烟
2. 1 到 2 step 的 pretrain / SFT 冒烟
3. checkpoint 存取
4. dense 与 MoE 的最小对照实验

---

## 13. 常见坑

## 13.1 用 AttnRes 权重去加载标准结构

会失败或行为不对。

原因：

- 新增了 mixer 参数
- 残差拓扑已经改变

## 13.2 `sglang` rollout 不能直接用

目前在代码里已经显式限制。

原因：

- AttnRes 改了 backbone 的内部残差结构
- 这轮没有做外部 rollout backend 的兼容适配

## 13.3 Qwen 兼容导出不能直接用

当前只允许：

- 原生 MiniMind 结构导出

Qwen 兼容导出会在 AttnRes 模式下直接报错。

## 13.4 block size 不要乱设太小

`attnres_block_size` 太小会导致：

- block 太碎
- 对照意义变差
- 训练行为更像 full mode，但不一定更稳

建议先用：

- `4`
- `8`

---

## 14. 面试时怎么讲这件事

一个比较自然的说法是：

1. 我选了一个教学型全链路 LLM 项目 MiniMind 作为实验底座
2. 参考 Kimi Attention Residuals，把固定残差加法换成了沿深度维度的可学习聚合
3. 为了便于实验和讲解，我保留了 baseline，同时实现了 full 和 block 两个版本
4. 我没有只改 dense，而是把 MoE 也一并接入，验证这个思路和 routed FFN 是否正交
5. 我把这套改动接到了 pretrain、SFT、DPO、PPO、GRPO 所需的关键入口，并处理了 checkpoint 和脚本兼容
6. 在工程上，我明确限制了暂不支持的生态兼容点，比如 Qwen 导出和 SGLang rollout，优先保证项目内训练闭环

这套叙事的优点是：

- 有算法点
- 有工程落地点
- 有实验设计意识
- 有范围控制意识

---

## 15. 你接下来最值得做的事

建议按这个顺序继续：

1. 在新机器装好 `torch`
2. 先跑 6 组最小 forward/generate 冒烟
3. 再跑 dense baseline vs dense attnres_full 的 pretrain 小实验
4. 再跑 moe baseline vs moe attnres_full
5. 如果结果有信号，再进入 SFT / DPO

如果你后面要继续扩展，我最推荐补的两块是：

- residual stats 的日志落盘与可视化
- README 中增加 AttnRes 专题章节和对照结果表
