  核心数学逻辑

  该模型的核心思想是：记忆状态 $S$ 不应该仅仅累加新的 $K, V$ 对（像标准的 Linear Attention），而应该根据当前状态对 $K$
  的预测误差来更新。 这就像是一个在线的梯度下降（Online Gradient Descent）过程。

  1. 符号定义
   - $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t \in \mathbb{R}^d$: 时刻 $t$ 的 Query, Key, Value 向量。
   - $S_t \in \mathbb{R}^{d \times d}$: 时刻 $t$ 的记忆状态矩阵（相当于 Transformer 中的 KV Cache，或者是 RNN 的 Hidden
     State）。
   - $\beta_t \in \mathbb{R}$: 更新强度（步长/学习率）。
   - $g_t \in \mathbb{R}$: 遗忘门的 Log 值（代码中通过 cumsum 累积使用）。

  2. Delta 更新规则 (The Delta Rule)
  标准的 Linear Attention 更新规则是 $S_t = S_{t-1} + \mathbf{v}_t \mathbf{k}_t^T$。
  但在 Delta Attention 中，我们希望 $S_t \mathbf{k}_t \approx \mathbf{v}t$。因此，我们计算一个**残差 (Residual/Delta)**
  $\mathbf{v}{new}$：

  $$
  \mathbf{v}_{new, t} = \mathbf{v}_t - S_{t-1} \mathbf{k}_t
  $$

  然后用这个残差来更新状态：

  $$
  S_t = S_{t-1} + \beta_t \mathbf{v}_{new, t} \mathbf{k}_t^T
  $$

  引入遗忘门（Decay）后，公式变为：

  $$
  S_t = \lambda_t S_{t-1} + \beta_t \mathbf{v}_{new, t} \mathbf{k}_t^T
  $$
  其中 $\lambda_t = \exp(g_t)$。

  3. 块并行计算 (Chunkwise Parallelism)
  为了在 GPU/TPU 上高效计算，代码使用了 Chunk Parallel 算法。它将序列分为若干块（Chunk），块大小为 $C$（例如 64 或
  128）。

  在块内部，由于 $S_t$ 依赖于 $S_{t-1}$，直接计算是串行的。为了并行化，我们需要求解一个线性系统来一次性得到整个块的
  $\mathbf{v}_{new}$。

  代码中的 compute_chunk_vars_local 函数就是在做这件事。

  块内推导：
  在一个块内，设 $K, V \in \mathbb{R}^{C \times d}$ 是块内的所有 K, V。我们需要找到 $V_{new}$。
  对于块内的第 $i$ 个位置，展开递归式：

  $$
  \mathbf{v}_{new, i} = \mathbf{v}_i - \left( S_{prev} \cdot \text{decay} + \sum_{j < i} \beta_j \mathbf{v}_{new, j}
  \mathbf{k}j^T \cdot \text{decay}{j \to i} \right) \mathbf{k}_i
  $$

  整理后可以写成矩阵形式：
  $$
  V_{new} = V - (S_{prev} K^T \odot \text{Decay})^T - (L - I) V_{new}
  $$
  其中 $L$ 是一个下三角矩阵，表示块内历史更新对当前的影响。
  代码中构建了矩阵 $L = I + A$：

```Python
   term = (k_blk[:, None, :] * k_blk[None, :, :]) * jnp.exp(safe_g_diff) # K * K^T * Decay
   A = jnp.tril(term) * beta # 加权的下三角相关性矩阵
   L = I + A
```

  然后求解线性方程组 $L V_{new} = V_{\text{target}}$：
  $$
  V_{new} = L^{-1} (V - \text{HistoryContribution})
  $$

  代码对应：

```Python
  T = jax.scipy.linalg.solve_triangular(L, eye, lower=True) # 计算 L 的逆 T
  u = T @ v_blk
  w = T @ (k_blk * exp(g)) # 用于计算历史状态的影响
  v_new = u - w @ prev_state # 最终得到块内修正后的 V_new
```

  4. 输出计算 (Output)
  最终的注意力输出由两部分组成：
   1. 历史状态贡献 (Inter-Chunk): $O_{hist} = Q S_{prev}$
   2. 块内贡献 (Intra-Chunk): $O_{intra} = \text{Attention}(Q, K) V_{new}$

  $$
  O = \text{Norm}(O_{hist} + O_{intra})
  $$

  代码对应：

```Python
  o_hist = jnp.matmul(q_i * jnp.exp(g_i), prev_state, precision=prec)
  o_intra = jnp.matmul(attn_local, v_new, precision=prec)
  o_block = o_hist + o_intra
```
  注意这里 o_intra 使用的是修正后的 v_new，而不是原始的 v。

  总结
  KDA (Kimi Delta Attention) 是 DeltaNet 的一种实现。它相比标准 Linear Attention 的关键区别在于：
   1. 写入时修正：不仅是简单的 $KV^T$ 累加，而是先计算当前状态对 $K$ 的拟合误差 $V_{new}$，然后只将误差写入记忆。
   2. 块级并行优化：利用矩阵求逆（三角解）在块内并行计算出修正后的 $V_{new}$，从而避免了纯串行的 RNN
      计算，同时保持了比标准 Attention 更低的推理复杂度（$O(1)$ 推理）。

