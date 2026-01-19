# Pallas Version Kimi Delta Attention Implementation Plan

## 1. Understanding the Goal
目标是使用 JAX Pallas 实现 `delta_attention_comparison/src/layers/kimi_delta_attention.py` 中的核心逻辑，即 `chunk_parallel_delta_attention` 函数。

当前 JAX 实现使用了 `jax.vmap` 和 `jax.lax.scan` 来组合块级并行（Chunkwise Parallel）和块间递归（Inter-chunk Recurrence）。使用 Pallas 的目的是通过手动管理 TPU 的内存层级（HBM -> VMEM -> VREG）和计算流水线，来显著提升该算子在 TPU 上的执行效率，减少 HBM 访问开销，并优化 Flash Attention 风格的 fused kernel。

通过分析 `@fla/fla/ops/kda` 中的 Triton 实现，我们确认了 **Fused approach** (融合 Intra-chunk 计算与 Inter-chunk Recurrence) 的可行性与必要性，同时借鉴其 **Block-wise Forward Substitution** 策略来优化块内计算。

## 2. Investigation & Analysis

### 2.1 核心算法分析
需要深入分析 `delta_attention_comparison/src/layers/kimi_delta_attention.py` 中的数学逻辑，特别是：
*   **Input Layout**: `q`, `k`, `v`, `g`, `beta` 的形状和维度。
*   **Chunk Logic**: 块内计算逻辑，涉及 `A_raw`, `A_inv` (矩阵求逆/求解), `u`, `w` 的计算。
    *   *Insight from Triton*: Triton 实现 (`chunk_intra.py`) 并没有直接做全块 ($64 \times 64$) 的求逆，而是采用分块 ($BC=16$ or $32$) 的 Forward Substitution。这能更好地利用 Tensor Cores (Matrix Units)。
*   **Recurrence Logic**: 块间的递归更新公式 $S_t = S_{t-1} \times 	\text{decay} + 	\text{update}$。
*   **State Size**: 递归状态 $S$ 的大小通常是 `[head_dim, head_dim]` (即 $d \times d$)。对于 $d=128$，这是 $128 \times 128 \times 4$ bytes $\approx 64$ KB，完全可以驻留在 TPU VMEM 中。

### 2.2 现有参考资源
*   **Reference Implementation**: `delta_attention_comparison/src/layers/kimi_delta_attention.py` (JAX Reference)
*   **Triton Reference**: `@fla/fla/ops/kda/**`
    *   `chunk_intra.py`: 核心参考。展示了如何通过 `chunk_kda_fwd_kernel_inter_solve_fused` 高效计算 Intra-chunk 的 $A$ 和 $A^{-1}$。
    *   `fused_recurrent.py`: 展示了 Recurrence 逻辑。
*   **Pallas Examples**: `tpu_inference_kernel/flash_attention/kernel.py` (Pallas 语法参考)

### 2.3 关键待决问题
1.  **Grid Strategy**: 采用 `(Batch, Head)` 作为 Grid 并在 Kernel 内部循环 Sequence chunks（State 驻留 VMEM）。
2.  **Intra-Chunk Computation**: JAX 原版使用 `fori_loop` 逐行更新来实现 $(I-A)^{-1}$。这种标量/向量级操作在 TPU 上效率低下。
    *   *Solution*: 采用 Triton 的 **Block-wise Forward Substitution**。
3.  **Numerical Stability**: Triton 代码中大量处理了 `exp(g - g_cumsum)` 的数值稳定性。Pallas 实现必须严格复刻这些细节。

## 3. Proposed Strategic Approach

我们将分四个阶段实施：

### Phase 1: Kernel Design & Mathematical Mapping (核心设计)
*   **Task**: 定义 Pallas Kernel 的 Grid 和 Memory Layout，并详细设计 Block-wise Inversion 算法。
*   **Strategy**:
    *   **Grid**: `(Batch, Num_Heads)`。Kernel 线程负责处理整个 Sequence (loop over chunks)。
    *   **Memory**:
        *   `S_state`: VMEM resident, initialized to 0.
        *   `u`, `w` buffers: VMEM resident scratchpad.
    *   **Sub-blocking Strategy**:
        *   将 Chunk Size (BT=64/128) 拆分为 Sub-block Size (BC=16 or 32)。
        *   **Algorithm: Block-wise Forward Substitution (referenced from `fla.ops.utils.solve_tril`)**
            *   Target: Compute $(I + A)^{-1}$ where $A$ is strictly lower triangular.
            *   Recursive Formula for $2 \times 2$ block partition:
                $$ \begin{pmatrix} L_{11} & 0 \\ L_{21} & L_{22} \end{pmatrix}^{-1} = \begin{pmatrix} L_{11}^{-1} & 0 \\ -L_{22}^{-1} L_{21} L_{11}^{-1} & L_{22}^{-1} \end{pmatrix} $$
                where $L = I+A$. Since $A$ is strictly lower triangular, $L_{ii}$ are also lower unit triangular.
            *   **Base Case (16x16)**:
                *   Use row-wise forward substitution (conceptually) or unrolled updates.
                *   In Pallas, we can iterate $i=0..15$: $x_i \leftarrow x_i - \sum_{j < i} A_{ij} x_j$.
            *   **Recursive Step (32x32 -> 64x64)**:
                *   Compute $L_{11}^{-1}$ (recursive).
                *   Compute $L_{22}^{-1}$ (recursive).
                *   Compute off-diagonal: $L_{21}^{-1} = -L_{22}^{-1} (L_{21} L_{11}^{-1})$.
                *   This maps efficiently to TPU Matrix Units (MXU) as it consists of Matrix-Matrix Multiplications (MM).

### Phase 2: Implementation of `pallas_call` (代码实现)
*   **File**: 创建 `delta_attention_comparison/src/layers/pallas_kda.py`。
*   **Step 1 - Kernel Skeleton**: 搭建 Pallas kernel 框架，处理 Input/Output `BlockSpec` 和 Prefetching。
*   **Step 2 - Intra-Chunk Logic (The Hard Part)**:
    *   实现 $g$ 的 local cumsum。
    *   计算 $A_{raw}$。
    *   实现 **Block-wise Inversion**:
        *   Loop over sub-blocks。
        *   Compute inverse of diagonal sub-block (small enough for unrolled math or smaller matrix op)。
        *   Update off-diagonal sub-blocks using matrix multiplication。
    *   Compute $w, u$。
*   **Step 3 - Recurrence Loop**:
    *   Standard Recurrence: $S_{new} = S_{old} 	\times 	\text{decay} + v 	\times k^T$。
    *   Output Projection: $O = q 	\times S_{new}$。
*   **Step 4 - Interface**: Python wrapper matching `chunk_parallel_delta_attention`。

### Phase 3: Integration (集成)
*   **Task**: 将新的 Pallas kernel 集成到 `KimiDeltaAttention` 类中。
*   **Action**: 修改 `delta_attention_comparison/src/layers/kimi_delta_attention.py` 或新建 `KimiDeltaAttentionPallas` 类，增加一个 `backend="pallas"` 选项。

### Phase 4: Performance Tuning (性能调优)
*   **Pipeline**: 利用 `pltpu.make_async_copy` 或 Pallas 的自动流水线特性。
*   **Tiling Tuning**: 调整 `BT` (Chunk Size) 和 `BC` (Sub-chunk Size)。例如尝试 `BT=128, BC=32`。

## 4. Verification Strategy

### 4.1 Correctness (正确性)
*   **Unit Test**: 复用并扩展 `delta_attention_comparison/tests/test_kda_equivalence.py`。
    *   新增 `test_pallas_jax_equivalence`: 对比 Pure JAX (High Precision Reference) vs Pallas Kernel。
    *   重点测试：Random $g$ gate values 很大或很小时的数值稳定性。

### 4.2 Performance (性能)
*   **Benchmark Script**: 编写基准测试脚本，对比 Pure JAX (XLA compiled) vs Pallas Kernel。
*   **Metric**: 测量不同 Sequence Length (1k - 128k) 下的 End-to-End Latency 和 TFLOPs。

## 5. Anticipated Challenges & Considerations

*   **Implementation Complexity**: 在 Pallas (Python DSL) 中手写 Block-wise inversion 逻辑比较繁琐，容易出错。
    *   *Mitigation*: 先用 `jax.numpy` 编写一个 "Reference Block-wise Logic" 并验证其与 "Row-wise Logic" 等价，然后再将其放入 Pallas kernel 中。
*   **Compiler Limitations**: Pallas/Mosaic 编译器对复杂控制流（嵌套循环 + 矩阵运算）的支持可能有限制。
    *   *Mitigation*: 尽量展开 (Unroll) 小循环 (Sub-block loop)，保持 Main Loop (Sequence Chunk loop) 简单。

## 6. Actionable Next Step
执行 **Phase 1** & **Phase 2 (Step 1 & 2)**：创建一个独立的 Pallas 开发脚本，首先尝试实现并验证 **Intra-Chunk 的 Block-wise Inversion** 逻辑的正确性。这是整个算子中最复杂且风险最高的部分。