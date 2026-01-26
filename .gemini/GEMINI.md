# Gemini Project Context: TPU Research

## 1. Core Mandates
*   **Remote Verification:** Functional changes must be verified on actual hardware (TPU/GPU) via SkyPilot. Local correctness does not guarantee remote performance or correctness (especially for JAX/Pallas).
*   **Infrastructure via SkyPilot:** Use [SkyPilot](https://skypilot.readthedocs.io/) for all resource management. Do not attempt to SSH manually without understanding the SkyPilot sync mechanism.
*   **Package Management:** Use `uv` (not `pip`) for local dependency management.
*   **Scope Isolation:**
    *   **Target Codebase:** `delta_attention_comparison/` (JAX/Pallas). This is where new research happens.
    *   **Reference Codebase:** `fla/` (PyTorch/Triton). This is a submodule for reference/correctness checking. **Do not modify** unless explicitly instructed.

## 2. Architecture & Navigation

### Directory Structure
*   **`delta_attention_comparison/`**: **PRIMARY WORKSPACE**. Contains JAX implementations of attention mechanisms.
    *   `src/layers/`: Place new attention kernels (Pallas/JAX) here.
    *   `benchmark_script.py`: The main entry point for performance testing.
*   **`fla/`**: **READ-ONLY REFERENCE**. A Git submodule containing state-of-the-art Linear Attention models in PyTorch/Triton.
*   **`scripts/`**: Infrastructure automation.
    *   `launch_tpu.sh` / `launch_gpu.sh`: Provisions/Starts the TPU or GPU cluster.
    *   `sync_code.sh`: Forces code synchronization to the remote cluster.

### Key Technologies
*   **Framework:** JAX (Frontend), Pallas (Kernel Language).
*   **Hardware:** Google Cloud TPU (v4, v5e, v6e) & NVIDIA GPUs.
*   **Orchestrator:** SkyPilot.

<!-- ## 3. Remote Development Workflow (TPU & GPU)

### Step 1: Provisioning
To start or update a cluster. Both scripts update their respective cluster name files.

**Option A: TPU (Primary)**
```bash
# Usage: bash scripts/launch_tpu.sh <accelerator_type> <experiment_name>
# Saves to .cluster_name_tpu
bash scripts/launch_tpu.sh tpu-v6e-1 matmul_bench
```

**Option B: GPU (Secondary/Baseline)**
```bash
# Usage: bash scripts/launch_gpu.sh <accelerator_type> <experiment_name>
# Saves to .cluster_name_gpu
bash scripts/launch_gpu.sh L4 matmul_bench
```

### Step 2: Synchronization & Execution
After modifying local code, run it on the remote instance. `scripts/sync_code.sh` auto-detects the most recently used cluster, or you can specify `tpu`/`gpu`.

**Standard Execution Pattern:**
```bash
# 1. Force sync (auto-detects newest)
bash scripts/sync_code.sh
# OR specify target:
# bash scripts/sync_code.sh tpu

# 2. Run benchmark on remote
# Note: Read from the correct file (.cluster_name_tpu or .cluster_name_gpu)
sky exec $(cat .cluster_name_tpu) "uv run --extra tpu python delta_attention_comparison/benchmark_script.py"

bash scripts/sync_code.sh gpu && sky exec $(cat .cluster_name_gpu) "export CUDA_VISIBLE_DEVICES=0; uv run --extra gpu python delta_attention_comparison/tests/test_kda_equivalence.py" 
```

### Step 3: Debugging
*   **Status Check:** `sky status`
*   **Logs:** SkyPilot streams stdout/stderr to your local terminal. -->

## 4. Pallas Kernel Guide (JAX)

### Syntax Overview
*   **Imports:**
    ```python
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    ```

### Core Concepts & Patterns

#### 1. Grid & Memory Hierarchy
*   **Layout (CRITICAL):** The Pallas TPU lowering requires that the **last two dimensions** of your block shape are **divisible by 8 and 128 respectively**, OR be **equal to the respective dimensions of the overall array**.
    *   *Constraint:* `block_shape[-2] % 8 == 0` AND `block_shape[-1] % 128 == 0`.
    *   *Exception:* If `block_shape[dim] == full_array_shape[dim]`, the divisibility requirement is waived for that dimension.
    *   *Anti-Pattern:* **Do NOT reshape** arrays to add a trailing `1` dimension (e.g., changing `(..., D)` to `(..., D, 1)`) just to satisfy this. This leads to inefficient layouts and memory access patterns.
    *   *Best Practice:* If your dimension `D` is not divisible by 128 (e.g., `HEAD_DIM=96`), set the **block size** for that dimension to be the full size `D`.
        ```python
        # Good: Block covers full dimension if not 128-aligned
        # full_array_shape: (B, H, T, 96)
        pl.BlockSpec(block_shape=(..., 128, 96), index_map=...)
        ```

*   **Grid Spec:** Use `pltpu.PrefetchScalarGridSpec` for automatic HBM<->VMEM management in simple cases.
    ```python
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(batch_size, num_heads, num_blocks),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM, ...), ...],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM, ...), ...],
        scratch_shapes=[pltpu.VMEM(...)]
    )
    ```
*   **BlockSpec Syntax:** Use explicit keyword arguments for `pl.BlockSpec` to prevent positional argument errors. The positional signature is `(block_shape, index_map)`, which is often counter-intuitive.
    *   **Wrong:** `pl.BlockSpec(lambda i, j: (i, j), (128, 128))` (Passes lambda as shape!)
    *   **Right:** `pl.BlockSpec(index_map=lambda i, j: (i, j), block_shape=(128, 128))`
*   **Compiler Params:** Use `pltpu.CompilerParams` to control dimension semantics (e.g., parallel vs. arbitrary) and other compiler options.
    ```python
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel", "arbitrary", "arbitrary"),
        # flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True}
    )
    # Usage in pallas_call:
    pl.pallas_call(
        kernel,
        out_shape=...,
        grid_spec=grid_spec,
        compiler_params=compiler_params
    )
    ```
*   **Manual Pipelining (Advanced):** For maximum performance, manage data movement manually to overlap Compute and DMA.
    *   Add a `stage` dimension to your grid (e.g., `grid=(..., 2)`).
    *   Use **Semaphores** (`pltpu.SemaphoreType.DMA`) to synchronize.
    *   Use `pltpu.make_async_copy(src, dst, sem).start()` to trigger transfers.
    *   **Pattern:**
        1.  Wait for previous stage's compute to finish (if dependent).
        2.  Start async load for *next* iteration.
        3.  Compute on *current* iteration's data.
        4.  Wait for current load to finish before next step.

#### 2. Control Flow
*   **Predicated Execution:** Use `@pl.when(condition)` heavily. It is more efficient than `jax.lax.cond` inside kernels for pipeline state management (e.g., handling the first/last block or specific pipeline stages).
    ```python
    @pl.when(stage_index == 0)
    def load_initial():
        ...
    ```

#### 3. Scalar Operations & Types
TPU scalar registers are strictly **32-bit**. Operations on sub-32-bit scalars (e.g., `int8`, `int16`, `bfloat16` loaded from a Ref) will fail or cause compilation errors if not handled correctly.

*   **Storage vs. Usage:**
    *   **Storage:** Use compact types (`jnp.int8`, `jnp.int16`, `jnp.bfloat16`) for metadata tables or constants in VMEM/SMEM to minimize memory footprint and bandwidth.
    *   **Usage:** Explicitly cast to **32-bit** (`jnp.int32`, `jnp.float32`) immediately upon loading a scalar value, before using it in any arithmetic, comparison, or control flow.
*   **Pattern:**
    ```python
    # Example: Loading a scalar index from a compact int8 table
    # table_ref: Ref[int8]
    
    # 1. Load:
    val_i8 = table_ref[idx]
    
    # 2. Cast immediately:
    val_i32 = val_i8.astype(jnp.int32)
    
    # 3. Use:
    # Do NOT use val_i8 directly in `if`, `pl.when`, arithmetic, or slicing
    @pl.when(val_i32 > 0)
    def do_work():
        target_idx = val_i32 * block_size  # Arithmetic safe on i32
        ...
    ```
*   **Advanced: Bitwise Unpacking (for packed sub-32bit types):**
    If you need to extract individual values from a packed 32-bit word (e.g., extracting two `bfloat16` values or eight `int4` values from a `uint32`), use bitwise shifts and masking.
    ```python
    # Example: Unpacking 2x bfloat16 from a uint32 loaded from VMEM
    # packed_val: uint32 containing [bf16_high | bf16_low]
    
    # Extract Low 16 bits
    val_low_bf16 = (packed_val & 0xFFFF0000).bitcast(jnp.float32).astype(jnp.bfloat16)
    
    # Extract High 16 bits
    val_high_bf16 = (packed_val << 16).bitcast(jnp.float32).astype(jnp.bfloat16)
    ```
*   **Alignment:** Ensure dimensions of sub-32bit arrays stored in memory are aligned to 32-bits (4 bytes).
    *   `int8`: Dimension must be multiple of 4.
    *   `int4`: Dimension must be multiple of 8.
    *   `bfloat16`/`int16`: Dimension must be multiple of 2.

### Optimization Checklist
1.  **Block Size & Alignment:**
    *   Align block sizes to **128 bytes** (or at least 8 elements) for efficient DMA.
    *   Pad dimensions if necessary to maintain alignment.
    *   Standard sweet spot: **128x128** or **128xHEAD_DIM**.
2.  **Memory Types:**
    *   `pltpu.VMEM`: **Local (Fast)**. Most compute happens here.
    *   `pltpu.SMEM`: **Scalar (Shared)**. Use for cross-block reductions (e.g., accumulating global loss).
    *   `pltpu.HBM`: **Global (Slow)**. Avoid direct access; DMA to VMEM first.
3.  **Vectorization:** Avoid scalar loops. Operate on full blocks using `jnp` functions.

### Reference Material
*   **Tokamax:** The submodule `tokamax/` contains production-grade Pallas kernels.
    *   **Study File:** `tokamax/tokamax/_src/ops/attention/pallas_triton.py and tpu_inference_kernel/flash_attention/kernel.py`
    *   **Look for:** Implementation of manual pipelining (using `stage_index`), async copies, and semaphores. Do not import `tokamax` directly in research code; strictly use it as a reference for implementation patterns.