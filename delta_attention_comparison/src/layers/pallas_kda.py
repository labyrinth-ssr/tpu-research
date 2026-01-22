
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

def solve_unit_lower_triangular(A, b):
    """
    Solves (I + A) x = b for x, where A is strictly lower triangular.
    Uses recursive divide-and-conquer to leverage TPU Matmuls.
    
    Args:
        A: (N, N) strictly lower triangular matrix in VMEM.
        b: (N, D) matrix in VMEM.
        
    Returns:
        x: (N, D) solution matrix.
    """
    N = A.shape[0]
    
    # Base case: Size 1
    # (I + 0) * x = b => x = b
    if N == 1:
        return b
        
    # Split into quadrants
    mid = N // 2
    
    # A structure:
    # [ A00   0 ]
    # [ A10 A11 ]
    # Note: A is strictly lower triangular, so diagonal blocks are also strictly lower triangular.
    # The system is:
    # (I + A00) x0 = b0
    # A10 x0 + (I + A11) x1 = b1  =>  (I + A11) x1 = b1 - A10 x0
    
    A00 = A[:mid, :mid]
    A10 = A[mid:, :mid]
    A11 = A[mid:, mid:]
    
    b0 = b[:mid]
    b1 = b[mid:]
    
    # 1. Solve top half recursively
    x0 = solve_unit_lower_triangular(A00, b0)
    
    # 2. Update bottom RHS: b1' = b1 - A10 @ x0
    # Use precision=HIGHEST for stability
    correction = jax.lax.dot_general(
        A10.astype(jnp.float32), x0.astype(jnp.float32),
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST
    ).astype(b.dtype)
    
    b1_prime = b1 - correction
    
    # 3. Solve bottom half recursively
    x1 = solve_unit_lower_triangular(A11, b1_prime)
    
    return jnp.concatenate([x0, x1], axis=0)

def kda_intra_chunk_kernel(
    # Inputs (Ref)
    k_ref, g_ref, beta_ref, v_ref,
    # Outputs (Ref)
    u_out_ref, w_out_ref,
    # Config
    chunk_size: int,
    head_dim: int,
):
    # Load inputs into VMEM
    # k: (C, D), g: (C, D), beta: (C, 1), v: (C, D)
    k = k_ref[0, 0, 0]
    g = g_ref[0, 0, 0]
    beta = beta_ref[0, 0, 0] # (C, 1)
    v = v_ref[0, 0, 0]

    # 1. Compute A matrix
    # A_raw_ij = sum_d k_id * k_jd * exp(g_id - g_jd)
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    mask = idx[:, None] > idx[None, :]
    
    # Broadcast g to (C, C, D)
    g_diff = g[:, None, :] - g[None, :, :]
    
    # Use einsum and log-space masking for efficiency and stability
    # Use additive masking to avoid boolean broadcast issues in Pallas TPU (vector<i1> reshape issue)
    mask_val = jnp.where(mask, 0.0, -jnp.inf)
    safe_g_diff = g_diff + mask_val[:, :, None]
    
    # Revert to broadcast and sum to avoid Pallas lowering issues with complex einsum.
    k_outer = k[:, None, :] * k[None, :, :]
    term = k_outer * jnp.exp(safe_g_diff)
    A_raw = jnp.sum(term, axis=-1)
    
    # Apply Beta and Mask
    # A[i, j] = A_raw[i, j] * beta[i] if i > j else 0
    A = A_raw * beta
    
    # 2. Batch solve for u and w
    # (I + A) u_unscaled = v
    # (I + A) w_unscaled = k * exp(g)
    
    target_w = k * jnp.exp(g)
    # Combine inputs along D axis to solve together: (C, 2D)
    combined_b = jnp.concatenate([v, target_w], axis=-1)
    combined_x = solve_unit_lower_triangular(A, combined_b)
    
    u = combined_x[:, :head_dim] * beta
    w = combined_x[:, head_dim:] * beta
    
    # Store outputs
    u_out_ref[0, 0, 0] = u
    w_out_ref[0, 0, 0] = w

@functools.partial(jax.jit, static_argnames=['chunk_size'])
def kda_intra_chunk_fwd(
    k: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    v: jax.Array,
    chunk_size: int = 128
):
    """
    Pallas implementation of KDA Intra-Chunk Forward Pass.
    
    Args:
        k: (B, H, T, D) Key
        g: (B, H, T, D) Cumulative Sum of Log-Decay
        beta: (B, H, T) Beta
        v: (B, H, T, D) Value
        chunk_size: Block size for Pallas kernel.
        
    Returns:
        u: (B, H, T, D)
        w: (B, H, T, D)
    """
    B, H, T, D = k.shape
    assert T % chunk_size == 0, "Sequence length must be divisible by chunk_size"
    num_chunks = T // chunk_size
    
    # Reshape to expose chunks: (B, H, num_chunks, chunk_size, D)
    k_reshaped = k.reshape(B, H, num_chunks, chunk_size, D)
    g_reshaped = g.reshape(B, H, num_chunks, chunk_size, D)
    beta_reshaped = beta.reshape(B, H, num_chunks, chunk_size, 1)
    v_reshaped = v.reshape(B, H, num_chunks, chunk_size, D)
    
    grid = (B, H, num_chunks)
    
    # Output buffers
    # Can interpret output as (B, H, num_chunks, chunk_size, D) and then reshape back
    
    # Pallas Call
    u_reshaped, w_reshaped = pl.pallas_call(
        functools.partial(kda_intra_chunk_kernel, chunk_size=chunk_size, head_dim=D),
        out_shape=[
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, D), dtype=k.dtype),
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, D), dtype=k.dtype)
        ],
        in_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # k
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # g
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, 1)), # beta
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # v
        ],
        out_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # u
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # w
        ],
        grid=grid,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel","parallel")),
    )(k_reshaped, g_reshaped, beta_reshaped, v_reshaped)
    
    return u_reshaped.reshape(B, H, T, D), w_reshaped.reshape(B, H, T, D)
