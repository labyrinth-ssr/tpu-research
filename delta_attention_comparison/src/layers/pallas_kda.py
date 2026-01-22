
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

def solve_unit_lower_triangular(A, b):
    """
    Solves (I + A) x = b for x, where A is strictly lower triangular.
    Uses block-based forward substitution for better performance on TPU.
    
    Args:
        A: (N, N) strictly lower triangular matrix in VMEM.
        b: (N, D) matrix in VMEM.
        
    Returns:
        x: (N, D) solution matrix.
    """
    N, D = b.shape
    # Block size for vectorized updates
    B = 16 
    num_blocks = N // B
    
    # Split b into blocks to avoid dynamic_update_slice
    # blocks will be a list of (B, D) arrays
    blocks = jnp.split(b, num_blocks, axis=0)
    
    for i in range(num_blocks):
        start = i * B
        end = (i + 1) * B
        
        # 1. Solve the current diagonal block row-by-row
        A_ii = A[start:end, start:end]
        x_block = blocks[i]
        
        # Unroll the inner loop using list of rows to avoid DUS
        rows = [x_block[r] for r in range(B)]
        for j in range(B):
            if j > 0:
                vec = A_ii[j, :j][None, :]  # Shape (1, j)
                # Stack previously solved rows to form matrix
                mat = jnp.stack(rows[:j])   # Shape (j, D)
                # correction = jnp.dot(vec, mat).squeeze(axis=0) # Shape (D,)
                correction = jax.lax.dot_general(
                    vec, mat,
                    (((1,), (0,)), ((), ())),
                    precision=jax.lax.Precision.HIGHEST
                ).squeeze(axis=0)
                rows[j] = rows[j] - correction
        
        x_block = jnp.stack(rows)
        blocks[i] = x_block
        
        # 2. Update remaining rows using a single matmul
        if i < num_blocks - 1:
            rest_start = (i + 1) * B
            
            # Form the rest of x as a single array for vectorized update
            x_rest = jnp.concatenate(blocks[i+1:], axis=0) # ((N-end), D)
            
            A_rest = A[rest_start:, start:end] # ((N-end), B)
            
            # update = A_rest @ x_block
            # update = jnp.dot(A_rest, x_block)
            update = jax.lax.dot_general(
                A_rest, x_block,
                (((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST
            )
            x_rest = x_rest - update
            
            # Split back into blocks
            remaining_blocks_count = num_blocks - 1 - i
            new_blocks = jnp.split(x_rest, remaining_blocks_count, axis=0)
            
            # Update the list
            for k, nb in enumerate(new_blocks):
                blocks[i + 1 + k] = nb
            
    # Reassemble final result
    x = jnp.concatenate(blocks, axis=0)
    return x

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
