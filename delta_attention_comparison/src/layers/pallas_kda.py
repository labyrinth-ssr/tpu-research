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
    B = 16
    num_blocks = N // B
    A = A.astype(jnp.float32)
    b = b.astype(jnp.float32)
    
    blocks = jnp.split(b, num_blocks, axis=0)
    
    for i in range(num_blocks):
        start = i * B
        end = (i + 1) * B
        
        A_ii = A[start:end, start:end]
        x_block = blocks[i]
        
        rows = [x_block[r] for r in range(B)]
        for j in range(B):
            if j > 0:
                vec = A_ii[j, :j][None, :]
                mat = jnp.stack(rows[:j])
                correction = jax.lax.dot_general(
                    vec, mat,
                    (((1,), (0,)), ((), ())),
<<<<<<< HEAD
                    precision=jax.lax.Precision.HIGHEST,
=======
>>>>>>> a1d2758
                    preferred_element_type=jnp.float32
                ).squeeze(axis=0)
                rows[j] = rows[j] - correction
        
        x_block = jnp.stack(rows)
        blocks[i] = x_block
        
        if i < num_blocks - 1:
            rest_start = (i + 1) * B
            
            x_rest = jnp.concatenate(blocks[i+1:], axis=0)
            A_rest = A[rest_start:, start:end]
            
            update = jax.lax.dot_general(
                A_rest, x_block,
                (((1,), (0,)), ((), ())),
<<<<<<< HEAD
                precision=jax.lax.Precision.HIGHEST,
=======
>>>>>>> a1d2758
                preferred_element_type=jnp.float32
            )
            x_rest = x_rest - update
            
            remaining_blocks_count = num_blocks - 1 - i
            new_blocks = jnp.split(x_rest, remaining_blocks_count, axis=0)
            
            for k, nb in enumerate(new_blocks):
                blocks[i + 1 + k] = nb
            
    x = jnp.concatenate(blocks, axis=0)
    return x

def kda_intra_chunk_kernel(
    # Inputs (Ref)
    q_ref, k_ref, g_ref, beta_ref, v_ref, segment_ids_ref,
    # Outputs (Ref)
    u_out_ref, w_out_ref, qg_out_ref, kg_out_ref, Aqk_out_ref, Akk_inv_out_ref,
    # Config
    chunk_size: int,
    head_dim: int,
    scale: float,
):
    dtype = q_ref.dtype
    # Load inputs into VMEM
    # q: (C, D), k: (C, D), g: (C, D), beta: (C, 1), v: (C, D)
    q = q_ref[0, 0, 0]
    k = k_ref[0, 0, 0]
    g = g_ref[0, 0, 0]
    beta = beta_ref[0, 0, 0] # (C, 1)
    v = v_ref[0, 0, 0]
    segment_ids = segment_ids_ref[0, 0, 0] 

    # 1. Compute A matrix using factorization for TPU MXU efficiency
    # Factorization: exp(g_i - g_j) = exp(g_i - g_ref) * exp(g_ref - g_j)
    # We choose g_ref to be the middle of the chunk for numerical stability (Safe Gate)
    
    # Pick reference g from the middle of the chunk
    g_ref_idx = chunk_size // 2
<<<<<<< HEAD
    g_ref = g[g_ref_idx][None, :] # (1, D)
    g_centered = g.astype(jnp.float32) - g_ref.astype(jnp.float32) # (C, D)
=======
    g_ref_val = g[g_ref_idx][None, :] # (1, D)
    g_centered = (g.astype(jnp.float32) - g_ref_val.astype(jnp.float32)) # (C, D)
>>>>>>> a1d2758

    # Compute Q and K states for matrix multiplication
    # q_state = q * exp(g - g_ref)
    # k_state = k * exp(g_ref - g)
    q_state = q * jnp.exp(g_centered).astype(q.dtype)
    
    # Re-use k_state logic for both Aqk and Akk
    # For Akk: K * exp(g - g_ref) vs K * exp(g_ref - g) logic
    # In original kda: A_ij = k_i * k_j * exp(g_i - g_j)
    # = (k_i * exp(g_i - g_ref)) * (k_j * exp(g_ref - g_j))
    # Let's call them k_state_q (acts like query) and k_state_k (acts like key)
    k_state_q = k * jnp.exp(g_centered)
    k_state_k = k * jnp.exp(-g_centered)
    
    # Perform Matrix Multiplication (C, D) @ (D, C) -> (C, C)
    # This maps to TPU MXU instructions
    # 256x256xbf16 * 256x256xbf16 -> 256x256xf32
    # Akk = k_state_q @ k_state_k.T
    Akk_raw = jax.lax.dot_general(
        k_state_q.astype(k.dtype), 
        k_state_k.astype(k.dtype), 
        (((1,), (1,)), ((), ())),
<<<<<<< HEAD
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32
    )
=======
        # precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32
    ).astype(dtype)
    
>>>>>>> a1d2758
    # Aqk = q_state @ k_state_k.T * scale
    Aqk_raw = jax.lax.dot_general(
        q_state, k_state_k,
        (((1,), (1,)), ((), ())),
<<<<<<< HEAD
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32
    ).astype(q.dtype)
=======
        # precision=jax.lax.Precision.HIGHEST
        preferred_element_type=jnp.float32
    ).astype(dtype)
>>>>>>> a1d2758
    
    # Apply Mask and Beta
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    causal_mask = idx[:, None] > idx[None, :]
    causal_mask_qk = idx[:, None] >= idx[None, :] # Aqk usually includes diagonal

    # Segment mask: i and j must belong to the same segment
    # segment_ids: (C,)
    segment_mask = segment_ids[:, None] == segment_ids[None, :]
    # Combine masks
    mask = causal_mask & segment_mask
    mask_qk = causal_mask_qk & segment_mask

    # Akk[i, j] = Akk_raw[i, j] * beta[i] if i > j else 0
    Akk = jnp.where(mask, Akk_raw * beta, 0.0)
    
    # Aqk[i, j] = Aqk_raw[i, j] * scale if i >= j else 0
    Aqk = jnp.where(mask_qk, Aqk_raw * scale, 0.0)

    # 2. Batch solve for u and w and Akk_inv
    # (I + Akk) u = v * beta
    # (I + Akk) w = k * exp(g) * beta
    # (I + Akk) Akk_inv = I
    
    # Apply beta to RHS inputs BEFORE solving
    v_scaled = v * beta
    target_w = k * jnp.exp(g) * beta
    identity = jnp.eye(chunk_size, dtype=v.dtype)
    
    combined_b = jnp.concatenate([v_scaled, target_w, identity], axis=-1)
    combined_x = solve_unit_lower_triangular(Akk, combined_b)
    
    u = combined_x[:, :head_dim]
    w = combined_x[:, head_dim:2*head_dim]
    Akk_inv = combined_x[:, 2*head_dim:]
    
<<<<<<< HEAD
    # Store outputs
    u_out_ref[0, 0, 0] = u.astype(u_out_ref.dtype)
    w_out_ref[0, 0, 0] = w.astype(w_out_ref.dtype)
    qg_out_ref[0, 0, 0] = q_state # Wait, qg != q_state. qg = q * exp(g). q_state = q * exp(g - g_ref).
    # I need to compute qg separately or adjust.
    # qg = q * exp(g)
    # q_state = q * exp(g - g_ref)
    # So qg = q_state * exp(g_ref)
    
=======
>>>>>>> a1d2758
    qg = q * jnp.exp(g)
    
    g_last = g[chunk_size-1][None, :]
    kg = k * jnp.exp(g_last - g)

    u_out_ref[0, 0, 0] = u.astype(u_out_ref.dtype)
    w_out_ref[0, 0, 0] = w.astype(w_out_ref.dtype)
    qg_out_ref[0, 0, 0] = qg
    kg_out_ref[0, 0, 0] = kg
    Aqk_out_ref[0, 0, 0] = Aqk
    Akk_inv_out_ref[0, 0, 0] = Akk_inv.astype(Akk_inv_out_ref.dtype)

@functools.partial(jax.jit, static_argnames=['chunk_size', 'scale'])
def kda_intra_chunk_fwd(
    q: jax.Array,
    k: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array = None,
    scale: float = 1.0,
    chunk_size: int = 128
):
    """
    Pallas implementation of KDA Intra-Chunk Forward Pass.
    
    Args:
        q: (B, H, T, D) Query
        k: (B, H, T, D) Key
        g: (B, H, T, D) Cumulative Sum of Log-Decay
        beta: (B, H, T) Beta
        v: (B, H, T, D) Value
        segment_ids: (B, T) Segment IDs for variable length sequences. 
                     Tokens with different IDs will not attend to each other.
        scale: Attention scale factor
        chunk_size: Block size for Pallas kernel.
        
    Returns:
        u: (B, H, T, D)
        w: (B, H, T, D)
        qg: (B, H, T, D)
        kg: (B, H, T, D)
        Aqk: (B, H, num_chunks, chunk_size, chunk_size)
        Akk_inv: (B, H, num_chunks, chunk_size, chunk_size)
    """
    B, H, T, D = k.shape
    assert T % chunk_size == 0, "Sequence length must be divisible by chunk_size"
    num_chunks = T // chunk_size

    # Handle segment_ids
    if segment_ids is None:
        # Default: all tokens belong to segment 0 (or distinct segments per batch, doesn't matter since B dim is separated)
        segment_ids = jnp.zeros((B, T), dtype=jnp.int32)
    
    # Reshape to expose chunks: (B, H, num_chunks, chunk_size, D)
    q_reshaped = q.reshape(B, H, num_chunks, chunk_size, D)
    k_reshaped = k.reshape(B, H, num_chunks, chunk_size, D)
    g_reshaped = g.reshape(B, H, num_chunks, chunk_size, D)
    beta_reshaped = beta.reshape(B, H, num_chunks, chunk_size, 1)
    v_reshaped = v.reshape(B, H, num_chunks, chunk_size, D)
    segment_ids_reshaped = segment_ids.reshape(B, 1, num_chunks, chunk_size)
    
    grid = (B, H, num_chunks)
    
    # Pallas Call
    u_reshaped, w_reshaped, qg_reshaped, kg_reshaped, Aqk_reshaped, Akk_inv_reshaped = pl.pallas_call(
        functools.partial(kda_intra_chunk_kernel, chunk_size=chunk_size, head_dim=D, scale=scale),
        out_shape=[
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, D), dtype=k.dtype), # u
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, D), dtype=k.dtype), # w
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, D), dtype=k.dtype), # qg
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, D), dtype=k.dtype), # kg
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, chunk_size), dtype=k.dtype), # Aqk
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, chunk_size), dtype=k.dtype), # Akk_inv
        ],
        in_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # q
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # k
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # g
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, 1)), # beta
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # v
            pl.BlockSpec(index_map=lambda i, j, l: (i, 0, l, 0), block_shape=(1, 1, 1, chunk_size)),       # segment_ids
        ],
        out_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # u
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # w
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # qg
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # kg
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, chunk_size)), # Aqk
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, chunk_size)), # Akk_inv
        ],
        grid=grid,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel","parallel")),
    )(q_reshaped, k_reshaped, g_reshaped, beta_reshaped, v_reshaped, segment_ids_reshaped)
    
    return (
        u_reshaped.reshape(B, H, T, D),
        w_reshaped.reshape(B, H, T, D),
        qg_reshaped.reshape(B, H, T, D),
        kg_reshaped.reshape(B, H, T, D),
        Aqk_reshaped,
        Akk_inv_reshaped
    )