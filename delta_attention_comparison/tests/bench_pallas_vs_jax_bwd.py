import time
import jax
import jax.numpy as jnp
import numpy as np
import functools

# Add project root to path to ensure imports work
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layers.pallas_kda import kda_intra_chunk_fwd, kda_intra_chunk_bwd

# ==============================================================================
# JAX Reference Implementation (Optimized with vmap/jit)
# ==============================================================================

def compute_chunk_vars_ref(k_blk, g_blk, beta_blk, v_blk, chunk_size=128):
    """
    Reference implementation of KDA intra-chunk computation.
    
    Args:
        k_blk: (C, D)
        g_blk: (C, D) - cumulative sum of logs
        beta_blk: (C,)
        v_blk: (C, D)
    Returns:
        u: (C, D)
        w: (C, D)
    """
    prec = jax.lax.Precision.HIGHEST
    g_diff = jnp.expand_dims(g_blk, -2) - jnp.expand_dims(g_blk, -3)
    decay_full = jnp.exp(g_diff)
    
    idx = jnp.arange(chunk_size)
    
    # [STRICT MASK] Stage 2: i > j (Strict Lower)
    # Matches PyTorch triu(0) masked_fill 0
    mask = idx[:, None] > idx[None, :] 
    decay_mask = jnp.where(jnp.expand_dims(mask, -1), decay_full, 0.0)
    
    A_raw = jnp.einsum('id, jd, ijd -> ij', k_blk, k_blk, decay_mask, precision=prec)

    # [BETA ROW]
    A = A_raw * jnp.expand_dims(beta_blk, -1)
    
    # [INVERT] Matches PyTorch logic A = -A then closure
    A_neg = -A
    
    def invert_body(i, m):
        row = m[i]
        mask_idx = jnp.arange(chunk_size) < i
        row = jnp.where(mask_idx, row, 0.0)
        increment = jnp.dot(row, m, precision=prec)
        increment = jnp.where(mask_idx, increment, 0.0)
        return m.at[i].set(row + increment)

    A_inv = jax.lax.fori_loop(1, chunk_size, invert_body, A_neg)
    
    # [BETA COL] Matches PyTorch (A_inv + I) * beta_col
    T = A_inv + jnp.eye(chunk_size)
    T_final = T * jnp.expand_dims(beta_blk, -2) 
    
    # Compute u, w
    u = jnp.matmul(T_final, v_blk, precision=prec)
    w = jnp.matmul(T_final, k_blk * jnp.exp(g_blk), precision=prec)
    
    return u, w

def compute_A_ref(q_blk, k_blk, g_blk, beta_blk, scale, chunk_size):
    # Logic mirrors kda_intra_chunk_kernel in pallas_kda.py
    
    # Factorization: exp2(g_i - g_j) = exp2(g_i - g_ref) * exp2(g_ref - g_j)
    # g_ref is middle of chunk
    g_ref_idx = chunk_size // 2
    g_ref_val = g_blk[g_ref_idx][None, :]
    g_centered = g_blk - g_ref_val
    
    # Note: Pallas uses exp2, here we use exp2 to match
    q_state = q_blk * jnp.exp2(g_centered)
    k_state_q = k_blk * jnp.exp2(g_centered)
    k_state_k = k_blk * jnp.exp2(-g_centered)
    
    # Akk = k_state_q @ k_state_k.T
    Akk_raw = jnp.dot(k_state_q, k_state_k.T)
    # Aqk = q_state @ k_state_k.T
    Aqk_raw = jnp.dot(q_state, k_state_k.T)
    
    idx = jnp.arange(chunk_size)
    
    # Akk Mask: i > j (Strict Lower)
    mask_akk = idx[:, None] > idx[None, :]
    Akk = jnp.where(mask_akk, Akk_raw * beta_blk[:, None], 0.0)
    
    # Aqk Mask: i >= j (Lower + Diagonal)
    mask_aqk = idx[:, None] >= idx[None, :]
    Aqk = jnp.where(mask_aqk, Aqk_raw * scale, 0.0)
    
    return Aqk, Akk

compute_A_ref_vmap = jax.vmap(jax.vmap(jax.vmap(compute_A_ref, in_axes=(0,0,0,0,None,None)), in_axes=(0,0,0,0,None,None)), in_axes=(0,0,0,0,None,None))

@functools.partial(jax.jit, static_argnames=['chunk_size'])
def jax_intra_chunk_fwd(k, g, beta, v, chunk_size=128):
    """
    JAX baseline that mimics the Pallas interface:
    Input: (B, H, T, D)
    Output: (B, H, T, D)
    """
    B, H, T, D = k.shape
    num_chunks = T // chunk_size
    
    # Reshape to (B, H, num_chunks, chunk_size, D)
    k_c = k.reshape(B, H, num_chunks, chunk_size, D)
    g_c = g.reshape(B, H, num_chunks, chunk_size, D)
    beta_c = beta.reshape(B, H, num_chunks, chunk_size)
    v_c = v.reshape(B, H, num_chunks, chunk_size, D)
    
    # Vmap over Batch(0), Head(1), Chunks(2)
    # compute_chunk_vars_ref expects (C, D) inputs
    vmap_fn = jax.vmap(jax.vmap(jax.vmap(
        functools.partial(compute_chunk_vars_ref, chunk_size=chunk_size), 
        in_axes=(0,0,0,0)), in_axes=(0,0,0,0)), in_axes=(0,0,0,0))
    
    u_c, w_c = vmap_fn(k_c, g_c, beta_c, v_c)
    
    return u_c.reshape(B, H, T, D), w_c.reshape(B, H, T, D)

@functools.partial(jax.jit, static_argnames=['chunk_size'])
def jax_intra_chunk_bwd(q, k, g, beta, dAqk, dAkk, chunk_size=128):
    B, H, T, D = q.shape
    num_chunks = T // chunk_size
    scale = 1.0
    
    def loss_fn(q, k, g, beta):
        q_c = q.reshape(B, H, num_chunks, chunk_size, D)
        k_c = k.reshape(B, H, num_chunks, chunk_size, D)
        g_c = g.reshape(B, H, num_chunks, chunk_size, D)
        beta_c = beta.reshape(B, H, num_chunks, chunk_size)
        
        Aqk, Akk = compute_A_ref_vmap(q_c, k_c, g_c, beta_c, scale, chunk_size)
        return jnp.sum(Aqk * dAqk) + jnp.sum(Akk * dAkk)

    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2, 3))
    return grad_fn(q, k, g, beta)

# ==============================================================================
# Benchmarking Utilities
# ==============================================================================

def benchmark_fn(name, fn, args, n_warmup=5, n_iters=20):
    # Warmup
    print(f"  Warmup {name}...", end="\r")
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    
    # Measure
    print(f"  Timing {name}... ", end="\r")
    start = time.perf_counter()
    for _ in range(n_iters):
        out = fn(*args)
        jax.block_until_ready(out)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / n_iters * 1000
    return avg_time_ms

def main():
    print("=" * 80)
    print("KDA Intra-Chunk Benchmark: JAX (XLA) vs Pallas (TPU Kernel)")
    print(f"Device: {jax.devices()[0]}")
    print("=" * 80)

    # Configurations
    H = 16
    D = 128
    CHUNK_SIZE = 128 # Matching the pallas kernel default/typical usage
    DTYPE = jnp.bfloat16
    
    # Batch sizes and Sequence lengths to test
    configs = [
        # (Batch, SeqLen)
        (1, 1024),
        (1, 4096),
        (2, 4096),
        (4, 4096),
        (8, 4096),
        (1, 8192),
        (1, 16384),
        (1, 32768), 
    ]

    print(f"{'B':<4} | {'T':<6} | {'H':<3} | {'D':<3} | {'JAX FWD':<10} | {'Pal FWD':<10} | {'Spd FWD':<7} | {'JAX BWD':<10} | {'Pal BWD':<10} | {'Spd BWD':<7}")
    print("-" * 100)

    for B, T in configs:
        # Generate Inputs
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        
        q = jax.random.normal(k1, (B, H, T, D), dtype=DTYPE)
        k = jax.random.normal(k1, (B, H, T, D), dtype=DTYPE)
        g_raw = jax.nn.log_sigmoid(jax.random.normal(k2, (B, H, T, D), dtype=DTYPE))
        # Ensure g is cumsum-ed as expected by kernel
        num_chunks = T // CHUNK_SIZE
        g_reshaped = g_raw.reshape(B, H, num_chunks, CHUNK_SIZE, D)
        g = jnp.cumsum(g_reshaped, axis=-2).reshape(B, H, T, D)
        
        beta = jax.nn.sigmoid(jax.random.normal(k3, (B, H, T), dtype=DTYPE))
        v = jax.random.normal(k4, (B, H, T, D), dtype=DTYPE)
        
        # Args for FWD
        args_fwd_jax = (k, g, beta, v, CHUNK_SIZE)
        args_fwd_pallas = (q, k, g, beta, v, CHUNK_SIZE)

        # Args for BWD
        dAqk = jax.random.normal(k5, (B, H, num_chunks, CHUNK_SIZE, CHUNK_SIZE), dtype=DTYPE)
        dAkk = jax.random.normal(k6, (B, H, num_chunks, CHUNK_SIZE, CHUNK_SIZE), dtype=DTYPE)
        args_bwd_jax = (q, k, g, beta, dAqk, dAkk, CHUNK_SIZE)
        # Pallas BWD wrapper needs to match signature of jax BWD or be called appropriately
        # kda_intra_chunk_bwd(q, k, g, beta, segment_ids, dAqk, dAkk, scale, chunk_size)
        
        try:
            # --- FWD Bench ---
            # Benchmark JAX
            t_jax_fwd = benchmark_fn("JAX FWD", jax_intra_chunk_fwd, args_fwd_jax)
            
            # Benchmark Pallas
            # Drop A (3rd output) to match JAX signature
            pallas_fwd_wrapper = lambda *a: kda_intra_chunk_fwd(*a)[:2]
            t_pallas_fwd = benchmark_fn("Pallas FWD", pallas_fwd_wrapper, args_fwd_pallas)
            
            spd_fwd = t_jax_fwd / t_pallas_fwd

            # --- BWD Bench ---
            t_jax_bwd = benchmark_fn("JAX BWD", jax_intra_chunk_bwd, args_bwd_jax)

            pallas_bwd_wrapper = lambda q, k, g, beta, daqk, dakk, cs: kda_intra_chunk_bwd(
                q, k, g, beta, None, daqk, dakk, scale=1.0, chunk_size=cs
            )
            t_pallas_bwd = benchmark_fn("Pallas BWD", pallas_bwd_wrapper, args_bwd_jax)

            spd_bwd = t_jax_bwd / t_pallas_bwd
            
            print(f"{B:<4} | {T:<6} | {H:<3} | {D:<3} | {t_jax_fwd:<10.3f} | {t_pallas_fwd:<10.3f} | {spd_fwd:<7.2f}x | {t_jax_bwd:<10.3f} | {t_pallas_bwd:<10.3f} | {spd_bwd:<7.2f}x")
            
        except Exception as e:
            print(f"{B:<4} | {T:<6} | {H:<3} | {D:<3} | {'ERROR':<10} | {str(e)}")

if __name__ == "__main__":
    main()
