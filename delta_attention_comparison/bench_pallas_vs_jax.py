import time
import jax
import jax.numpy as jnp
import numpy as np
import functools

# Add project root to path to ensure imports work
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layers.pallas_kda import kda_intra_chunk_fwd

# ==============================================================================
# JAX Reference Implementation (Optimized with vmap/jit)
# ==============================================================================

def compute_chunk_vars_ref(k_blk, g_blk, beta_blk, v_blk, chunk_size=128):
    """
    Standard JAX implementation of the intra-chunk logic using solve_triangular.
    """
    prec = jax.lax.Precision.HIGHEST

    # g_diff[i, j] = g[i] - g[j]
    g_diff = g_blk[:, None, :] - g_blk[None, :, :]
    
    idx = jnp.arange(chunk_size)
    mask = idx[:, None] > idx[None, :] 
    
    # Mask positive exponents
    safe_g_diff = jnp.where(jnp.expand_dims(mask, -1), g_diff, -float('inf'))
    
    # A = tril(K * K^T * exp(g_diff)) * beta
    term = (k_blk[:, None, :] * k_blk[None, :, :]) * jnp.exp(safe_g_diff)
    A_raw = jnp.sum(term, axis=-1)
    A = A_raw * jnp.expand_dims(beta_blk, -1)
    
    eye = jnp.eye(chunk_size, dtype=A.dtype)
    L = eye + A
    
    # Solve (I + A) X = [V, K*exp(g)]
    # We solve them separately here for clarity, though concatenation is possible
    T = jax.scipy.linalg.solve_triangular(L, eye, lower=True)
    T_final = T * jnp.expand_dims(beta_blk, -1) 
    
    u = jnp.matmul(T_final, v_blk, precision=prec)
    w = jnp.matmul(T_final, k_blk * jnp.exp(g_blk), precision=prec)
    
    return u, w

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
    CHUNK_SIZE = 256
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

    print(f"{'B':<4} | {'T':<6} | {'H':<3} | {'D':<3} | {'JAX (ms)':<10} | {'Pallas (ms)':<12} | {'Speedup':<8}")
    print("-" * 75)

    for B, T in configs:
        # Generate Inputs
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        k = jax.random.normal(k1, (B, H, T, D), dtype=DTYPE)
        g_raw = jax.nn.log_sigmoid(jax.random.normal(k2, (B, H, T, D), dtype=DTYPE))
        # Ensure g is cumsum-ed as expected by kernel
        g_reshaped = g_raw.reshape(B, H, T // CHUNK_SIZE, CHUNK_SIZE, D)
        g = jnp.cumsum(g_reshaped, axis=-2).reshape(B, H, T, D)
        
        beta = jax.nn.sigmoid(jax.random.normal(k3, (B, H, T), dtype=DTYPE))
        v = jax.random.normal(k4, (B, H, T, D), dtype=DTYPE)
        
        args = (k, g, beta, v, CHUNK_SIZE)

        try:
            # Benchmark JAX
            t_jax = benchmark_fn("JAX", jax_intra_chunk_fwd, args)
            
            # Benchmark Pallas
            # Drop A (3rd output) to match JAX signature
            pallas_wrapper = lambda *a: kda_intra_chunk_fwd(*a)[:2]
            t_pallas = benchmark_fn("Pallas", pallas_wrapper, args)
            
            speedup = t_jax / t_pallas
            print(f"{B:<4} | {T:<6} | {H:<3} | {D:<3} | {t_jax:<10.3f} | {t_pallas:<12.3f} | {speedup:<8.2f}x")
            
        except Exception as e:

            print(f"{B:<4} | {T:<6} | {H:<3} | {D:<3} | {'ERROR':<10} | {'ERROR':<12} | {str(e)}")

if __name__ == "__main__":
    main()
