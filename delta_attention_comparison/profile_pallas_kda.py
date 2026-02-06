import os
import jax
import jax.numpy as jnp
import functools
from src.layers.pallas_kda import kda_intra_chunk_fwd

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

def profile_jax():
    B, H, T, D = 4, 16, 8192, 2048
    CHUNK_SIZE = 64
    dtype = jnp.bfloat16
    
    print(f"Profiling Configuration: B={B}, H={H}, T={T}, D={D}, Chunk={CHUNK_SIZE}, Dtype={dtype}")
    
    # 2. Initialize Inputs
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    q = jax.random.normal(k1, (B, H, T, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, T, D), dtype=dtype)
    g = jax.random.normal(k3, (B, H, T, D), dtype=dtype) # Log decay
    beta = jax.random.normal(k4, (B, H, T), dtype=dtype)
    v = jax.random.normal(k5, (B, H, T, D), dtype=dtype)
    
    # 3. JIT Compile & Warmup
    print("Compiling and Warming up...")
    # Trigger JIT
    out = jax_intra_chunk_fwd(k, g, beta, v,  chunk_size=CHUNK_SIZE)
    jax.block_until_ready(out)
    
    # Run a few times to ensure steady state
    for _ in range(5):
        out = jax_intra_chunk_fwd(k, g, beta, v,  chunk_size=CHUNK_SIZE)
        jax.block_until_ready(out)

    # 4. Profile
    profile_dir = "/tmp/tpu_logs/pallas_profile/profiles_jax_kda"
    print(f"Starting Profiler. Output dir: {profile_dir}")
    
    # Start trace
    with jax.profiler.trace(profile_dir):
        # Run multiple iterations to capture average performance
        # Use TraceAnnotation to mark the step in the timeline
        for i in range(5):
            with jax.profiler.TraceAnnotation(f"step_{i}"):
                out = jax_intra_chunk_fwd(k, g, beta, v,  chunk_size=CHUNK_SIZE)
                jax.block_until_ready(out)
                
    print(f"Profiling complete. Run 'tensorboard --logdir={profile_dir}' to view results.")
def profile_kernel():
    # 1. Configuration
    B, H, T, D = 4, 16, 8192, 2048
    CHUNK_SIZE = 64
    dtype = jnp.bfloat16
    
    print(f"Profiling Configuration: B={B}, H={H}, T={T}, D={D}, Chunk={CHUNK_SIZE}, Dtype={dtype}")
    
    # 2. Initialize Inputs
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    q = jax.random.normal(k1, (B, H, T, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, T, D), dtype=dtype)
    g = jax.random.normal(k3, (B, H, T, D), dtype=dtype) # Log decay
    beta = jax.random.normal(k4, (B, H, T), dtype=dtype)
    v = jax.random.normal(k5, (B, H, T, D), dtype=dtype)
    
    # 3. JIT Compile & Warmup
    print("Compiling and Warming up...")
    # Trigger JIT
    out = kda_intra_chunk_fwd(q, k, g, beta, v, scale=1.0, chunk_size=CHUNK_SIZE)
    jax.block_until_ready(out)
    
    # Run a few times to ensure steady state
    for _ in range(5):
        out = kda_intra_chunk_fwd(q, k, g, beta, v, scale=1.0, chunk_size=CHUNK_SIZE)
        jax.block_until_ready(out)

    # 4. Profile
    profile_dir = "/tmp/tpu_logs/pallas_profile/profiles_pallas_kda"
    print(f"Starting Profiler. Output dir: {profile_dir}")
    
    # Start trace
    with jax.profiler.trace(profile_dir):
        # Run multiple iterations to capture average performance
        # Use TraceAnnotation to mark the step in the timeline
        for i in range(5):
            with jax.profiler.TraceAnnotation(f"step_{i}"):
                out = kda_intra_chunk_fwd(q, k, g, beta, v, scale=1.0, chunk_size=CHUNK_SIZE)
                jax.block_until_ready(out)
                
    print(f"Profiling complete. Run 'tensorboard --logdir={profile_dir}' to view results.")

if __name__ == "__main__":
    profile_jax()
    # profile_kernel()
