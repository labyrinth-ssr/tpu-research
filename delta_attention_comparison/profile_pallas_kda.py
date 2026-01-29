import os
import jax
import jax.numpy as jnp
import functools
from src.layers.pallas_kda import kda_intra_chunk_fwd

def profile_kernel():
    # 1. Configuration
    B, H, T, D = 1, 16, 4096, 128
    CHUNK_SIZE = 128
    dtype = jnp.float32
    
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
    profile_kernel()
