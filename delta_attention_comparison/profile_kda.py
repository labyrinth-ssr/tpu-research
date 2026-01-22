import os
import jax
import jax.numpy as jnp
from flax import nnx
import jax.profiler

from src.layers.kimi_delta_attention import KimiDeltaAttention

# Dummy Config (Same as benchmark_script.py)
class BenchmarkConfig:
    def __init__(self):
        self.emb_dim = 2048
        self.hidden_size = 2048
        self.gdn_num_value_heads = 16
        self.gdn_num_key_heads = 16
        self.gdn_value_head_dim = 128
        self.gdn_key_head_dim = 128
        self.gdn_conv_kernel_dim = 4
        self.gdn_chunk_size = 256
        self.dtype = jnp.bfloat16 # Use bfloat16 for TPU/GPU benchmarking
        self.weight_dtype = jnp.bfloat16
        self.normalization_layer_epsilon = 1e-6
        self.use_qk_norm_in_gdn = True
        self.matmul_precision = "default"

config = BenchmarkConfig()

def main():
    print("-" * 40)
    print(f"JAX Default Backend: {jax.default_backend()}")
    devices = jax.local_devices()
    print(f"Available Devices: {devices}")
    print("-" * 40)
    
    # Parameters from benchmark_script.py
    # Select a representative configuration for profiling to capture a clear trace
    # without generating too much data.
    # BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
    # SEQ_LENS = [1024, 2048, 4096, 8192]
    
    # Profiling specific config:
    BATCH_SIZES = [1]
    SEQ_LENS = [4096] 

    profile_root = "profiles_kda_fwd"
    os.makedirs(profile_root, exist_ok=True)

    rngs = nnx.Rngs(0)
    
    # Kimi Args
    kimi_args = {
        "hidden_size": config.hidden_size,
        "num_heads": config.gdn_num_value_heads,
        "head_dim": config.gdn_value_head_dim,
        "conv_kernel_size": config.gdn_conv_kernel_dim,
        "normalization_layer_epsilon": config.normalization_layer_epsilon,
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
    }
    
    # Initialize Model
    print("Initializing KimiDeltaAttention model...")
    model = KimiDeltaAttention(**kimi_args, rngs=rngs)

    # Define Forward Step
    @nnx.jit
    def forward_fn(model, x):
        # Only run forward pass
        out = model(x)
        # Handle tuple return (out, state)
        if isinstance(out, tuple):
            out = out[0]
        # Return sum to ensure computation isn't optimized away (though block_until_ready handles this)
        return out

    print(f"{'Batch':<6} | {'Seq':<6} | {'Status':<30}")
    print("-" * 50)

    for b in BATCH_SIZES:
        for s in SEQ_LENS:
            try:
                input_shape = (b, s, config.emb_dim)
                key = jax.random.PRNGKey(0)
                x = jax.random.normal(key, input_shape, dtype=config.dtype)
                
                # Warmup
                print(f"{b:<6} | {s:<6} | {'Warming up...':<30}")
                for _ in range(10): # Warmup a bit more for stable traces
                    out = forward_fn(model, x)
                    out.block_until_ready()
                
                # Profile
                trace_dir = os.path.join(profile_root, f"B{b}_L{s}")
                print(f"{b:<6} | {s:<6} | {f'Profiling to {trace_dir}':<30}")
                
                with jax.profiler.trace(trace_dir):
                    # Capture 3 steps
                    for step in range(3):
                        with jax.profiler.TraceAnnotation(f"forward_step_{step}"):
                            out = forward_fn(model, x)
                            out.block_until_ready()
                
                print(f"{b:<6} | {s:<6} | {'Done.':<30}")
                
            except Exception as e:
                print(f"{b:<6} | {s:<6} | {f'Error: {e}':<30}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
