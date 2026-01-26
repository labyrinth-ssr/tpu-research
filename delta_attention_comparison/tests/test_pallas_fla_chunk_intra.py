
import os
import torch
import jax
import jax.numpy as jnp
import numpy as np
import unittest
import torch.nn.functional as F

# Set environment variables for Triton on CPU (if applicable)
os.environ['TRITON_INTERPRET'] = '1'

# Add paths
sys_path_dirs = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../fla'))
]
for p in sys_path_dirs:
    if p not in os.sys.path:
        os.sys.path.append(p)

from delta_attention_comparison.src.layers.pallas_kda import kda_intra_chunk_fwd

try:
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False

class TestKDAIntraChunk(unittest.TestCase):
    def test_equivalence(self):
        print("\n=== Testing KDA Intra Chunk Equivalence (Pallas vs Fla Triton) ===")
        torch.manual_seed(42)

        B, H, T, D = 1, 4, 128, 64
        chunk_size = 64
        dtype = torch.float32
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Inputs Setup (following fla/tests/ops/test_kda.py)
        # 1. k, q: Normalized to prevent instability
        q = torch.randn(B, T, H, D, dtype=dtype, device=device)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        safe_gate = True
        
        v = torch.randn(B, T, H, D, dtype=dtype, device=device)
        
        # 2. Beta: Sigmoid to (0, 1)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
        
        # 3. Gate g: Log-space decay. 
        # Important: Fla's chunk_kda logic typically computes cumulative sum of g before intra-chunk.
        # kda_intra_chunk_fwd also expects cumulative g.
        # We generate random logsigmoid values (always negative) and cumsum them.
        g_step = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device))
        g = torch.cumsum(g_step, dim=1) # (B, T, H, D)
        
        # --- Run Pallas (JAX/TPU) ---
        print("Running Pallas Kernel...")
        # Pallas expects (B, H, T, D) layout usually, check signature:
        # k: (B, H, T, D)
        # Pallas implementation in `pallas_kda.py` takes (B, H, T, D).
        # Our tensors are (B, T, H, D). Need permute.
        
        k_jax = jnp.array(k.detach().cpu().permute(0, 2, 1, 3).numpy())
        g_jax = jnp.array(g.detach().cpu().permute(0, 2, 1, 3).numpy())
        beta_jax = jnp.array(beta.detach().cpu().permute(0, 2, 1).numpy())
        v_jax = jnp.array(v.detach().cpu().permute(0, 2, 1, 3).numpy())
        
        # Pallas returns u, w
        u_jax, w_jax, A = kda_intra_chunk_fwd(k_jax, g_jax, beta_jax, v_jax, chunk_size=chunk_size)
        print("A max:", A.max(), "min:", A.min())
        
        u_jax = np.array(u_jax)
        w_jax = np.array(w_jax)

        # --- Run Fla (Triton) ---
        if FLA_AVAILABLE:
            print("Running Fla Triton Kernel...")
            # Fla's chunk_kda_fwd_intra signature:
            # q, k, v, gk, beta, scale, chunk_size...
            # Inputs expected shape: (B, T, H, D) based on `chunk.py` usage?
            # Let's check `chunk_intra.py`: 
            #   q += (bos * H + i_h) * K ... implies (B*T, H, K) or similar flattened, but wrapper handles it.
            #   Usually Fla ops take (B, T, H, D).
            
            # Key Note: Fla's Triton kernel uses `exp2` for efficiency.
            # So input `g` must be scaled by 1/ln(2) if the Pallas kernel uses `exp`.
            # Pallas: `jnp.exp(safe_g_diff)` -> Natural exponential.
            # Fla: `exp2(b_g - b_gn)` -> Base-2 exponential.
            # So pass g * RCP_LN2 to Fla.
            RCP_LN2 = 1.44269504
            g_fla = g * RCP_LN2
            
            # Scale: Pallas kernel doesn't seem to apply a global scale factor to A?
            # A_raw = sum(k*k*exp(g)). A = A_raw * beta.
            # Fla: b_Aqk += tl.dot(...) ... tl.store(..., b_Aqk * scale).
            # If Pallas effectively uses scale=1.0, we pass 1.0.
            scale = 1.0

            w_fla, u_fla, _, _, _, _ = chunk_kda_fwd_intra(
                q=q, 
                k=k, 
                v=v, 
                gk=g_fla,
                beta=beta, 
                scale=scale, 
                chunk_size=chunk_size,
                safe_gate=safe_gate
            )
            
            # Fla returns (B, T, H, D). Permute to compare with Pallas (B, H, T, D).
            u_fla_np = u_fla.detach().cpu().permute(0, 2, 1, 3).numpy()
            w_fla_np = w_fla.detach().cpu().permute(0, 2, 1, 3).numpy()
            
            # Check for NaNs
            if np.isnan(u_jax).any(): print("NaN in Pallas Output!")
            if np.isnan(u_fla_np).any(): print("NaN in Fla Output!")

            # Compare
            diff_u = np.abs(u_jax - u_fla_np).max()
            diff_w = np.abs(w_jax - w_fla_np).max()
            
            print(f"Max Diff U: {diff_u:.6e}")
            print(f"Max Diff W: {diff_w:.6e}")
            
            self.assertTrue(diff_u < 1e-3, f"U mismatch: {diff_u}")
            self.assertTrue(diff_w < 1e-3, f"W mismatch: {diff_w}")
        else:
            print("Fla not available, skipping Triton comparison.")


if __name__ == "__main__":
    unittest.main()
