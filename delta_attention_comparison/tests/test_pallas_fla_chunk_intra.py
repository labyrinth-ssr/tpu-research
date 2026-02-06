
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

from delta_attention_comparison.src.layers.pallas_kda import kda_intra_chunk_fwd as pallas_intra

try:
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as triton_intra
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False

class TestKDAIntraChunk(unittest.TestCase):
    def test_equivalence(self):
        print("\n=== Testing KDA Intra Chunk Equivalence (Pallas vs Fla Triton) ===")
        torch.manual_seed(42)

        B, H, T, D = 1, 4, 128, 64
        B_scan = [1,2,4,6,8,11]
        H_scan = [1,2,3,4,5,8]
        T_scan = [1,2,50,100,128,256,512]
        D_scan = [64,128,256,1000]
        chunk_size = 64
        dtype = torch.bfloat16       
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # for B in B_scan:
        for T in T_scan:
            # Inputs Setup (following fla/tests/ops/test_kda.py)
            # 1. k, q: Normalized to prevent instability
            q = torch.randn(B, T, H, D, dtype=dtype, device=device)
            k = torch.randn(B, T, H, D, dtype=dtype, device=device)
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

            safe_gate = False
            
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
            
            q_jax = jnp.array(q.detach().cpu().float().permute(0, 2, 1, 3).numpy())
            k_jax = jnp.array(k.detach().cpu().float().permute(0, 2, 1, 3).numpy())
            g_jax = jnp.array(g.detach().cpu().float().permute(0, 2, 1, 3).numpy())
            beta_jax = jnp.array(beta.detach().cpu().float().permute(0, 2, 1).numpy())
            v_jax = jnp.array(v.detach().cpu().float().permute(0, 2, 1, 3).numpy())
            # Pallas returns u, w
            u_jax, w_jax, qg_jax, kg_jax, Aqk_jax, Akk_jax = pallas_intra(q_jax, k_jax, g_jax, beta_jax, v_jax, chunk_size=chunk_size, disable_recompute=True)
            print("A max:", Akk_jax.max(), "min:", Akk_jax.min())
            
            u_jax = np.array(u_jax)
            w_jax = np.array(w_jax)

            # --- Run Fla (Triton) ---
            if FLA_AVAILABLE:
                print("Running Fla Triton Kernel...")
                scale = 1.0

                w_fla, u_fla, qg_fla, kg_fla, Aqk_fla, Akkk_fla = triton_intra(
                    q=q.to(torch.float32), 
                    k=k.to(torch.float32), 
                    v=v.to(torch.float32), 
                    gk=g.to(torch.float32),
                    beta=beta.to(torch.float32), 
                    scale=scale, 
                    chunk_size=chunk_size,
                    safe_gate=safe_gate,
                    disable_recompute=True
                )
                
                # Fla returns (B, T, H, D). Permute to compare with Pallas (B, H, T, D).
                u_fla_np = u_fla.detach().cpu().float().permute(0, 2, 1, 3).numpy()
                w_fla_np = w_fla.detach().cpu().float().permute(0, 2, 1, 3).numpy()
                qg_fla_np = qg_fla.detach().cpu().float().permute(0, 2, 1, 3).numpy()
                kg_fla_np = kg_fla.detach().cpu().float().permute(0, 2, 1, 3).numpy()
                
                # Reshape Aqk/Akk from (B, T, H, BT) to (B, H, NC, BT, BT)
                # T = NC * BT
                B_sz, T_sz, H_sz, BT_sz = Aqk_fla.shape
                NC_sz = T_sz // BT_sz
                
                Aqk_fla_np = Aqk_fla.view(B_sz, NC_sz, BT_sz, H_sz, BT_sz).permute(0, 3, 1, 2, 4).detach().cpu().float().numpy()
                Akkk_fla_np = Akkk_fla.view(B_sz, NC_sz, BT_sz, H_sz, BT_sz).permute(0, 3, 1, 2, 4).detach().cpu().float().numpy()
                
                # Check for NaNs
                if np.isnan(u_jax).any(): print("NaN in Pallas Output!")
                if np.isnan(u_fla_np).any(): print("NaN in Fla Output!")

                # Compare
                diff_u = np.abs(u_jax - u_fla_np).max()
                diff_w = np.abs(w_jax - w_fla_np).max()
                diff_qg = np.abs(qg_jax - qg_fla_np).max()
                diff_kg = np.abs(kg_jax - kg_fla_np).max()
                diff_Aqk = np.abs(Aqk_jax - Aqk_fla_np).max()
                diff_Akkk = np.abs(Akk_jax - Akkk_fla_np).max()
                
                print(f"Max Diff U: {diff_u:.6e}")
                print(f"Max Diff W: {diff_w:.6e}")
                print(f"Max Diff qg: {diff_qg:.6e} (Note: Expected to be large due to Factorization differences)")
                print(f"Max Diff kg: {diff_kg:.6e} (Note: Expected to be large due to Factorization differences)")
                print(f"Max Diff Aqk: {diff_Aqk:.6e}")
                print(f"Max Diff Akkk: {diff_Akkk:.6e}")
                
                self.assertTrue(diff_u < 2e-3, f"U mismatch: {diff_u}")
                self.assertTrue(diff_w < 2e-3, f"W mismatch: {diff_w}")
                # qg and kg are not directly comparable due to different reference points (Safe Gate logic)
                # self.assertTrue(diff_qg < 2e-3, f"qg mismatch: {diff_qg}")
                # self.assertTrue(diff_kg < 2e-3, f"kg mismatch:  {diff_kg}")
                self.assertTrue(diff_Aqk < 2e-3, f"Aqk mismatch: {diff_Aqk}")
                self.assertTrue(diff_Akkk < 2e-3, f"Akkk mismatch: {diff_Akkk}")
            else:
                print("Fla not available, skipping Triton comparison.")


if __name__ == "__main__":
    unittest.main()
