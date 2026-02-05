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

from delta_attention_comparison.src.layers.pallas_kda import kda_intra_chunk_bwd as pallas_intra_bwd

try:
    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as triton_intra_bwd
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False

class TestKDAIntraChunkBwd(unittest.TestCase):
    def test_equivalence(self):
        print("\n=== Testing KDA Intra Chunk Backward Equivalence (Pallas vs Fla Triton) ===")
        torch.manual_seed(42)

        B, H, T, D = 1, 1, 256, 128
        chunk_size = 64
        dtype = torch.bfloat16        
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Inputs Setup
        # 1. k, q: Normalized to prevent instability
        q = torch.randn(B, T, H, D, dtype=dtype, device=device)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        safe_gate = False
        
        # 2. Beta: Sigmoid to (0, 1)
        beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
        
        # 3. Gate g: Log-space decay. 
        g_step = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device))
        g = torch.cumsum(g_step, dim=1) # (B, T, H, D)
        
        # 4. Gradients from next layer (Random)
        # dAqk and dAkk in Fla are expected to be (B, T, H, BT)
        # But wait, in the forward test, Aqk output from Fla was reshaped from (B, T, H, BT).
        # Let's verify Fla's expected shape for dAqk/dAkk.
        # In chunk_intra.py: dAqk += (bos * H + i_h) * BT.
        # So yes, they are (B, T, H, BT) in memory layout conceptually, but effectively (B, T, H, chunk_size).
        
        dAqk = torch.randn(B, T, H, chunk_size, dtype=dtype, device=device)
        dAkk = torch.randn(B, T, H, chunk_size, dtype=dtype, device=device)

        # --- Run Pallas (JAX/TPU) ---
        print("Running Pallas Kernel...")
        
        # Prepare inputs for Pallas (B, H, T, D)
        q_jax = jnp.array(q.detach().cpu().float().permute(0, 2, 1, 3).numpy())
        k_jax = jnp.array(k.detach().cpu().float().permute(0, 2, 1, 3).numpy())
        g_jax = jnp.array(g.detach().cpu().float().permute(0, 2, 1, 3).numpy())
        beta_jax = jnp.array(beta.detach().cpu().float().permute(0, 2, 1).numpy())
        
        # Prepare Gradients for Pallas
        # Fla dAqk: (B, T, H, BT). Pallas expects: (B, H, num_chunks, chunk_size, chunk_size)
        # We need to reshape dAqk from Fla format to Pallas format.
        # Fla (B, T, H, BT) -> reshape (B, NC, BT, H, BT) -> permute (B, H, NC, BT, BT)
        num_chunks = T // chunk_size
        dAqk_jax = jnp.array(dAqk.detach().cpu().float().reshape(B, num_chunks, chunk_size, H, chunk_size).permute(0, 3, 1, 2, 4).numpy())
        print("daqk max:", dAqk.max().item(), "dAqk min:", dAqk.min().item())
        dAkk_jax = jnp.array(dAkk.detach().cpu().float().reshape(B, num_chunks, chunk_size, H, chunk_size).permute(0, 3, 1, 2, 4).numpy())

        # Pallas returns dq, dk, dg, dbeta
        dq_jax, dk_jax, dg_jax, dbeta_jax = pallas_intra_bwd(
            q_jax, k_jax, g_jax, beta_jax, 
            segment_ids=None,
            dAqk=dAqk_jax, dAkk=dAkk_jax,
            chunk_size=chunk_size
        )
        
        dq_jax = np.array(dq_jax)
        dk_jax = np.array(dk_jax)
        dg_jax = np.array(dg_jax)
        dbeta_jax = np.array(dbeta_jax)

        # --- Run Fla (Triton) ---
        if FLA_AVAILABLE:
            print("Running Fla Triton Kernel...")
            
            # Initialize gradients for Fla
            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k)
            dg = torch.zeros_like(g)
            db = torch.zeros_like(beta) # dbeta in Fla is accumulated into this
            
            # Fla bwd call
            dq_fla, dk_fla, db_fla, dg_fla = triton_intra_bwd(
                q=q.to(torch.float32), 
                k=k.to(torch.float32), 
                g=g.to(torch.float32), 
                beta=beta.to(torch.float32),
                dAqk=dAqk.to(torch.float32),
                dAkk=dAkk.to(torch.float32),
                dq=dq.to(torch.float32),
                dk=dk.to(torch.float32),
                db=db.to(torch.float32),
                dg=dg.to(torch.float32),
                chunk_size=chunk_size,
                safe_gate=safe_gate
            )
            
            # Fla returns (B, T, H, D). Permute to compare with Pallas (B, H, T, D).
            dq_fla_np = dq_fla.detach().cpu().float().permute(0, 2, 1, 3).numpy()
            dk_fla_np = dk_fla.detach().cpu().float().permute(0, 2, 1, 3).numpy()
            dg_fla_np = dg_fla.detach().cpu().float().permute(0, 2, 1, 3).numpy()
            db_fla_np = db_fla.detach().cpu().float().permute(0, 2, 1).numpy()
            
            # Check for NaNs
            if np.isnan(dq_jax).any(): print("NaN in Pallas dq!")
            if np.isnan(dq_fla_np).any(): print("NaN in Fla dq!")

            # Compare
            diff_dq = np.abs(dq_jax - dq_fla_np).max()
            diff_dk = np.abs(dk_jax - dk_fla_np).max()
            diff_dg = np.abs(dg_jax - dg_fla_np).max()
            diff_db = np.abs(dbeta_jax - db_fla_np).max()
            
            print(f"Max Diff dq: {diff_dq:.6e}")
            print(f"Max Diff dk: {diff_dk:.6e}")
            print(f"Max Diff dg: {diff_dg:.6e}")
            print(f"Max Diff db: {diff_db:.6e}")
            
            # Tolerances might need adjustment depending on bf16/float32 mixed precision
            self.assertTrue(diff_dq < 1e-2, f"dq mismatch: {diff_dq}")
            self.assertTrue(diff_dk < 1e-2, f"dk mismatch: {diff_dk}")
            self.assertTrue(diff_dg < 1e-2, f"dg mismatch: {diff_dg}")
            self.assertTrue(diff_db < 1e-2, f"db mismatch: {diff_db}")

        else:
            print("Fla not available, skipping Triton comparison.")


if __name__ == "__main__":
    unittest.main()
