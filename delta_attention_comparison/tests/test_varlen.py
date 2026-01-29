import os
import torch
import jax
import jax.numpy as jnp
import numpy as np
import unittest
import torch.nn.functional as F

# Add paths
sys_path_dirs = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
]
for p in sys_path_dirs:
    if p not in os.sys.path:
        os.sys.path.append(p)

from delta_attention_comparison.src.layers.pallas_kda import kda_intra_chunk_fwd

class TestPallasVarLen(unittest.TestCase):
    def test_varlen_isolation(self):
        print("\n=== Testing Pallas KDA Variable Length Sequence Support ===")
        torch.manual_seed(42)

        # Config
        H, D = 4, 64
        chunk_size = 128
        # Total tokens needs to be large enough to span multiple chunks
        # and support arbitrary split points
        total_T = 2048 
        assert total_T % chunk_size == 0
        
        # 1. Generate Input Data (Single batch, packed)
        q = torch.randn(1, total_T, H, D)
        k = torch.randn(1, total_T, H, D)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        v = torch.randn(1, total_T, H, D)
        beta = torch.randn(1, total_T, H).sigmoid()
        
        # G must be cumsum'd. But for varlen, it should be cumsum'd PER SEQUENCE.
        # Here we just generate a global random G for simplicity of implementation testing.
        # The Kernel masking logic should work regardless of G values.
        # However, to be strict, let's just use random values and assume they are pre-cumsumed.
        g = torch.randn(1, total_T, H, D)
        
        # 2. Define Segment IDs (VarLen Layout)
        # Create 3 segments:
        # Seq 0: [0, 150) -> spans chunk 0, 1
        # Seq 1: [150, 200) -> inside chunk 1
        # Seq 2: [200, 1024) -> spans chunk 1...7
        # Seq 3: [1024, 2048) -> spans chunk 8...15
        
        # Simplified for chunks:
        # Let's make random cuts
        splits = [0, 100, 300, 1000, 1500, total_T]
        segment_ids = torch.zeros(1, total_T, dtype=torch.int32)
        
        for i in range(len(splits) - 1):
            start = splits[i]
            end = splits[i+1]
            segment_ids[:, start:end] = i
            
        print(f"Split points: {splits}")
        
        # 3. Run Pallas (Packed VarLen Mode)
        print("Running Pallas Kernel in Packed VarLen Mode...")
        # Prepare JAX inputs: (B, H, T, D)
        q_jax = jnp.array(q.permute(0, 2, 1, 3).numpy())
        k_jax = jnp.array(k.permute(0, 2, 1, 3).numpy())
        g_jax = jnp.array(g.permute(0, 2, 1, 3).numpy())
        beta_jax = jnp.array(beta.permute(0, 2, 1).numpy())
        v_jax = jnp.array(v.permute(0, 2, 1, 3).numpy())
        # segment_ids: (B, T)
        seg_jax = jnp.array(segment_ids.numpy())

        u_packed, w_packed, _, _, _, _ = kda_intra_chunk_fwd(
            q_jax, k_jax, g_jax, beta_jax, v_jax, 
            segment_ids=seg_jax, 
            chunk_size=chunk_size
        )
        u_packed = np.array(u_packed) # (B, H, T, D)
        w_packed = np.array(w_packed)

        # 4. Run Reference (Split & Loop Mode)
        print("Running Reference (Split & Loop)...")
        u_ref_list = []
        w_ref_list = []
        
        # We need to reconstruct the full output tensor.
        # Since the packed output is (B, H, T, D), we will fill a similar buffer.
        u_ref = np.zeros_like(u_packed)
        w_ref = np.zeros_like(w_packed)
        
        for i in range(len(splits) - 1):
            start = splits[i]
            end = splits[i+1]
            
            # Pad length to be divisible by chunk_size for the kernel requirement
            # Or assume we can just pass the sliced data if it's chunk aligned?
            # Actually, `kda_intra_chunk_fwd` asserts T % chunk_size == 0.
            # So for the reference run, we can't easily use the SAME kernel on arbitrary slice lengths 
            # unless we pad them.
            
            # Workaround:
            # We trust that the kernel works for standard sequences (tested in `test_pallas_fla_chunk_intra.py`).
            # So if we run VarLen mode, we expect:
            # Output at [start:end] should ONLY depend on Input at [start:end].
            
            # Let's verify this property: Isolation.
            # If we change input values outside [start:end], the output at [start:end] should NOT change.
            pass

        # Better Verification Strategy:
        # Run 1: Original packed data.
        # Run 2: Modified packed data where we perturb Seq 0.
        # Check: Seq 1's output should be IDENTICAL between Run 1 and Run 2.
        
        print("Verifying Isolation Property...")
        
        # Create a perturbed input
        q_perturb = q.clone()
        k_perturb = k.clone()
        # Perturb the first sequence (0:100) significantly
        q_perturb[:, 0:100] += torch.randn_like(q_perturb[:, 0:100]) * 10
        k_perturb[:, 0:100] += torch.randn_like(k_perturb[:, 0:100]) * 10
        
        q_perturb_jax = jnp.array(q_perturb.permute(0, 2, 1, 3).numpy())
        k_perturb_jax = jnp.array(k_perturb.permute(0, 2, 1, 3).numpy())
        
        u_perturb, w_perturb, _, _, _, _ = kda_intra_chunk_fwd(
            q_perturb_jax, k_perturb_jax, g_jax, beta_jax, v_jax, 
            segment_ids=seg_jax, 
            chunk_size=chunk_size
        )
        u_perturb = np.array(u_perturb)
        
        # Check Seq 1 (100:300)
        # It should be exactly the same as u_packed[:, :, 100:300, :]
        # Because Seq 0 (perturbed) should not leak into Seq 1.
        
        seq1_start, seq1_end = splits[1], splits[2]
        print(f"Checking Seq 1 range: [{seq1_start}, {seq1_end})")
        
        diff_seq1 = np.abs(u_packed[:, :, seq1_start:seq1_end, :] - u_perturb[:, :, seq1_start:seq1_end, :]).max()
        print(f"Max Diff in Seq 1 after perturbing Seq 0: {diff_seq1:.6e}")
        
        self.assertTrue(diff_seq1 < 1e-6, f"Isolation Failed! Perturbing Seq 0 affected Seq 1 by {diff_seq1}")
        
        # Check Seq 0
        # It SHOULD be different
        diff_seq0 = np.abs(u_packed[:, :, 0:100, :] - u_perturb[:, :, 0:100, :]).max()
        print(f"Max Diff in Seq 0 (Expected > 0): {diff_seq0:.6e}")
        self.assertTrue(diff_seq0 > 1e-3, "Perturbation didn't work?")

if __name__ == "__main__":
    unittest.main()
