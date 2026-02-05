
import os
import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from delta_attention_comparison.src.layers.pallas_kda import kda_intra_chunk_fwd, kda_intra_chunk_bwd

def compute_chunk_vars_ref(k_blk, g_blk, beta_blk, v_blk, chunk_size=128):
    """
    Reference implementation of KDA intra-chunk computation.
    """
    prec = jax.lax.Precision.HIGHEST
    g_diff = jnp.expand_dims(g_blk, -2) - jnp.expand_dims(g_blk, -3)
    decay_full = jnp.exp2(g_diff)
    
    idx = jnp.arange(chunk_size)
    
    mask = idx[:, None] > idx[None, :] 
    decay_mask = jnp.where(jnp.expand_dims(mask, -1), decay_full, 0.0)
    
    A_raw = jnp.einsum('id, jd, ijd -> ij', k_blk, k_blk, decay_mask, precision=prec)

    A = A_raw * jnp.expand_dims(beta_blk, -1)
    
    A_neg = -A
    
    def invert_body(i, m):
        row = m[i]
        mask_idx = jnp.arange(chunk_size) < i
        row = jnp.where(mask_idx, row, 0.0)
        increment = jnp.dot(row, m, precision=prec)
        increment = jnp.where(mask_idx, increment, 0.0)
        return m.at[i].set(row + increment)

    A_inv = jax.lax.fori_loop(1, chunk_size, invert_body, A_neg)
    
    T = A_inv + jnp.eye(chunk_size)
    T_final = T * jnp.expand_dims(beta_blk, -2) 
    
    u = jnp.matmul(T_final, v_blk, precision=prec)
    w = jnp.matmul(T_final, k_blk * jnp.exp2(g_blk), precision=prec)
    
    return u, w, T

# Vmap over Batch, Heads, and Chunks
compute_chunk_ref_vmap = jax.vmap(jax.vmap(jax.vmap(compute_chunk_vars_ref, in_axes=(0,0,0,0,None)), in_axes=(0,0,0,0,None)), in_axes=(0,0,0,0,None))

class TestPallasKDAPadding(unittest.TestCase):
    def test_intra_chunk_fwd_padding(self):
        print(f"JAX Backend: {jax.default_backend()}")
        
        # Config: T is NOT divisible by chunk_size
        chunk_size = 128
        T = 130 # 1 full chunk + 2 elements
        B, H, D = 1, 2, 64
        dtype = jnp.float32
        
        # Seeds
        key = random.PRNGKey(0)
        k1, k2, k3, k4, k5 = random.split(key, 5)
        
        # Init inputs
        k = random.normal(k1, (B, H, T, D), dtype=dtype)
        q = random.normal(k5, (B, H, T, D), dtype=dtype)
        k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        g_raw = jax.nn.log_sigmoid(random.normal(k2, (B, H, T, D), dtype=dtype))
        
        # Prepare g as cumulative sum for pallas input
        # Note: logic for cumsum across chunks if we were processing contiguous stream.
        # But here we assume each chunk treats g as cumulative from start of chunk?
        # Re-reading previous test: "g passed to kernel is cumulative sum from the start of the chunk."
        # So we need to emulate that.
        # For valid T, we have chunks.
        # For padding, we pad T to multiple of 128.
        padded_T = ((T + chunk_size - 1) // chunk_size) * chunk_size
        pad_len = padded_T - T
        
        # We manually construct g input that looks like what Pallas expects (chunk-local cumsum)
        # First, pad g to padded_T to calculate cumsum easily
        g_padded = jnp.pad(g_raw, ((0,0), (0,0), (0, pad_len), (0,0)), mode='edge')
        g_reshaped = g_padded.reshape(B, H, padded_T // chunk_size, chunk_size, D)
        g_cumsum_padded = jnp.cumsum(g_reshaped, axis=-2)
        g_in = g_cumsum_padded.reshape(B, H, padded_T, D)
        
        # Slice back to T for input to function (since function will handle padding internally)
        g_in = g_in[:, :, :T, :]
        
        beta = jax.nn.sigmoid(random.normal(k3, (B, H, T), dtype=dtype))
        v = random.normal(k4, (B, H, T, D), dtype=dtype)
        
        print(f"Testing with T={T}, chunk_size={chunk_size}, padded_T={padded_T}")
        
        # Run Pallas
        try:
            # This should NOT crash now
            u_pallas, w_pallas, _, _, _, _ = kda_intra_chunk_fwd(q, k, g_in, beta, v, chunk_size=chunk_size)
            print("Pallas execution successful.")
        except Exception as e:
            self.fail(f"Pallas execution failed: {e}")

        # Validate Shapes
        self.assertEqual(u_pallas.shape, (B, H, T, D))
        self.assertEqual(w_pallas.shape, (B, H, T, D))

        # Check Output Correctness via Reference (on padded inputs)
        # Pad inputs manually for reference
        k_padded = jnp.pad(k, ((0,0), (0,0), (0, pad_len), (0,0)))
        beta_padded = jnp.pad(beta, ((0,0), (0,0), (0, pad_len)))
        v_padded = jnp.pad(v, ((0,0), (0,0), (0, pad_len), (0,0)))
        
        # g_in is already chunk-cumulative but cut short.
        # We need the full padded g_cumsum for reference
        g_ref_padded = g_cumsum_padded # (B, H, num_chunks, chunk_size, D)
        
        k_c = k_padded.reshape(B, H, -1, chunk_size, D)
        beta_c = beta_padded.reshape(B, H, -1, chunk_size)
        v_c = v_padded.reshape(B, H, -1, chunk_size, D)
        
        u_ref_c, w_ref_c, _ = compute_chunk_ref_vmap(k_c, g_ref_padded, beta_c, v_c, chunk_size)
        
        u_ref = u_ref_c.reshape(B, H, padded_T, D)[:, :, :T, :]
        w_ref = w_ref_c.reshape(B, H, padded_T, D)[:, :, :T, :]
        
        # Compare
        diff_u = jnp.max(jnp.abs(u_ref - u_pallas))
        print(f"Max Diff U: {diff_u}")
        diff_w = jnp.max(jnp.abs(w_ref - w_pallas))
        print(f"Max Diff U: {diff_w}")

        
        np.testing.assert_allclose(u_pallas, u_ref, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(w_pallas, w_ref, atol=1e-5, rtol=1e-4)
        print("Test Passed!")

if __name__ == '__main__':
    unittest.main()
