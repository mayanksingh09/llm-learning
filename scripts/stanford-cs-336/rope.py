"""rope.py: Minimal rotary positional embedding implementation in PyTorch.



Usage example::

    from rope import RotaryEmbedding, apply_rotary_pos_emb

    B, T, H, D = 2, 128, 8, 64  # batch, seq_len, heads, head_dim (D even)
    q = torch.randn(B, T, H, D) # (batch, seq_len, heads, head_dim)
    k = torch.randn_like(q) # rand_like creates a tensor with the same shape as q

    rope = RotaryEmbedding(D)
    cos, sin = rope(seq_len=T, device=q.device, dtype=q.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert (q_rot - q).abs().max() > 1e-4, "RoPE had no effect—check shapes!"

    # Pass q_rot/k_rot to your attention kernel.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last-dim pairs (x₀, x₁) → (−x₁, x₀)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query & key tensors.

    Args:
        q, k: tensors of shape (B, seq_len, n_heads, head_dim) *or any* shape
               where the last two dims are *(seq_len, head_dim)*.
        cos, sin: broadcastable tensors of shape (1, seq_len, 1, head_dim).

    Returns:
        Tuple of rotated (q, k).
    """
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# -----------------------------------------------------------------------------
# Rotary embedding module
# -----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Generate sin/cos tables for Rotary Positional Embeddings.

    Args:
        dim: per‑head dimension (must be even).
        base: frequency base (default **10 000**).
        interleaved: GPT‑J style interleaved freq layout (default **False**).
    """

    def __init__(self, dim: int, base: float = 10_000.0, interleaved: bool = False):
        super().__init__()
        if dim % 2:
            raise ValueError("`dim` must be even for RoPE.")
        self.dim = dim
        self.base = base
        self.interleaved = interleaved 

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(cos, sin)` broadcastable to `(1, seq_len, 1, dim)`."""
        device = device or self.inv_freq.device
        dtype = dtype or self.inv_freq.dtype

        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i , j -> i j", positions, self.inv_freq.to(dtype))

        emb = (
            torch.cat((freqs, freqs), dim=-1)
            if self.interleaved
            else torch.repeat_interleave(freqs, 2, dim=-1)
        )

        cos = emb.cos()[None, :, None, :]  # (1, seq_len, 1, dim)
        sin = emb.sin()[None, :, None, :]
        return cos, sin

# -----------------------------------------------------------------------------
# Minimal sanity check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Understanding Interleaved vs Non-Interleaved RoPE ===\n")
    
    # Let's demonstrate with a small dimension
    dim = 8
    seq_len = 3
    
    print("1. NON-INTERLEAVED (default, used in most models like LLaMA)")
    rope_normal = RotaryEmbedding(dim, interleaved=False)
    cos_normal, sin_normal = rope_normal(seq_len)
    
    print("2. INTERLEAVED (used in GPT-J style models)")
    rope_interleaved = RotaryEmbedding(dim, interleaved=True)
    cos_interleaved, sin_interleaved = rope_interleaved(seq_len)
    
    print(f"Dimension: {dim}, Sequence length: {seq_len}")
    print(f"Base frequencies (inv_freq): {rope_normal.inv_freq}")
    
    print("\n--- Position 0 (should be all 1s for cos, all 0s for sin) ---")
    print(f"Normal cos[0]:       {cos_normal[0, 0, 0]}")
    print(f"Interleaved cos[0]:  {cos_interleaved[0, 0, 0]}")
    
    print("\n--- Position 1 (shows the difference) ---")
    print(f"Normal cos[1]:       {cos_normal[0, 1, 0]}")
    print(f"Normal sin[1]:       {sin_normal[0, 1, 0]}")
    print(f"Interleaved cos[1]:  {cos_interleaved[0, 1, 0]}")
    print(f"Interleaved sin[1]:  {sin_interleaved[0, 1, 0]}")
    
    print("\n--- Explanation ---")
    print("Non-interleaved: frequencies are repeated in pairs")
    print("  Dimensions 0,1 use freq[0]")
    print("  Dimensions 2,3 use freq[1]") 
    print("  Dimensions 4,5 use freq[2]")
    print("  Dimensions 6,7 use freq[3]")
    
    print("\nInterleaved: frequencies are spread across dimensions")
    print("  Dimensions 0,4 use freq[0]")
    print("  Dimensions 1,5 use freq[1]")
    print("  Dimensions 2,6 use freq[2]") 
    print("  Dimensions 3,7 use freq[3]")
    
    # Show the actual frequency arrangement
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, rope_normal.inv_freq)
    
    print(f"\nBase freqs matrix (pos × freq_idx):\n{freqs}")
    
    # Non-interleaved: repeat each frequency twice
    emb_normal = torch.repeat_interleave(freqs, 2, dim=-1)
    print(f"\nNon-interleaved embedding (repeat each freq twice):")
    print(f"Shape: {emb_normal.shape}")
    print(f"Position 1: {emb_normal[1]}")
    
    # Interleaved: concatenate freqs with itself
    emb_interleaved = torch.cat((freqs, freqs), dim=-1)
    print(f"\nInterleaved embedding (concat freqs with itself):")
    print(f"Shape: {emb_interleaved.shape}")
    print(f"Position 1: {emb_interleaved[1]}")
    
    print("\n=== Testing both on actual tensors ===")
    B, T, H, D = 1, 4, 1, 8
    q = torch.randn(B, T, H, D)
    
    # Test normal RoPE
    cos_n, sin_n = rope_normal(T)
    q_rot_normal, _ = apply_rotary_pos_emb(q, q, cos_n, sin_n)
    
    # Test interleaved RoPE  
    cos_i, sin_i = rope_interleaved(T)
    q_rot_interleaved, _ = apply_rotary_pos_emb(q, q, cos_i, sin_i)
    
    print("Original q[0,:,0,:]:")
    print(q[0, :, 0, :])
    print("\nNormal RoPE result q[0,:,0,:]:")
    print(q_rot_normal[0, :, 0, :])
    print("\nInterleaved RoPE result q[0,:,0,:]:")
    print(q_rot_interleaved[0, :, 0, :])
    
    print(f"\nNormal diff max: {(q_rot_normal - q).abs().max().item():.6f}")
    print(f"Interleaved diff max: {(q_rot_interleaved - q).abs().max().item():.6f}")
    
    print("\n=== Summary ===")
    print("• Non-interleaved (default): Used in LLaMA, GPT-NeoX, most modern models")
    print("• Interleaved: Used in GPT-J and some other variants")
    print("• Both achieve the same goal but organize frequencies differently")
    print("• Choose based on the model architecture you're implementing")
