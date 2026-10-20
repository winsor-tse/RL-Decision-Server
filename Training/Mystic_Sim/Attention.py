"""Player-conditioned enemy cross-attention pooling.

This module encodes one player query and a set of enemy keys/values. It is
non-causal: masks exclude padding, not later enemy slots. Inputs should already
be normalized by the caller. No slot-position embeddings are used.

This standalone module does not change RecurrentAgent/checkpoint architecture.
Enemy-to-enemy self-attention, if desired, is a separate preceding stage.
"""
import torch
from torch import nn


class CrossAttention(nn.Module):
    """Pool [B, N, Enemy_input] into [B, dimensions] using [B, Player_input].

    enemy_valid is bool [B, N], True for observed living enemies. If omitted,
    all slots are valid. return_weights=True also returns [B, num_heads, N].
    Features must be floating tensors on the model's device with matching dtype;
    the Boolean mask must be on that same device. PyTorch checks tensor operations.
    Empty sets produce exactly zero summaries and weights, with finite gradients.
    For PPO sequences, flatten time/environment axes into B before this module
    and restore them before the LSTM without changing chronological order.
    """

    def __init__(self, Player_input, Enemy_input, dimensions, num_heads=1):
        super().__init__()
        if num_heads < 1 or dimensions < 1 or dimensions % num_heads:
            raise ValueError("Use positive dimensions divisible by a positive num_heads")
        self.d_model = dimensions
        self.num_heads = num_heads
        self.head_dim = dimensions // num_heads
        # Shared across slots; these are layers, applied to tensors in forward.
        self.player_encoded = nn.Linear(Player_input, dimensions)
        self.enemy_encoded = nn.Linear(Enemy_input, dimensions)
        self.W_q = nn.Linear(dimensions, dimensions, bias=False)
        self.W_k = nn.Linear(dimensions, dimensions, bias=False)
        self.W_v = nn.Linear(dimensions, dimensions, bias=False)
        # Standard multi-head output projection: mixes the concatenated heads.
        # This is NOT an attention score; scores are Q @ K.T / sqrt(head_dim).
        self.W_o = nn.Linear(dimensions, dimensions, bias=False)

    def forward(self, player_features, enemy_features, enemy_valid=None, *, return_weights=False):
        batch, count, _ = enemy_features.shape
        if enemy_valid is None:
            enemy_valid = torch.ones((batch, count), dtype=torch.bool, device=enemy_features.device)
        # True means this slot is padding and should be ignored. logical_not is
        # the spelled-out equivalent of ~ for Boolean tensors.
        padding = torch.logical_not(enemy_valid)
        attention_padding = padding[:, None, None, :]  # [B, 1, 1, N]

        # 1. Embed the player and each enemy.
        # Zero padding BEFORE projection, so even NaNs in unused slots cannot
        # enter attention or its gradients. Valid inputs must still be finite.
        enemies = self.enemy_encoded(enemy_features.masked_fill(padding[..., None], 0))
        player = self.player_encoded(player_features)
        # 2. Project and split into heads: Q [B,H,1,d], K/V [B,H,N,d].
        q = self.W_q(player).reshape(batch, self.num_heads, 1, self.head_dim)
        k = self.W_k(enemies).reshape(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(enemies).reshape(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.masked_fill(padding[:, None, :, None], 0)
        # 3. Attention scores, then probabilities over enemies.
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(attention_padding, float("-inf"))
        # An all-masked row must not reach softmax as [-inf, ...]. For N=0,
        # softmax/matmul on the empty axis naturally yield a zero summary.
        has_enemy = enemy_valid.any(dim=-1)
        scores = torch.where(has_enemy[:, None, None, None], scores, torch.zeros_like(scores))
        weights = scores.softmax(dim=-1).masked_fill(attention_padding, 0)
        # 4. Weighted values per head, concatenate, then mix heads with W_o.
        # weights are [B,H,1,N]; summary is a feature vector, not probabilities.
        summary = (weights @ v).reshape(batch, self.d_model)
        summary = self.W_o(summary)
        if return_weights:
            return summary, weights.squeeze(2)
        return summary


# Integration notes:
# Current normalized observation: player/map [0:6], enemy blocks [6:46]
# reshaped to [..., 10, 4], five dx/dy pairs [46:56], history [56:80].
# Only the first five enemies have exact coordinates. Use coordinate-known
# flags if combining both formats; never infer exact dx/dy from coarse direction.
# Enemy HP fraction > 0 identifies valid slots in this simulator.
# Concatenate the summary with player/history/terrain information before LSTM.
# Keep actor/critic encoders separate if preserving independent gradient clipping.
# Policy wiring requires versioned architecture metadata and matching train,
# evaluation and viewer construction; old flat-encoder weights cannot load as-is.


def demo():
    """Run with: python -m Training.Mystic_Sim.Attention (no training)."""
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)
    # Small illustrative features, not the full simulator observation contract.
    player = torch.tensor([[0.75, 0.50]])  # HP fraction, MP fraction
    enemies = torch.tensor([[
        [0.10, 0.00, 0.20],   # Enemy A: dx/99, dy/99, HP fraction
        [-0.05, 0.10, 0.90],  # Enemy B
        [0.00, -0.20, 0.50],  # Enemy C
        [0.00, 0.00, 0.00],   # Padding, not an enemy
    ]])
    valid = torch.tensor([[True, True, True, False]])
    model = CrossAttention(Player_input=2, Enemy_input=3, dimensions=8, num_heads=2)
    model.eval()
    with torch.no_grad():
        summary, weights = model(player, enemies, valid, return_weights=True)
        empty_summary = model(player, enemies, torch.zeros_like(valid))

    print('Random, untrained weights; seed 42. No game strategy learned yet.')
    print('\nPlayer [HP fraction, MP fraction]:', player)
    print('\nEnemies [dx/99, dy/99, HP fraction]; last row is padding:\n', enemies[0])
    print('\nD=8, H=2, features per head=4')
    print('Q: [1, 2, 1, 4]; K/V: [1, 2, 4, 4]')
    print('\nAttention weights: rows=heads, columns=A/B/C/padding:\n', weights[0])
    print('Sum per head:', weights[0].sum(dim=-1))
    print('\nFinal summary after concatenation and W_o:', summary)
    print('Summary shape:', tuple(summary.shape))
    print('Summary features are learned numbers, not HP values or action probabilities.')
    print('\nAll enemies masked -> zero summary:', empty_summary)


if __name__ == '__main__':
    demo()
