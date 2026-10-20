"""Player-conditioned enemy cross-attention pooling.

This module encodes one player query and a set of enemy keys/values. It is
non-causal: masks exclude padding, not later enemy slots. Inputs should already
be normalized by the caller. No slot-position embeddings are used.

RecurrentAgent uses this through EnemyAttentionEncoder for new recurrent runs.
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


class EnemyAttentionEncoder(nn.Module):
    """Convert normalized 80-feature observations to 512-feature LSTM inputs.

    Each actor/critic owns its own instance. No information is added to the
    observation: absent coordinates have zero placeholders and a known flag.
    """

    def __init__(self):
        super().__init__()
        self.attention = CrossAttention(6, 7, dimensions=64, num_heads=4)
        # Match the old encoder's hidden width/depth after pooling. Player/map
        # and all 24 history/sensor features bypass pooling, preserving context.
        self.fusion = nn.Sequential(nn.Linear(94, 640), nn.ReLU(),
                                    nn.Linear(640, 640), nn.ReLU(), nn.Linear(640, 512))

    @staticmethod
    def unpack(observations):
        # Flatten only leading time/environment axes, preserving their order.
        flat = observations.reshape(-1, 80)
        player = flat[:, :6]
        enemies = flat[:, 6:46].reshape(-1, 10, 4)
        valid = enemies[:, :, 2] > 0  # Living-enemy HP, not coordinates.
        offsets = flat.new_zeros((flat.shape[0], 10, 2))
        offsets[:, :5] = flat[:, 46:56].reshape(-1, 5, 2)
        known = flat.new_zeros((flat.shape[0], 10, 1))
        known[:, :5, 0] = valid[:, :5].to(flat.dtype)
        # distance, direction, HP, MP, dx, dy, coordinates_known
        entities = torch.cat((enemies, offsets, known), dim=-1)
        entities = entities.masked_fill(torch.logical_not(valid[..., None]), 0)
        return player, entities, valid, flat[:, 56:80]

    def forward(self, observations):
        player, enemies, valid, history = self.unpack(observations)
        summary = self.attention(player, enemies, valid)
        fused = self.fusion(torch.cat((player, summary, history), dim=-1))
        return fused.reshape(*observations.shape[:-1], 512)


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
