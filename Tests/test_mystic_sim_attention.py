"""Cross-attention math, padding safety and permutation invariance."""
import unittest

import torch

from Training.Mystic_Sim.Attention import CrossAttention


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_numeric_attention_and_player_conditioning(self):
        model = CrossAttention(2, 2, 2)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.copy_(torch.eye(2))
                    if module.bias is not None:
                        module.bias.zero_()
        enemies = torch.eye(2).unsqueeze(0)
        result, weights = model(torch.tensor([[2., 0.]]), enemies, return_weights=True)
        expected = torch.softmax(torch.tensor([2. / 2**.5, 0.]), dim=0)
        torch.testing.assert_close(weights[0, 0], expected)
        torch.testing.assert_close(result[0], expected)
        other = model(torch.tensor([[0., 2.]]), enemies)
        torch.testing.assert_close(other[0], expected.flip(0))

    def test_multihead_matches_pytorch_reference(self):
        model = CrossAttention(3, 5, 8, num_heads=2)
        p, e = torch.randn(2, 3), torch.randn(2, 4, 5)
        valid = torch.tensor([[True, False, True, True], [True, True, False, False]])
        reference = torch.nn.MultiheadAttention(8, 2, bias=False, batch_first=True)
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([model.W_q.weight, model.W_k.weight, model.W_v.weight]))
            reference.out_proj.weight.copy_(model.W_o.weight)
        expected, weights = reference(model.player_encoded(p).unsqueeze(1),
            model.enemy_encoded(e), model.enemy_encoded(e), key_padding_mask=~valid,
            average_attn_weights=False)
        actual, actual_weights = model(p, e, valid, return_weights=True)
        torch.testing.assert_close(actual, expected.squeeze(1))
        torch.testing.assert_close(actual_weights, weights.squeeze(2))

    def test_padding_permutation_and_gradients(self):
        model = CrossAttention(3, 4, 8, num_heads=2)
        p = torch.randn(2, 3, requires_grad=True)
        e = torch.randn(2, 5, 4, requires_grad=True)
        valid = torch.tensor([[True, False, True, False, True], [False]*5])
        output, weights = model(p, e, valid, return_weights=True)
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertEqual(tuple(weights.shape), (2, 2, 5))
        torch.testing.assert_close(weights[0].sum(-1), torch.ones(2))
        self.assertFalse(weights[1].any())
        self.assertFalse(output[1].any())
        contaminated = e.detach().masked_fill(~valid[..., None], float('nan'))
        torch.testing.assert_close(model(p, contaminated, valid), output)
        perm = torch.tensor([4, 0, 3, 1, 2])
        torch.testing.assert_close(model(p, e[:, perm], valid[:, perm]), output)
        output.square().sum().backward()
        self.assertFalse(e.grad[~valid].any())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.player_encoded.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.enemy_encoded.weight.grad.abs().sum().item(), 0)

    def test_empty_sets_batch_one_and_validation(self):
        model = CrossAttention(3, 4, 8, num_heads=2)
        for count in (0, 3):
            model.zero_grad(set_to_none=True)
            p, e = torch.randn(1, 3), torch.randn(1, count, 4)
            out, weights = model(p, e, torch.zeros(1, count, dtype=torch.bool), return_weights=True)
            self.assertEqual(tuple(out.shape), (1, 8))
            self.assertEqual(tuple(weights.shape), (1, 2, count))
            self.assertFalse(out.any())
            out.sum().backward()
            for param in model.parameters():
                self.assertTrue(torch.isfinite(param.grad).all())
        with self.assertRaises(ValueError):
            CrossAttention(3, 4, 7, num_heads=2)
        with self.assertRaises(RuntimeError):
            model(torch.zeros(1, 2), torch.zeros(1, 3, 4))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_parity(self):
        cpu = CrossAttention(3, 4, 8, num_heads=2)
        gpu = CrossAttention(3, 4, 8, num_heads=2).cuda()
        gpu.load_state_dict(cpu.state_dict())
        p, e = torch.randn(2, 3), torch.randn(2, 5, 4)
        valid = torch.tensor([[True, False, True, True, False], [False]*5])
        expected = cpu(p, e, valid)
        actual = gpu(p.cuda(), e.cuda(), valid.cuda())
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-4)
        actual.square().sum().backward()
        self.assertTrue(all(torch.isfinite(param.grad).all() for param in gpu.parameters()))
