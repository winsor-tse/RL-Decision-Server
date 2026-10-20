"""Optimized recurrence must preserve outputs, memory resets and gradients."""
from unittest.mock import patch
from itertools import product
import unittest
import torch

from Training.Mystic_Sim.recurrent import RecurrentAgent, RecurrentBranch
from Training.Mystic_Sim.benchmark import reference_forward


class RecurrentSpeedTests(unittest.TestCase):
    def setUp(self):
        # Test arithmetic equivalence in full FP32; cuDNN TF32 kernel selection
        # can otherwise change rounding between single-step and sequence calls.
        self.tf32=torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32=False

    def tearDown(self):
        torch.backends.cudnn.allow_tf32=self.tf32

    def test_chunked_outputs_and_gradients_match_reference(self):
        torch.set_num_threads(1)
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for head,device in product(('linear','tanh'),devices):
            for reset_mode in ('none','mixed','every'):
                torch.manual_seed(12)
                agent=RecurrentAgent([1.]*50,head).to(device)
                obs=torch.randn(9,3,50,requires_grad=True,device=device)
                starts=torch.zeros(9,3,dtype=torch.bool,device=device)
                if reset_mode=='mixed': starts[0,1]=True;starts[3,0]=True;starts[7,2]=True
                if reset_mode=='every': starts[:]=True
                state=tuple(torch.randn_like(s,requires_grad=True) for s in agent.initial_state(3))
                actual=agent.sequence(obs,state,starts)
                loss=actual[0].square().sum()+actual[1].square().sum()
                loss.backward()
                actual_grads=[p.grad.clone() for p in agent.parameters()]
                obs_grad=obs.grad.clone();state_grads=[s.grad.clone() for s in state]
                agent.zero_grad(set_to_none=True);obs.grad=None
                for s in state:s.grad=None
                with patch.object(RecurrentBranch,'forward',reference_forward):
                    expected=agent.sequence(obs,state,starts)
                    (expected[0].square().sum()+expected[1].square().sum()).backward()
                for a,b in zip((actual[0],actual[1],*actual[2],actual[3]),
                               (expected[0],expected[1],*expected[2],expected[3])):
                    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
                for a,b in zip(actual_grads,[p.grad for p in agent.parameters()]):
                    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(obs_grad,obs.grad,atol=2e-6,rtol=2e-5)
                for a,b in zip(state_grads,[s.grad for s in state]):
                    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_bootstrap_matches_full_forward_without_actor_or_state_mutation(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            agent=RecurrentAgent([1.]*50).to(device)
            obs=torch.randn(3,50,device=device)
            state=tuple(torch.randn_like(s) for s in agent.initial_state(3))
            saved=tuple(s.clone() for s in state)
            with torch.no_grad():
                expected=agent.sequence(obs.unsqueeze(0),state,torch.zeros(1,3,dtype=torch.bool,device=device))[1][0]
                with patch.object(agent.actor,'forward',side_effect=AssertionError('Unused actor evaluated')):
                    actual=agent.bootstrap_value(obs,state)
            torch.testing.assert_close(actual,expected)
            for a,b in zip(state,saved):torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
