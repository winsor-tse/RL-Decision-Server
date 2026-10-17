"""Short repeatable SPS/kernel benchmarks. No optimizer updates or training runs."""
import argparse
from contextlib import nullcontext
import json
import time
from unittest.mock import patch

import numpy as np
import torch

from .core import SimBatch, make_env, metadata
from .recurrent import RecurrentAgent, RecurrentBranch


def reference_forward(self, observations, state, starts):
    """Pre-optimization implementation retained only for parity/benchmark tests."""
    embedded = self.network(observations)
    outputs = []
    for embedding, start in zip(embedded, starts):
        mask = (~start.bool()).to(embedding.dtype).view(1,-1,1)
        output,state = self.lstm(embedding.unsqueeze(0),tuple(x*mask for x in state))
        outputs.append(output)
    if self.head_type=='linear':
        hidden=torch.cat(outputs)
        return self.head(hidden),state,hidden
    hidden=self.head[:4](torch.cat(outputs))
    return self.head[4](hidden),state,hidden


def synchronize(device):
    if device=='cuda':
        torch.cuda.synchronize()


def benchmark(device='cpu',num_envs=4,steps=256,iterations=10,compare_reference=False):
    env=make_env()
    try:
        high=metadata(env,recurrent=True)['observation_scale']
    finally:
        env.close()
    actions=np.random.default_rng(123).integers(0,7,size=(steps,num_envs))
    results={'device':device,'num_envs':num_envs,'transitions':steps*num_envs,
             'optimizer_updates':0,'scripted_actions':True}
    batch=SimBatch(num_envs,42,recurrent=True)
    try:
        start=time.perf_counter()
        for action in actions: batch.step(action)
        results['environment_sps']=steps*num_envs/(time.perf_counter()-start)
    finally:
        batch.close()
    for name,reference in ([('reference',True)] if compare_reference else [])+[('optimized',False)]:
        context=patch.object(RecurrentBranch,'forward',reference_forward) if reference else nullcontext()
        with context:
            torch.manual_seed(42)
            agent=RecurrentAgent(high).to(device)
            x=torch.rand(64,num_envs,len(high),device=device)
            starts=torch.zeros(64,num_envs,dtype=torch.bool,device=device)
            starts[20,0]=True;starts[45,-1]=True
            # Warm up backend kernels; compute gradients but never update weights.
            for i in range(iterations+2):
                if i==2:
                    synchronize(device);start=time.perf_counter()
                agent.zero_grad(set_to_none=True)
                logits,value,_,_=agent.sequence(x,agent.initial_state(num_envs),starts)
                (logits.square().mean()+value.square().mean()).backward()
            synchronize(device)
            kernel_ms=(time.perf_counter()-start)*1000/iterations
            batch=SimBatch(num_envs,42,recurrent=True)
            try:
                state=agent.initial_state(num_envs)
                episode_starts=torch.ones(num_envs,dtype=torch.bool,device=device)
                synchronize(device);start=time.perf_counter()
                with torch.no_grad():
                    for action in actions:
                        obs=torch.as_tensor(batch.observations,device=device)
                        _,_,_,_,state=agent.step(obs,state,episode_starts,
                                                action=torch.as_tensor(action,device=device))
                        final,_,terminated,truncated,_=batch.step(action)
                        final=torch.as_tensor(final,device=device)
                        if reference:
                            agent.sequence(final.unsqueeze(0),state,
                                           torch.zeros((1,num_envs),dtype=torch.bool,device=device))
                        else:
                            agent.bootstrap_value(final,state)
                        episode_starts=torch.as_tensor(terminated|truncated,device=device)
                synchronize(device)
                results[name]={'rollout_sps':steps*num_envs/(time.perf_counter()-start),
                               'recurrent_forward_backward_ms':kernel_ms}
            finally:
                batch.close()
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--num-envs',type=int,default=4)
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--iterations',type=int,default=10)
    parser.add_argument('--torch-threads',type=int,default=1)
    parser.add_argument('--compare-reference',action='store_true')
    args=parser.parse_args(argv)
    if min(args.num_envs,args.steps,args.iterations,args.torch_threads)<1:
        parser.error('Counts must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    result=benchmark(args.device,args.num_envs,args.steps,args.iterations,args.compare_reference)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    main()
