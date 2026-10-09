"""Train standalone Mystic Sim PPO: python -m Training.Mystic_Sim.train --help."""
import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from Custom_enviornments.Mystic_Sim.actions import ACTIONS
from Custom_enviornments.Mystic_Sim.rewards import COMPONENTS
from Utils.ppo_checkpoint import atomic_torch_save, capture_rng_state, restore_rng_state
from .core import (Agent, PPOConfig, SimBatch, accumulate, advantages, checkpoint_updates,
                   episode_metrics, finish_metrics, json_write, load_checkpoint, make_env, metadata)
from .evaluate import evaluate


def critic_loss(value, old_value, returns, clip_coef=None):
    error = (value-returns).square()
    if clip_coef is not None:
        clipped = old_value + (value-old_value).clamp(-clip_coef, clip_coef)
        error = torch.maximum(error, (clipped-returns).square())
    return .5 * error.mean()


def clip_gradients(agent, config):
    # Independent limits prevent large critic gradients from scaling actor gradients.
    actor = nn.utils.clip_grad_norm_(agent.actor.parameters(), config.actor_max_grad_norm, error_if_nonfinite=True)
    critic = nn.utils.clip_grad_norm_(agent.critic.parameters(), config.critic_max_grad_norm, error_if_nonfinite=True)
    return actor, critic


def optimize(agent, optimizer, config, observations, actions, old_logprobs, values, adv, returns):
    observations = observations.flatten(0, 1)
    actions, old_logprobs, values, adv, returns = [x.flatten() for x in (actions, old_logprobs, values, adv, returns)]
    indices = np.arange(config.batch_size)
    minibatch = config.batch_size // config.num_minibatches
    rows = []
    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        epoch_kl = []
        for start in range(0, config.batch_size, minibatch):
            ix = torch.as_tensor(indices[start:start+minibatch], device=observations.device)
            _, logprob, entropy, value = agent.action_value(observations[ix], actions[ix])
            logratio = logprob-old_logprobs[ix]
            ratio = logratio.exp()
            advantage = adv[ix]
            advantage = (advantage-advantage.mean()) / (advantage.std(unbiased=False)+1e-8)
            policy_loss = torch.maximum(-advantage*ratio, -advantage*ratio.clamp(1-config.clip_coef, 1+config.clip_coef)).mean()
            value_loss = critic_loss(value, values[ix], returns[ix], config.value_clip_coef)
            entropy = entropy.mean()
            loss = policy_loss + config.vf_coef*value_loss - config.ent_coef*entropy
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_grad_norm, critic_grad_norm = clip_gradients(agent, config)
            optimizer.step()
            with torch.no_grad():
                kl = ((ratio-1)-logratio).mean()
                clipfrac = ((ratio-1).abs() > config.clip_coef).float().mean()
                rows.append([float(x) for x in (policy_loss, value_loss, entropy, kl, clipfrac,
                                                actor_grad_norm, critic_grad_norm)])
                epoch_kl.append(float(kl))
        if config.target_kl and np.mean(epoch_kl) > config.target_kl:
            break
    result = dict(zip(('policy_loss','value_loss','entropy','approx_kl','clipfrac',
                       'actor_grad_norm','critic_grad_norm'), np.mean(rows, axis=0).tolist()))
    variance = returns.var(unbiased=False)
    result['explained_variance'] = float(1-(returns-values).var(unbiased=False)/variance) if variance > 0 else 0.
    with torch.no_grad():
        predictions = agent.value(observations)
        hidden = agent.critic[:4](agent.normalized(observations))
        result['critic_value_std'] = float(predictions.std(unbiased=False))
        result['critic_target_std'] = float(returns.std(unbiased=False))
        result['critic_hidden_saturation'] = float((hidden.abs() > .99).float().mean())
        result['post_update_explained_variance'] = float(1-(returns-predictions).var(unbiased=False)/variance) if variance > 0 else 0.
    return result


def log_episode(writer, row, step, episodes, wins):
    for key in ('episode_return', 'length', 'kills', 'survival_seconds', 'damage_dealt',
                'damage_taken', 'invalid_cast_rate', 'collisions', 'cooldown_rejections'):
        writer.add_scalar(f'episodes/{key}', row[key], step)
    writer.add_scalar('episodes/win_rate', wins / episodes, step)
    for key, value in row['components'].items():
        writer.add_scalar(f'rewards/episode_{key}', value, step)


def train(config, *, run_dir=None, device='auto', resume=None, record_gameplay=True,
          record_stride=5, stop_after_checkpoint=None):
    if device not in ('auto','cpu','cuda'): raise ValueError('device must be auto, cpu, or cuda')
    if record_stride < 1: raise ValueError('record_stride must be positive')
    if stop_after_checkpoint is not None and not 1 <= stop_after_checkpoint <= 10:
        raise ValueError('stop_after_checkpoint must be in 1..10')
    resolved_device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
    if resolved_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu or install a CUDA-enabled PyTorch')
    probe = make_env()
    try:
        contract = metadata(probe)
        observation_high = probe.observation_space.high.copy()
    finally:
        probe.close()
    payload = load_checkpoint(resume, contract) if resume else None
    if payload:
        config = PPOConfig(**payload['ppo_config'])
    config.validate()
    schedule = checkpoint_updates(config.updates)
    completed = payload['update'] if payload else 0
    if payload and (completed not in schedule or payload['global_step'] != completed*config.batch_size):
        raise ValueError('Invalid checkpoint progress')
    if completed >= config.updates: raise ValueError('This checkpoint has already finished its training budget')
    if stop_after_checkpoint and completed and stop_after_checkpoint <= schedule[completed]:
        raise ValueError('Stop checkpoint must be after the resumed checkpoint')
    if run_dir is None:
        run_dir = Path(resume).resolve().parent.parent if resume else Path('runs/mystic_sim') / (
            datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + f'_seed{config.seed}_{uuid.uuid4().hex[:6]}')
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        if not resume or run_dir != Path(resume).resolve().parent.parent:
            raise ValueError('Use an empty run directory for a new run or resume branch')
        if any((run_dir/'checkpoints'/f'checkpoint_{i:02d}.pt').exists() for i in range(schedule[completed]+1,11)):
            raise ValueError('Later checkpoints already exist; resume into a new --run-dir to preserve them')
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    agent = Agent(observation_high).to(resolved_device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    episodes, wins = (payload['episodes'],payload['wins']) if payload else (0,0)
    if payload:
        agent.load_state_dict(payload['agent'])
        optimizer.load_state_dict(payload['optimizer'])
        restore_rng_state(payload['rng'])
    run_metadata = {'metadata': contract, 'ppo_config': asdict(config), 'device': resolved_device,
                    'requested_timesteps': config.total_timesteps,
                    'actual_timesteps': config.updates*config.batch_size,
                    'checkpoint_steps': {str(i): u*config.batch_size for u,i in schedule.items()},
                    'resume_from': str(Path(resume).resolve()) if resume else None,
                    'record_gameplay': record_gameplay, 'record_stride': record_stride,
                    'torch': str(torch.__version__), 'batch_backend': 'synchronous_independent_worlds'}
    # Preserve the original manifest across continuation; each invocation records its settings.
    if not (run_dir/'run.json').exists(): json_write(run_dir/'run.json', run_metadata)
    json_write(run_dir/f'invocation_after_update_{completed:06d}.json', run_metadata)
    print(f'Run: {run_dir}\nDevice: {resolved_device}; envs: {config.num_envs}; '
          f'rollout: {config.batch_size}; updates: {config.updates}; ten checkpoints', flush=True)
    writer = SummaryWriter(str(run_dir/'tensorboard'), purge_step=completed*config.batch_size+1 if payload else None)
    batch = None
    start_time, training_seconds = time.perf_counter(), 0.
    invocation_start_step = completed*config.batch_size
    global_step = invocation_start_step
    try:
        writer.add_text('run/config', json.dumps(run_metadata, indent=2))
        seeds = list(range(config.eval_seed, config.eval_seed+config.eval_episodes))
        baseline_path = run_dir/'random_baseline.json'
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        else:
            baseline = evaluate(None, seeds)
            json_write(baseline_path, baseline)
        for key,value in baseline['summary'].items():
            if key != 'components': writer.add_scalar(f'baseline/{key}', value, global_step)
        # Fresh deterministic episode seeds on resume; simulator world snapshots are not claimed.
        batch = SimBatch(config.num_envs, config.seed + completed*config.num_envs)
        running = [episode_metrics() for _ in batch.envs]
        shape = (config.num_steps,config.num_envs)
        obs_buffer = torch.empty((*shape,26),device=resolved_device)
        actions = torch.empty(shape,dtype=torch.long,device=resolved_device)
        logprobs, values, rewards, next_values = [torch.empty(shape,device=resolved_device) for _ in range(4)]
        terminated, truncated = [torch.empty(shape,dtype=torch.bool,device=resolved_device) for _ in range(2)]
        for update in range(completed+1, config.updates+1):
            update_start = time.perf_counter()
            lr = config.learning_rate*(1-(update-1)/config.updates) if config.anneal_lr else config.learning_rate
            optimizer.param_groups[0]['lr'] = lr
            component_sums = dict.fromkeys(COMPONENTS,0.)
            action_counts = np.zeros(8,dtype=np.int64)
            cast_count = invalid_count = collision_count = cooldown_count = 0
            before = nn.utils.parameters_to_vector(agent.parameters()).detach().clone()
            agent.train()
            for t in range(config.num_steps):
                observation = torch.as_tensor(batch.observations,device=resolved_device)
                obs_buffer[t] = observation
                with torch.no_grad():
                    action, logprob, _, value = agent.action_value(observation)
                actions[t], logprobs[t], values[t] = action,logprob,value
                cpu_actions = action.cpu().numpy()
                final_obs, reward, term, trunc, infos = batch.step(cpu_actions)
                rewards[t] = torch.as_tensor(reward,device=resolved_device)
                terminated[t], truncated[t] = torch.as_tensor(term,device=resolved_device),torch.as_tensor(trunc,device=resolved_device)
                with torch.no_grad():
                    next_values[t] = agent.value(torch.as_tensor(final_obs,device=resolved_device))
                global_step += config.num_envs
                action_counts += np.bincount(cpu_actions,minlength=8)
                for i,info in enumerate(infos):
                    accumulate(running[i],int(cpu_actions[i]),reward[i],info)
                    for key in COMPONENTS: component_sums[key] += info['reward_components'][key]
                    cast_count += int(cpu_actions[i]>=5)
                    invalid_count += int(cpu_actions[i]>=5 and not info['action_applied'])
                    collision_count += int(info['collision_kind'] is not None)
                    cooldown_count += int(info['cooldown_penalty']<0)
                    if term[i] or trunc[i]:
                        row = finish_metrics(running[i],info)
                        episodes += 1
                        wins += int(row['win'])
                        log_episode(writer,row,global_step,episodes,wins)
                        running[i] = episode_metrics()
            with torch.no_grad():
                # Only learning targets use scaled rewards. Environment, episode,
                # evaluation and reward-component metrics remain in original units.
                adv, returns = advantages(rewards * config.reward_scale,values,next_values,
                                          terminated,truncated,config.gamma,config.gae_lambda)
            if not torch.isfinite(returns).all(): raise FloatingPointError('Nonfinite PPO targets')
            losses = optimize(agent,optimizer,config,obs_buffer,actions,logprobs,values,adv,returns)
            parameter_delta = float((nn.utils.parameters_to_vector(agent.parameters()).detach()-before).norm())
            if not math.isfinite(parameter_delta): raise FloatingPointError('Nonfinite PPO parameters')
            training_seconds += time.perf_counter()-update_start
            for key,value in losses.items(): writer.add_scalar(f'losses/{key}',value,global_step)
            writer.add_scalar('charts/parameter_delta',parameter_delta,global_step)
            writer.add_scalar('charts/learning_rate',lr,global_step)
            writer.add_scalar('charts/SPS_training',(global_step-invocation_start_step)/training_seconds,global_step)
            writer.add_scalar('charts/SPS_wall',(global_step-invocation_start_step)/(time.perf_counter()-start_time),global_step)
            writer.add_scalar('charts/mean_step_reward',float(rewards.mean()),global_step)
            writer.add_scalar('charts/mean_scaled_step_reward',float(rewards.mean())*config.reward_scale,global_step)
            for key,value in component_sums.items(): writer.add_scalar(f'rewards/{key}',value/config.batch_size,global_step)
            for i,label in enumerate(ACTIONS): writer.add_scalar(f'actions/{label.replace(":","_")}',action_counts[i]/config.batch_size,global_step)
            writer.add_scalar('environment/invalid_cast_rate',invalid_count/max(1,cast_count),global_step)
            writer.add_scalar('environment/collision_rate',collision_count/config.batch_size,global_step)
            writer.add_scalar('environment/cooldown_rejection_rate',cooldown_count/config.batch_size,global_step)
            writer.add_scalar('environment/player_hp',float(obs_buffer[:,:,3].mean()),global_step)
            writer.add_scalar('environment/player_mp',float(obs_buffer[:,:,4].mean()),global_step)
            if update in schedule:
                index = schedule[update]
                checkpoint = run_dir/'checkpoints'/f'checkpoint_{index:02d}.pt'
                atomic_torch_save({'format':'mystic_sim_ppo_v2','metadata':contract,
                    'ppo_config':asdict(config),'agent':agent.state_dict(),'optimizer':optimizer.state_dict(),
                    'update':update,'global_step':global_step,'rng':capture_rng_state(),
                    'episodes':episodes,'wins':wins},checkpoint)
                agent.eval()
                recording = run_dir/'gameplay'/f'checkpoint_{index:02d}.gif' if record_gameplay else None
                report = evaluate(agent,seeds,device=resolved_device,record_path=recording,record_stride=record_stride)
                report.update(checkpoint=str(checkpoint),global_step=global_step,
                              return_vs_random=report['summary']['episode_return']-baseline['summary']['episode_return'])
                json_write(run_dir/'evaluation'/f'checkpoint_{index:02d}.json',report)
                for key,value in report['summary'].items():
                    if key != 'components': writer.add_scalar(f'evaluation/{key}',value,global_step)
                for key,value in report['summary']['components'].items(): writer.add_scalar(f'evaluation/rewards/{key}',value,global_step)
                writer.add_scalar('evaluation/return_vs_random',report['return_vs_random'],global_step)
                if report['recording_error']:
                    print('Recording unavailable:',report['recording_error'],flush=True)
                    writer.add_text('recording/error',report['recording_error'],global_step)
                json_write(run_dir/'progress.json',{'checkpoint':str(checkpoint),'global_step':global_step,
                    'update':update,'complete':update==config.updates,'losses':losses,
                    'parameter_delta':parameter_delta,'episodes':episodes,'wins':wins,
                    'training_sps':(global_step-invocation_start_step)/training_seconds,
                    'wall_sps':(global_step-invocation_start_step)/(time.perf_counter()-start_time)})
                writer.flush()
                print(f'Checkpoint {index}/10: step={global_step}, eval_return={report["summary"]["episode_return"]:.2f}, '
                      f'kills={report["summary"]["kills"]:.2f}, SPS={(global_step-invocation_start_step)/training_seconds:.0f}',flush=True)
                if stop_after_checkpoint == index: break
        return run_dir
    finally:
        if batch: batch.close()
        writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = PPOConfig()
    for field in fields(defaults):
        default = getattr(defaults,field.name)
        options = {'action':argparse.BooleanOptionalAction} if isinstance(default,bool) else {
            'type':float if field.name == 'value_clip_coef' else type(default)}
        parser.add_argument('--'+field.name.replace('_','-'),default=default,**options)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--resume',type=Path,help='Restore saved hyperparameters/optimizer/RNG; fresh simulator episodes')
    parser.add_argument('--record-gameplay',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--record-stride',type=int,default=5,help='Record every N decisions; 5 = one frame per simulated second')
    parser.add_argument('--torch-threads',type=int,default=1)
    parser.add_argument('--stop-after-checkpoint',type=int,help='Stop cleanly after checkpoint 1..10; resume later')
    args = parser.parse_args(argv)
    if args.torch_threads < 1: parser.error('--torch-threads must be positive')
    torch.set_num_threads(args.torch_threads)
    config = PPOConfig(**{field.name:getattr(args,field.name) for field in fields(defaults)})
    train(config,run_dir=args.run_dir,device=args.device,resume=args.resume,
          record_gameplay=args.record_gameplay,record_stride=args.record_stride,
          stop_after_checkpoint=args.stop_after_checkpoint)


if __name__ == '__main__':
    main()
