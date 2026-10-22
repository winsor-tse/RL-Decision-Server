"""Train standalone Mystic Sim PPO: python -m Training.Mystic_Sim.train --help."""
import argparse
from collections import Counter
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
from .evaluate import evaluate, evaluation_line
from .recurrent import RecurrentAgent, optimize_recurrent
from .core import saved_config
from .mixed_scenarios import MixedEncounter
from .encounter_diagnostics import METRICS as ENCOUNTER_METRICS
from .curriculum import (encounter_factory, encounter_from_dict, load_stage_checkpoint,
                         stage_schedule, learning_rate)


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
    for key in ENCOUNTER_METRICS:
        if key in row:
            writer.add_scalar(f'combat_navigation/{key}', row[key], step)
    for key in ('episode_return', 'length', 'kills', 'survival_seconds', 'damage_dealt',
                'damage_taken', 'invalid_cast_rate', 'collisions', 'cooldown_rejections'):
        writer.add_scalar(f'episodes/{key}', row[key], step)
    writer.add_scalar('episodes/win_rate', wins / episodes, step)
    for key, value in row['components'].items():
        writer.add_scalar(f'rewards/episode_{key}', value, step)


def checkpoint_recordings(run_dir, index, enabled, first_checkpoint):
    if not enabled or index < first_checkpoint:
        return None, None
    return (run_dir/'gameplay'/f'checkpoint_{index:02d}.gif',
            run_dir/'gameplay'/f'checkpoint_{index:02d}_sampled.gif')


def full_evaluations(agent, seeds, *, enabled, encounter, device, stage_reports):
    if not enabled:
        return {}
    return {mode: (evaluate(agent,seeds,device=device,stochastic=mode=='sampled',factory=make_env)
                   if encounter is not None else dict(report))
            for mode,report in stage_reports.items()}


def train(config, *, run_dir=None, device='auto', resume=None, record_gameplay=True,
          record_stride=5, stop_after_checkpoint=None, encounter=None, continue_from=None,
          schedule_timesteps=None, record_from_checkpoint=1, evaluate_full_scenario=True):
    if resume and continue_from:
        raise ValueError('Choose resume or continue_from, not both')
    if device not in ('auto','cpu','cuda'): raise ValueError('device must be auto, cpu, or cuda')
    if record_stride < 1: raise ValueError('record_stride must be positive')
    if type(record_from_checkpoint) is not int or not 1 <= record_from_checkpoint <= 10:
        raise ValueError('record_from_checkpoint must be in 1..10')
    if type(evaluate_full_scenario) is not bool:
        raise ValueError('evaluate_full_scenario must be boolean')
    if stop_after_checkpoint is not None and not 1 <= stop_after_checkpoint <= 10:
        raise ValueError('stop_after_checkpoint must be in 1..10')
    resolved_device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
    if resolved_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu or install a CUDA-enabled PyTorch')
    if resume:
        raw = torch.load(resume, map_location='cpu', weights_only=True)
        config = saved_config(raw)
        saved_encounter = encounter_from_dict(raw['metadata']['scenario'].get('encounter'))
        if encounter is not None and encounter != saved_encounter:
            raise ValueError('Resume cannot change encounter; use stage continuation in a new run')
        encounter = saved_encounter
    factory = make_env if encounter is None else encounter_factory(encounter)
    choices = encounter.mixture if isinstance(encounter, MixedEncounter) else ()
    probe = factory()
    try:
        contract = metadata(probe, recurrent=config.recurrent, actor_head=config.actor_head, recurrent_encoder=config.recurrent_encoder)
        observation_high = np.asarray(contract['observation_scale'], dtype=np.float32)
    finally:
        probe.close()
    payload = load_checkpoint(resume, contract) if resume else None
    if payload:
        config = saved_config(payload)
    config.validate()
    source_payload = load_stage_checkpoint(continue_from, contract, config) if continue_from else None
    curriculum = stage_schedule(config, resume_payload=payload, source_payload=source_payload,
                                schedule_timesteps=schedule_timesteps)
    if continue_from:
        curriculum['source_checkpoint'] = str(Path(continue_from).resolve())
    if choices:
        curriculum['mode'] = 'fixed_episode_mixture'
        curriculum['episode_probabilities'] = {c.name: c.weight / sum(x.weight for x in choices) for c in choices}
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
    agent = (RecurrentAgent(observation_high, config.actor_head, config.recurrent_encoder) if config.recurrent
             else Agent(observation_high)).to(resolved_device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    episodes, wins = (payload['episodes'],payload['wins']) if payload else (0,0)
    restored = payload if payload is not None else source_payload
    if restored:
        # Preserve actor, critic, both LSTMs and Adam moments. Episode memory is
        # initialized below for fresh worlds; no learned parameter is wiped.
        agent.load_state_dict(restored['agent'])
        optimizer.load_state_dict(restored['optimizer'])
        restore_rng_state(restored['rng'])
    run_metadata = {'metadata': contract, 'ppo_config': asdict(config), 'device': resolved_device,
                    'requested_timesteps': config.total_timesteps,
                    'actual_timesteps': config.updates*config.batch_size,
                    'checkpoint_steps': {str(i): u*config.batch_size for u,i in schedule.items()},
                    'resume_from': str(Path(resume).resolve()) if resume else None,
                    'record_gameplay': record_gameplay, 'record_stride': record_stride,
                    'record_from_checkpoint': record_from_checkpoint,
                    'evaluate_full_scenario': evaluate_full_scenario,
                    'curriculum': curriculum,
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
            baseline = evaluate(None, seeds, factory=factory)
            json_write(baseline_path, baseline)
        for key,value in baseline['summary'].items():
            if key != 'components': writer.add_scalar(f'baseline/{key}', value, global_step)
        # Fresh deterministic episode seeds on resume; simulator world snapshots are not claimed.
        batch = SimBatch(config.num_envs, config.seed + completed*config.num_envs,
                         factory=factory, recurrent=config.recurrent,
                         sampling_states=payload.get('scenario_sampler_states') if payload and choices else None)
        running = [episode_metrics() for _ in batch.envs]
        shape = (config.num_steps,config.num_envs)
        obs_buffer = torch.empty((*shape,len(observation_high)),device=resolved_device)
        actions = torch.empty(shape,dtype=torch.long,device=resolved_device)
        logprobs, values, rewards, next_values = [torch.empty(shape,device=resolved_device) for _ in range(4)]
        terminated, truncated = [torch.empty(shape,dtype=torch.bool,device=resolved_device) for _ in range(2)]
        if config.recurrent:
            recurrent_state = agent.initial_state(config.num_envs)
            episode_starts = torch.ones(config.num_envs, dtype=torch.bool, device=resolved_device)
            starts_buffer = torch.empty(shape, dtype=torch.bool, device=resolved_device)
            state_buffers = tuple(torch.empty((config.num_steps, *s.shape), device=resolved_device) for s in recurrent_state)
        for update in range(completed+1, config.updates+1):
            update_start = time.perf_counter()
            lr = learning_rate(config, update, curriculum)
            optimizer.param_groups[0]['lr'] = lr
            component_sums = dict.fromkeys(COMPONENTS,0.)
            action_counts = np.zeros(len(ACTIONS),dtype=np.int64)
            cast_count = invalid_count = collision_count = cooldown_count = 0
            failure_counts = Counter()
            successful_moves = 0
            scenario_steps = Counter()
            scenario_episodes = Counter()
            before = nn.utils.parameters_to_vector(agent.parameters()).detach().clone()
            agent.train()
            for t in range(config.num_steps):
                observation = torch.as_tensor(batch.observations,device=resolved_device)
                obs_buffer[t] = observation
                with torch.no_grad():
                    if config.recurrent:
                        starts_buffer[t] = episode_starts
                        for buffer, state in zip(state_buffers, recurrent_state):
                            buffer[t] = state
                        action, logprob, _, value, recurrent_state = agent.step(observation, recurrent_state, episode_starts)
                    else:
                        action, logprob, _, value = agent.action_value(observation)
                actions[t], logprobs[t], values[t] = action,logprob,value
                cpu_actions = action.cpu().numpy()
                final_obs, reward, term, trunc, infos = batch.step(cpu_actions)
                rewards[t] = torch.as_tensor(reward,device=resolved_device)
                terminated[t], truncated[t] = torch.as_tensor(term,device=resolved_device),torch.as_tensor(trunc,device=resolved_device)
                with torch.no_grad():
                    if config.recurrent:
                        # Bootstrap from FINAL state with pre-reset memory. Do not
                        # commit this lookahead state: next rollout step consumes
                        # that observation once, or masks memory after autoreset.
                        next_values[t] = agent.bootstrap_value(
                            torch.as_tensor(final_obs,device=resolved_device),recurrent_state)
                        episode_starts = terminated[t] | truncated[t]
                    else:
                        next_values[t] = agent.value(torch.as_tensor(final_obs,device=resolved_device))
                global_step += config.num_envs
                action_counts += np.bincount(cpu_actions,minlength=len(ACTIONS))
                for i,info in enumerate(infos):
                    scenario_name = info.get('training_scenario')
                    if scenario_name:
                        scenario_steps[scenario_name] += 1
                    accumulate(running[i],int(cpu_actions[i]),reward[i],info)
                    for key in COMPONENTS: component_sums[key] += info['reward_components'][key]
                    cast_count += int(cpu_actions[i]>=4)
                    invalid_count += int(cpu_actions[i]>=4 and not info['action_applied'])
                    collision_count += int(info['collision_kind'] is not None)
                    cooldown_count += int(info['cooldown_penalty']<0)
                    successful_moves += int(cpu_actions[i] < 4 and info['action_applied'])
                    if not info['action_applied']:
                        failure_counts[f"{ACTIONS[cpu_actions[i]].replace(':', '_')}/{info['action_failure_reason'] or 'unknown'}"] += 1
                    if term[i] or trunc[i]:
                        row = finish_metrics(running[i],info)
                        episodes += 1
                        wins += int(row['win'])
                        log_episode(writer,row,global_step,episodes,wins)
                        if scenario_name:
                            scenario_episodes[scenario_name] += 1
                            for metric in ('win', 'episode_return', 'kills', 'length'):
                                writer.add_scalar(f'training_scenarios/{scenario_name}/{metric}', row[metric], global_step)
                        running[i] = episode_metrics()
            rollout_seconds = time.perf_counter()-update_start
            optimization_start = time.perf_counter()
            with torch.no_grad():
                # Only learning targets use scaled rewards. Environment, episode,
                # evaluation and reward-component metrics remain in original units.
                adv, returns = advantages(rewards * config.reward_scale,values,next_values,
                                          terminated,truncated,config.gamma,config.gae_lambda)
            if not torch.isfinite(returns).all(): raise FloatingPointError('Nonfinite PPO targets')
            if config.recurrent:
                losses = optimize_recurrent(agent,optimizer,config,obs_buffer,actions,logprobs,
                                            values,adv,returns,state_buffers,starts_buffer)
                writer.add_scalar('lstm/actor_hidden_norm',float(recurrent_state[0].norm()),global_step)
                writer.add_scalar('lstm/critic_hidden_norm',float(recurrent_state[2].norm()),global_step)
                writer.add_scalar('lstm/sequence_length',config.num_steps//config.num_minibatches,global_step)
            else:
                losses = optimize(agent,optimizer,config,obs_buffer,actions,logprobs,values,adv,returns)
            parameter_delta = float((nn.utils.parameters_to_vector(agent.parameters()).detach()-before).norm())
            if not math.isfinite(parameter_delta): raise FloatingPointError('Nonfinite PPO parameters')
            training_seconds += time.perf_counter()-update_start
            writer.add_scalar('charts/SPS_rollout',config.batch_size/rollout_seconds,global_step)
            writer.add_scalar('charts/optimization_seconds',time.perf_counter()-optimization_start,global_step)
            for key,value in losses.items(): writer.add_scalar(f'losses/{key}',value,global_step)
            for choice in choices:
                writer.add_scalar(f'training_scenarios/{choice.name}/completed_episodes',
                                  scenario_episodes[choice.name], global_step)
                writer.add_scalar(f'training_scenarios/{choice.name}/transition_fraction',
                                  scenario_steps[choice.name]/config.batch_size, global_step)
            writer.add_scalar('charts/parameter_delta',parameter_delta,global_step)
            writer.add_scalar('charts/learning_rate',lr,global_step)
            writer.add_scalar('curriculum/cumulative_timesteps',curriculum['step_offset']+global_step,global_step)
            writer.add_scalar('charts/SPS_training',(global_step-invocation_start_step)/training_seconds,global_step)
            writer.add_scalar('charts/SPS_wall',(global_step-invocation_start_step)/(time.perf_counter()-start_time),global_step)
            writer.add_scalar('charts/mean_step_reward',float(rewards.mean()),global_step)
            writer.add_scalar('charts/mean_scaled_step_reward',float(rewards.mean())*config.reward_scale,global_step)
            for key,value in component_sums.items(): writer.add_scalar(f'rewards/{key}',value/config.batch_size,global_step)
            for i,label in enumerate(ACTIONS): writer.add_scalar(f'actions/{label.replace(":","_")}',action_counts[i]/config.batch_size,global_step)
            writer.add_scalar('environment/invalid_cast_rate',invalid_count/max(1,cast_count),global_step)
            writer.add_scalar('environment/collision_rate',collision_count/config.batch_size,global_step)
            writer.add_scalar('environment/cooldown_rejection_rate',cooldown_count/config.batch_size,global_step)
            writer.add_scalar('environment/successful_move_rate',successful_moves/config.batch_size,global_step)
            for key, count in failure_counts.items():
                writer.add_scalar(f'environment/failure_rate/{key}',count/config.batch_size,global_step)
            writer.add_scalar('environment/player_hp',float(obs_buffer[:,:,3].mean()),global_step)
            writer.add_scalar('environment/player_mp',float(obs_buffer[:,:,4].mean()),global_step)
            if update in schedule:
                index = schedule[update]
                checkpoint = run_dir/'checkpoints'/f'checkpoint_{index:02d}.pt'
                atomic_torch_save({'format':'mystic_sim_ppo_v2','metadata':contract,
                    'ppo_config':asdict(config),'agent':agent.state_dict(),'optimizer':optimizer.state_dict(),
                    'update':update,'global_step':global_step,'rng':capture_rng_state(),
                    'episodes':episodes,'wins':wins,'curriculum':curriculum,
                    'scenario_sampler_states': [e.sampling_state() for e in batch.envs] if choices else None},checkpoint)
                agent.eval()
                recording, sampled_recording = checkpoint_recordings(
                    run_dir,index,record_gameplay,record_from_checkpoint)
                report = evaluate(agent,seeds,device=resolved_device,record_path=recording,
                                  record_stride=record_stride,factory=factory)
                sampled_report = evaluate(agent,seeds,device=resolved_device,record_path=sampled_recording,
                                          record_stride=record_stride,stochastic=True,factory=factory)
                # The original environment stays an independent comparison even
                # while a selected easier encounter is being trained.
                full_reports = full_evaluations(agent,seeds,enabled=evaluate_full_scenario,
                    encounter=encounter,device=resolved_device,
                    stage_reports={'greedy':report,'sampled':sampled_report})
                for mode, full_report in full_reports.items():
                    full_report.update(checkpoint=str(checkpoint),global_step=global_step,
                                       scenario='full',cumulative_timesteps=curriculum['step_offset']+global_step)
                    json_write(run_dir/'evaluation'/f'checkpoint_{index:02d}_full_{mode}.json',full_report)
                    for key,value in full_report['summary'].items():
                        if key != 'components': writer.add_scalar(f'evaluation_full/{mode}/{key}',value,global_step)
                sampled_report.update(checkpoint=str(checkpoint),global_step=global_step,
                                      encounter=contract['scenario'].get('encounter'),
                                      cumulative_timesteps=curriculum['step_offset']+global_step)
                json_write(run_dir/'evaluation'/f'checkpoint_{index:02d}_sampled.json',sampled_report)
                # Evaluate every mixture member separately: aggregate success can
                # hide failure on hard encounters. These reports never gate training.
                for choice in choices:
                    if choice.encounter is None and not evaluate_full_scenario:
                        continue
                    for mode in ('sampled', 'greedy'):
                        member_report = (dict(full_reports[mode]) if choice.encounter is None else
                                         evaluate(agent,seeds,device=resolved_device,
                                                  stochastic=mode=='sampled',factory=encounter_factory(choice.encounter)))
                        member_report.update(checkpoint=str(checkpoint),global_step=global_step,
                                             scenario=choice.name,encounter=asdict(choice.encounter) if choice.encounter else None)
                        json_write(run_dir/'evaluation'/f'checkpoint_{index:02d}_scenario_{choice.name}_{mode}.json',member_report)
                        for key,value in member_report['summary'].items():
                            if key != 'components':
                                writer.add_scalar(f'evaluation_scenarios/{choice.name}/{mode}/{key}',value,global_step)
                        print('  ' + evaluation_line(f'{choice.name} {mode}',member_report),flush=True)
                for mode, evaluation in (('greedy', report), ('sampled', sampled_report)):
                    for key,value in evaluation['summary'].items():
                        if key != 'components': writer.add_scalar(f'evaluation/{mode}/{key}',value,global_step)
                    for key,value in evaluation['diagnostics'].items():
                        writer.add_scalar(f'evaluation/{mode}/{key}',value,global_step)
                    if evaluation['recording_error']:
                        writer.add_text(f'recording/{mode}_error',evaluation['recording_error'],global_step)
                report.update(checkpoint=str(checkpoint),global_step=global_step,
                              encounter=contract['scenario'].get('encounter'),
                              cumulative_timesteps=curriculum['step_offset']+global_step,
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
                    'curriculum':curriculum,'cumulative_timesteps':curriculum['step_offset']+global_step,
                    'parameter_delta':parameter_delta,'episodes':episodes,'wins':wins,
                    'training_sps':(global_step-invocation_start_step)/training_seconds,
                    'wall_sps':(global_step-invocation_start_step)/(time.perf_counter()-start_time)})
                writer.flush()
                print(f'Checkpoint {index}/10: step={global_step}, '
                      f'SPS={(global_step-invocation_start_step)/training_seconds:.0f}', flush=True)
                print('  ' + evaluation_line('stage sampled', sampled_report), flush=True)
                print('  ' + evaluation_line('stage greedy', report), flush=True)
                if encounter is not None and full_reports:
                    for mode in ('sampled', 'greedy'):
                        print('  ' + evaluation_line(f'full {mode}', full_reports[mode]), flush=True)
                if stop_after_checkpoint == index: break
        return run_dir
    finally:
        if batch: batch.close()
        writer.close()


def main(argv=None, *, defaults=None, record_from_checkpoint=1, run_root=None, full_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = defaults if defaults is not None else PPOConfig()
    for field in fields(defaults):
        default = getattr(defaults,field.name)
        options = {'action':argparse.BooleanOptionalAction} if isinstance(default,bool) else {
            'type':float if field.name == 'value_clip_coef' else type(default)}
        if field.name == 'actor_head': options['choices'] = ('linear', 'tanh')
        if field.name == 'recurrent_encoder': options['choices'] = ('flat', 'cross_attention')
        parser.add_argument('--'+field.name.replace('_','-'),default=argparse.SUPPRESS,**options)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--run-dir',type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--resume',type=Path,help='Resume the original saved budget; fresh simulator episodes')
    initialization.add_argument('--continue-from',type=Path,
                                help='Continue weights/optimizer with a new --total-timesteps additional budget')
    parser.add_argument('--record-gameplay',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--record-stride',type=int,default=5,help='Record every N decisions; 5 = one frame per simulated second')
    parser.add_argument('--record-from-checkpoint',type=int,default=record_from_checkpoint)
    parser.add_argument('--evaluate-full-scenario',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--torch-threads',type=int,default=1)
    parser.add_argument('--stop-after-checkpoint',type=int,help='Stop cleanly after checkpoint 1..10; resume later')
    parser.add_argument('--validate-only',action='store_true',help='Check settings, map/reset and checkpoint without training or writing run files')
    args = parser.parse_args(argv)
    if args.torch_threads < 1: parser.error('--torch-threads must be positive')
    torch.set_num_threads(args.torch_threads)
    overrides = {field.name: getattr(args, field.name) for field in fields(defaults)
                 if hasattr(args, field.name)}
    if args.continue_from and 'total_timesteps' not in overrides:
        parser.error('--continue-from requires --total-timesteps (additional transitions)')
    config = PPOConfig(**{**asdict(defaults), **overrides})
    if run_root is not None:
        run_root = Path(run_root).resolve()
        if args.resume is not None: args.resume = (run_root/args.resume).resolve()
        if args.continue_from is not None: args.continue_from = (run_root/args.continue_from).resolve()
        if args.run_dir is not None: args.run_dir = (run_root/args.run_dir).resolve()
    source_path = args.resume or args.continue_from
    raw = torch.load(source_path,map_location='cpu',weights_only=True) if source_path else None
    if args.continue_from:
        config = PPOConfig(**{**asdict(saved_config(raw)), **overrides})
    if run_root is not None and args.run_dir is None and args.resume is None:
        args.run_dir = run_root/'runs'/'mystic_sim'/'full'/(
            datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+f'_seed{config.seed}_{uuid.uuid4().hex[:6]}')
    encounter = encounter_from_dict(raw['metadata']['scenario'].get('encounter')) if raw else None
    if full_only or args.validate_only or args.continue_from:
        if args.resume:
            config = saved_config(raw)
        encounter = encounter_from_dict(raw['metadata']['scenario'].get('encounter')) if raw else None
        if full_only and encounter is not None:
            parser.error('This launcher trains the original full scenario; use curriculum_run for curriculum checkpoints')
        config.validate()
        if args.device=='cuda' and not torch.cuda.is_available():
            parser.error('CUDA requested but unavailable; install CUDA-enabled PyTorch or use --device cpu')
        if args.record_stride < 1 or not 1 <= args.record_from_checkpoint <= 10:
            parser.error('record-stride must be positive; record-from-checkpoint must be in 1..10')
        if args.stop_after_checkpoint is not None and not 1 <= args.stop_after_checkpoint <= 10:
            parser.error('stop-after-checkpoint must be in 1..10')
        env = encounter_factory(encounter)()
        try:
            if args.resume:
                payload = load_checkpoint(args.resume,metadata(env,recurrent=config.recurrent,actor_head=config.actor_head, recurrent_encoder=config.recurrent_encoder))
                if payload['update'] >= config.updates:
                    parser.error('Checkpoint already completed its training budget')
            if args.continue_from:
                source = load_stage_checkpoint(args.continue_from,
                    metadata(env,recurrent=config.recurrent,actor_head=config.actor_head, recurrent_encoder=config.recurrent_encoder),config)
                continuation_schedule = stage_schedule(config, source_payload=source)
            env.reset(seed=config.seed)
            report = {'scenario':'full' if encounter is None else encounter.stage_id,
                      'enemy_count':len(env.world.monsters),'kill_goal':env.reward_config.win_kills,
                      'max_episode_steps':env.reward_config.max_episode_steps,
                      'initialization':'continue' if args.continue_from else 'resume' if raw else 'fresh','ppo_config':asdict(config),
                      'actual_timesteps':config.updates*config.batch_size,
                      'run_dir':str(args.run_dir or args.resume.resolve().parent.parent) if args.run_dir or args.resume else None,
                      'record_from_checkpoint':args.record_from_checkpoint,
                      'device':('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device}
            if args.continue_from:
                report['source_checkpoint'] = str(args.continue_from.resolve())
                report['cumulative_timesteps_after_run'] = continuation_schedule['step_offset'] + config.updates*config.batch_size
        finally:
            env.close()
        print(json.dumps(report,indent=2),flush=True)
        if args.validate_only:
            return report
    return train(config,run_dir=args.run_dir,device=args.device,resume=args.resume,
          continue_from=args.continue_from,encounter=encounter,
          record_gameplay=args.record_gameplay,record_stride=args.record_stride,
          record_from_checkpoint=args.record_from_checkpoint,evaluate_full_scenario=args.evaluate_full_scenario,
          stop_after_checkpoint=args.stop_after_checkpoint)


if __name__ == '__main__':
    main()
