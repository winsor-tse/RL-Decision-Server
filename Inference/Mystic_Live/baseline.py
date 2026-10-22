"""Freeze Phase 8A evidence without running a policy, training, or contacting a game.

python -m Inference.Mystic_Live.baseline --output-dir runs/mystic_live/baseline
Optional: --checkpoint PATH --capture PATH --extension-dir PATH
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import statistics
import subprocess

from Automation.Bridge.evidence import SCHEMA

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = 'mystic_live_baseline_v1'
SOURCE_FILES = (
    'Automation/Bridge/ws_zmq_bridge.py', 'Automation/Bridge/evidence.py',
    'Inference/Mystic_Live/baseline.py', 'Inference/Mystic_Live/PHASE_8A.md',
    'Tests/fixtures/live/phase8a_scenarios.json', 'Tests/fixtures/simulation/provenance.json',
    'Custom_enviornments/Config.yaml',
    'Custom_enviornments/BaseEnv.py', 'Custom_enviornments/Test_Env/Mystic.py',
    'Custom_enviornments/Test_Env/Env_conditions.py', 'Training/Mystic_Sim/core.py',
    'Training/Mystic_Sim/history.py', 'Training/Mystic_Sim/recurrent.py',
    'Training/Mystic_Sim/Attention.py', 'Custom_enviornments/Mystic_Sim/actions.py',
    'Custom_enviornments/Mystic_Sim/observation.py', 'Custom_enviornments/Mystic_Sim/engine.py',
    'Custom_enviornments/Mystic_Sim/config.py', 'Custom_enviornments/Mystic_Sim/mechanics_manifest.yaml',
)
EXTENSION_FILES = ('manifest.json', 'battler.js', 'background.js', 'content-bridge.js', 'contentScript.js')
SCENARIOS = ('movement', 'blocked_movement', 'spell_1', 'spell_2', 'spell_3',
             'no_target', 'cooldown_rejection', 'player_death', 'map_transition', 'disconnect')


def file_identity(path):
    path = Path(path)
    if not path.is_file():
        return dict(path=str(path), available=False, sha256=None)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), available=True, sha256=digest.hexdigest(), bytes=path.stat().st_size)


def git_identity(root):
    def run(*args):
        try:
            result = subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                                    text=True, check=True, timeout=10)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = run('status', '--porcelain=v1', '--untracked-files=all')
    return dict(revision=run('rev-parse', 'HEAD'), status_porcelain=status,
                dirty=bool(status) if status is not None else None,
                note='Revision plus working-tree status and source hashes; not a claim of a clean checkout.')


def extension_identity(directory):
    if directory is None:
        return dict(status='unselected', loaded_extension_verified=False, files={})
    directory = Path(directory).resolve()
    files = {name: file_identity(directory / name) for name in EXTENSION_FILES}
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8')) if files['manifest.json']['available'] else {}
    return dict(status='local_source_only', path=str(directory), name=manifest.get('name'),
                version=manifest.get('version'), files=files, loaded_extension_verified=False,
                note='Local files/version do not identify the extension actually loaded in the browser.')


def checkpoint_identity(path):
    if path is None:
        return dict(status='unselected', path=None, sha256=None, metadata=None,
                    local_compatibility='not_checked',
                    note='No deployment checkpoint selected; no latest-file inference or substitution.')
    import torch
    from Training.Mystic_Sim.core import load_checkpoint, metadata, saved_config
    from Training.Mystic_Sim.curriculum import checkpoint_factory
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'Checkpoint not found: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != 'mystic_sim_ppo_v2':
        raise ValueError('Expected a Mystic Sim PPO v2 checkpoint')
    result = dict(status='selected', **file_identity(path), format=payload['format'],
                  metadata=payload.get('metadata'), ppo_config=payload.get('ppo_config'),
                  global_step=payload.get('global_step'), update=payload.get('update'))
    env = None
    try:
        config = saved_config(payload)
        env = checkpoint_factory(payload)()
        expected = metadata(env, recurrent=config.recurrent, actor_head=config.actor_head,
                            recurrent_encoder=config.recurrent_encoder)
        load_checkpoint(path, expected)
        result.update(local_compatibility='passed', compatibility_error=None)
    except (ValueError, KeyError, TypeError) as error:
        result.update(local_compatibility='failed', compatibility_error=str(error))
    finally:
        if env is not None:
            env.close()
    # Recording metadata never restores Adam, RNG, episode state or model weights.
    return result


def read_capture(path):
    records = []
    def reject_constant(value):
        raise ValueError(f'Non-finite JSON number: {value}')
    with Path(path).open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=reject_constant)
            except ValueError as error:
                raise ValueError(f'{path}:{line_no}: invalid/truncated JSONL: {error}') from error
            if not isinstance(row, dict) or row.get('schema') != SCHEMA:
                raise ValueError(f'{path}:{line_no}: unsupported capture schema')
            if (not isinstance(row.get('session_id'), str) or not isinstance(row.get('event'), str)
                    or type(row.get('sequence')) is not int or row['sequence'] < 1):
                raise ValueError(f'{path}:{line_no}: missing/invalid session, event or sequence')
            records.append(row)
    return records


def capture_summary(path, records):
    events, requests, replies = {}, {}, {}
    client_identity, server_identity, labels = [], [], set()
    previous_sequence = {}
    gaps = []
    for row in records:
        event = row.get('event')
        events[event] = events.get(event, 0) + 1
        session = row.get('session_id')
        sequence = row.get('sequence')
        if isinstance(sequence, int):
            previous = previous_sequence.get(session, 0)
            if sequence != previous + 1:
                gaps.append(dict(session_id=session, after=previous, next=sequence))
            previous_sequence[session] = sequence
        key = (session, row.get('connection_id'), row.get('request_id'))
        if event == 'client_request' and isinstance(row.get('request'), dict) and row['request'].get('type') == 'ai_tick':
            requests.setdefault(key, []).append(row)
            request = row['request']
            for field, collection in (('clientInfo', client_identity), ('serverInfo', server_identity)):
                if request.get(field) is not None:
                    collection.append(dict(evidence='sender_reported_unverified', value=request[field]))
        if event == 'client_response_sent':
            replies.setdefault(key, []).append(row)
        if row.get('label'):
            labels.add(row['label'])
    endings = [r for r in records if r.get('event') == 'capture_end']
    sessions = {r['session_id'] for r in records}
    completed_sessions = {r['session_id'] for r in endings}
    started_sessions = {r['session_id'] for r in records if r.get('event') == 'capture_start'}
    last_events = {r['session_id']: r['event'] for r in records}
    roundtrip = []
    for key, input_rows in requests.items():
        output_rows = replies.get(key, [])
        if len(input_rows) == len(output_rows) == 1:
            start = input_rows[0].get('bridge_received_monotonic_ns', input_rows[0].get('monotonic_ns'))
            end = output_rows[0].get('monotonic_ns')
            if type(start) is int and type(end) is int and end >= start:
                roundtrip.append((end - start) / 1_000_000)
    return dict(file=file_identity(path), origins=sorted({r.get('evidence_origin', 'unspecified') for r in records}),
                events=events, ai_tick_count=sum(map(len, requests.values())),
                unmatched_requests=[dict(session_id=k[0], connection_id=k[1], request_id=k[2])
                                    for k in requests if k not in replies],
                duplicate_request_keys=[dict(session_id=k[0], connection_id=k[1], request_id=k[2])
                                        for k, rows in requests.items() if len(rows) > 1],
                request_id_mismatches=[dict(request_id=k[2], response_id=row.get('response', {}).get('requestId'))
                                       for k, rows in replies.items() for row in rows
                                       if row.get('response', {}).get('requestId') != k[2]],
                sequence_gaps=gaps, completion_records=endings,
                clean_completion=bool(endings) and sessions == completed_sessions == started_sessions
                                 and len(endings) == len(sessions)
                                 and all(event == 'capture_end' for event in last_events.values())
                                 and all(not r.get('queue_dropped') and not r.get('limit_dropped')
                                                     and not r.get('writer_error') for r in endings)
                                 and not gaps,
                reported_client_identity=client_identity, reported_server_identity=server_identity,
                bridge_roundtrip_ms=(dict(samples=len(roundtrip), minimum=min(roundtrip),
                                          median=statistics.median(roundtrip), maximum=max(roundtrip))
                                     if roundtrip else None),
                operator_labels=sorted(labels),
                scenario_coverage={name: 'labeled_only_execution_unverified' if name in labels else 'not_captured'
                                   for name in SCENARIOS},
                actual_loaded_extension='unverified', actual_game_build='unverified',
                note='Transport capture is not proof of an actual game client, build identity, or command execution.')


def nested(obj, path):
    for key in path.split('.'):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def field_matrix(requests, source, *, command_evidence=False):
    """Presence/type checks describe evidence; missing input is never invented."""
    import math
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)
    fields = (
        ('requestId', 'Exchange correlation', 'observed', lambda v: isinstance(v, str) and bool(v)),
        ('worldState.player.mapID', 'Configured map identity', 'observed', lambda v: isinstance(v, str) or type(v) is int),
        ('worldState.player.mapX', 'Player X', 'observed', number),
        ('worldState.player.mapY', 'Player Y', 'observed', number),
        ('worldState.player.direction', 'Player facing', 'observed', lambda v: v in ('up', 'down', 'left', 'right', 'idle')),
        ('worldState.player.hp', 'Player HP fraction numerator', 'observed', number),
        ('worldState.player.maxHp', 'Player HP fraction denominator', 'observed', lambda v: number(v) and v > 0),
        ('worldState.player.mp', 'Player MP fraction numerator', 'observed', number),
        ('worldState.player.maxMp', 'Player MP fraction denominator', 'observed', lambda v: number(v) and v >= 0),
        ('worldState.entities', 'Enemy slots, filtering and sorting', 'observed', lambda v: isinstance(v, list)),
        ('worldState.timestamp', 'Client snapshot time (not game-server execution time)', 'observed', number),
        ('timestamp', 'Client envelope time (not game-server execution time)', 'observed', number),
    )
    result = []
    for path, purpose, availability, valid in fields:
        count = sum(valid(nested(r, path)) for r in requests)
        result.append(dict(field=path, purpose=purpose, evidence=availability if count else 'unavailable',
                           source=source, valid_samples=count, total_samples=len(requests),
                           complete=bool(requests) and count == len(requests)))
    entity_fields = ('id', 'type', 'isCurrentPlayer', 'mapX', 'mapY', 'hp', 'maxHp', 'mp', 'maxMp', 'distance')
    entity_lists = [nested(r, 'worldState.entities') for r in requests]
    entities = [e for items in entity_lists if isinstance(items, list) for e in items if isinstance(e, dict)]
    def valid_entity_field(entity, field):
        value = entity.get(field)
        if field == 'id':
            return type(value) is int or (isinstance(value, str) and bool(value))
        if field == 'type':
            return isinstance(value, str) and bool(value)
        if field == 'isCurrentPlayer':
            return type(value) is bool
        return number(value) and (field not in ('maxHp', 'maxMp', 'distance') or value >= 0)
    for field in entity_fields:
        count = sum(valid_entity_field(e, field) for e in entities)
        result.append(dict(field=f'worldState.entities[].{field}', purpose='Enemy observation reference',
                           evidence='observed' if count else 'unavailable', source=source,
                           valid_samples=count, total_samples=len(entities),
                           complete=bool(entities) and count == len(entities)))
    for field, purpose in (
        ('execution.action_applied', 'Required recurrent history flag'),
        ('execution.movement_blocked', 'Required recurrent history flag'),
        ('execution.cooldown_rejected', 'Required recurrent history flag'),
        ('execution.successful_cast_time', 'Required confirmed cast ages/seen flags'),
        ('execution.game_server_time', 'Authoritative elapsed execution time'),
        ('identity.loaded_extension', 'Actually loaded browser extension identity'),
        ('identity.game_build', 'Connected game-server build'),
    ):
        result.append(dict(field=field, purpose=purpose, evidence='unavailable', source=source,
                           valid_samples=0, total_samples=len(requests), complete=False,
                           note='No validated producer/acknowledgement contract in the current plugin; raw extra fields are preserved for review.'))
    result.extend((
        dict(field='enemy_relative_offsets', purpose='Five dx/dy pairs', evidence='derived',
             source='Current player/entity coordinates; encoding parity is Phase 8C'),
        dict(field='terrain_rays', purpose='Eight recurrent terrain inputs', evidence='derived',
             source='User-confirmed shared map53 static collision layer; existing four-tile sensor algorithm'),
        dict(field='previous_chosen_action', purpose='Seven history one-hot flags',
             evidence='derived' if command_evidence else 'unavailable',
             source=('Correlated outbound command under an explicitly selected action ordering'
                     if command_evidence else 'No outbound action choice in the supplied evidence')),
    ))
    return result


def build_baseline(*, checkpoint=None, captures=(), extension_dir=None, deployment_info=None, root=ROOT):
    root = Path(root).resolve()
    fixture = root / 'Tests/fixtures/simulation/example_full_state.json'
    map_path = root / 'Tests/fixtures/simulation/map53.json'
    packaged_map = root / 'Custom_enviornments/Mystic_Sim/data/map53.json'
    map_data = json.loads(map_path.read_text(encoding='utf-8'))
    map_identity, packaged_identity = file_identity(map_path), file_identity(packaged_map)
    reference = json.loads(fixture.read_text(encoding='utf-8'))
    summaries, requests, has_commands = [], [], False
    for path in captures:
        rows = read_capture(path)
        summaries.append(capture_summary(path, rows))
        has_commands = has_commands or any(row.get('event') == 'client_response_sent'
                                           and isinstance(row.get('response'), dict)
                                           and isinstance(row['response'].get('move'), str) for row in rows)
        requests.extend(row['request'] for row in rows if row.get('event') == 'client_request'
                        and isinstance(row.get('request'), dict) and row['request'].get('type') == 'ai_tick')
    # The reference contains the inner state only; do not invent outer fields
    # and then report them as observed evidence.
    reference_request = dict(worldState=reference)
    matrix = (field_matrix(requests, 'supplied_capture', command_evidence=has_commands)
              if captures else field_matrix([reference_request], 'reference_fixture_only'))
    deployment = json.loads(Path(deployment_info).read_text(encoding='utf-8')) if deployment_info else None
    packages = {}
    for package in ('torch', 'numpy', 'gymnasium', 'pyzmq', 'websockets'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    from Custom_enviornments.Mystic_Sim.actions import contract_metadata
    from Training.Mystic_Sim.history import HISTORY_SCHEMA, HISTORY_FEATURES
    from Custom_enviornments.Load_env_config import load_env_config
    from Custom_enviornments.Mystic_Sim.env import MysticSimEnv
    from Training.Mystic_Sim.core import metadata
    reference_env = MysticSimEnv()
    try:
        current_recurrent_reference = metadata(reference_env, recurrent=True)
    finally:
        reference_env.close()
    source_files = set(SOURCE_FILES)
    for folder in ('Custom_enviornments/Mystic_Sim', 'Training/Mystic_Sim'):
        source_files.update(p.relative_to(root).as_posix() for p in (root / folder).glob('*.py'))
    return dict(schema=MANIFEST_SCHEMA, created_at_utc=datetime.now(timezone.utc).isoformat(),
                scope='single_episode_frozen_actor_inference', repository=git_identity(root),
                environment=dict(python=platform.python_version(), platform=platform.platform(), packages=packages),
                source_inventory={path: file_identity(root / path) for path in sorted(source_files)},
                checkpoint=checkpoint_identity(checkpoint),
                current_reference_contract=dict(base=contract_metadata(), recurrent_schema=HISTORY_SCHEMA,
                                                recurrent_size=80, history_features=HISTORY_FEATURES,
                                                recurrent_defaults=current_recurrent_reference,
                                                note='Current code defaults for reference only, not a selected deployment model.',
                                                deployment_checkpoint_selected=checkpoint is not None),
                legacy_live_config=load_env_config(),
                map=dict(fixture=map_identity, packaged=packaged_identity,
                         hashes_match=map_identity['sha256'] == packaged_identity['sha256'],
                         width=map_data['width'], height=map_data['height'],
                         blocked_cells=sum(bool(tile) for layer in map_data['layers'] if layer.get('name') == 'blocked'
                                           for tile in layer.get('data', [])),
                         live_geometry_evidence='user_confirmed_same_map_and_obstacles'),
                payload_reference=dict(file=file_identity(fixture), example_map_id=reference['player']['mapID'],
                                       original=file_identity(root / 'Simulation/Source-Code/Example_Full_State.txt'),
                                       note='Nested worldState reference, not an ai_tick envelope or a live map53 capture.'),
                extension=extension_identity(extension_dir),
                deployment=dict(operator_report=deployment, evidence='operator_reported_unverified' if deployment else 'unavailable',
                                actual_loaded_extension='unverified', actual_game_build='unverified'),
                captures=summaries, field_availability=matrix,
                validation=dict(real_game_session='not_verified', policy_live_compatible=False,
                                phase_8a='evidence_inventory_only',
                                missing_dependencies=[r['field'] for r in matrix if r['evidence'] == 'unavailable']))


def markdown_report(manifest):
    checkpoint = manifest['checkpoint']
    lines = ['# Phase 8A baseline evidence', '',
             f'Generated: {manifest["created_at_utc"]}', '',
             f'Repository revision: `{manifest["repository"]["revision"]}`; dirty: `{manifest["repository"]["dirty"]}`.',
             f'Checkpoint: **{checkpoint["status"]}**. Local compatibility: `{checkpoint["local_compatibility"]}`.',
             'No deployment checkpoint is chosen automatically. No policy was run or trained.',
             'Current simulator reference: seven actions, 56 base / 80 recurrent inputs; '
             'divide-by-Box-high preprocessing. Full architecture, scales and scenario defaults are in baseline.json.',
             'Legacy live reference: eleven actions and 26 inputs. These contracts are not interchangeable.', '',
             f'Shared map53 hashes match: **{manifest["map"]["hashes_match"]}**; '
             f'{manifest["map"]["width"]} x {manifest["map"]["height"]}; {manifest["map"]["blocked_cells"]} blocked cells.',
             'Live obstacle equality is user-confirmed. Player/entity state remains dynamic.', '',
             '**Actually loaded extension and connected game build: unverified.** Local extension hashes '
             'and optional operator/sender reports are recorded separately from deployment verification.', '',
             '**Real game validation: not verified.** A fixture or transport capture cannot prove authoritative '
             'execution, selected target, server timing, or successful casts.', '',
             '## Field availability', '', '| Field | Evidence | Valid / total samples | Source / dependency |', '|---|---|---|---|']
    for row in manifest['field_availability']:
        detail = row.get('note', row['source']).replace('|', '\\|')
        samples = f'{row["valid_samples"]} / {row["total_samples"]}' if 'valid_samples' in row else 'derived'
        lines.append(f'| `{row["field"]}` | {row["evidence"]} | {samples} | {detail} |')
    lines += ['', '## Capture inventory', '']
    if not manifest['captures']:
        lines += ['No live transport captures supplied. Availability above is from the reference fixture only.']
    for summary in manifest['captures']:
        lines += [f'- `{summary["file"]["path"]}`: {summary["ai_tick_count"]} ticks; '
                  f'clean completion: {summary["clean_completion"]}; origins: {", ".join(summary["origins"])}.']
        lines += [f'  - {name}: {state}' for name, state in summary['scenario_coverage'].items()]
    lines += ['', '## Remaining work', '',
              '- Select a deployment checkpoint explicitly when ready.',
              '- Identify the actually loaded extension and game build; absent identity telemetry stays unknown.',
              '- Record required scripted cases through the bridge when a real game is available.',
              '- Provide validated applied/blocked/cooldown/cast-success feedback for recurrent history (Phase 8C).',
              '- Implement single-run lifecycle and death stop in Phase 8B; Phase 8A never sends its own actions.',
              '- Optional reward/kill reporting, reset loops and training are not inference prerequisites.', '']
    return '\n'.join(lines)


def write_baseline(output_dir, manifest):
    output_dir = Path(output_dir)
    # Freeze a distinct bundle; reruns must choose a new directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'baseline.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output_dir / 'report.md').write_text(markdown_report(manifest), encoding='utf-8')
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Explicitly selected simulator checkpoint; optional, no default')
    parser.add_argument('--capture', type=Path, action='append', default=[], help='Versioned bridge JSONL; repeat for several files')
    parser.add_argument('--extension-dir', type=Path, help='Local plugin source inventory; not proof it is loaded')
    parser.add_argument('--deployment-info', type=Path, help='Optional operator-reported build identity JSON, kept unverified')
    parser.add_argument('--output-dir', type=Path, required=True, help='New baseline bundle directory')
    args = parser.parse_args(argv)
    try:
        manifest = build_baseline(checkpoint=args.checkpoint, captures=args.capture,
                                  extension_dir=args.extension_dir, deployment_info=args.deployment_info)
        output = write_baseline(args.output_dir, manifest)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Baseline failed: {error}\n')
    print(f'Baseline: {output / "baseline.json"}\nReport: {output / "report.md"}\n'
          f'Checkpoint: {manifest["checkpoint"]["status"]}; real-game validation: not verified', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
