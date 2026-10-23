# Step 1: frozen actor integration

Implemented 2026-10-23. `policy.py:FrozenPolicy` reuses
`Training.Mystic_Sim.evaluate.load_evaluation_agent`, including exact architecture
and checkpoint validation. It loads existing feedforward, flat recurrent and
cross-attention recurrent policies. Inputs are raw float32 vectors, normalized
once by the saved scale. Live/map bounds and the loaded normalization buffer
must match. Nonfinite weights fail before binding the server.

The model is frozen in eval mode. Decisions run under inference mode, evaluate
only the actor, and return `PolicyChoice`. Recurrent memory contains actor hidden
and cell state only, initialized once per run. No optimizer is constructed or
restored; no simulator world is reset or stepped. The original checkpoint format
still contains training fields required by the shared loader.

## Current CLI

Observation mode remains available:

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.server --observe-only
```

For a compatible 56-input checkpoint (Step 2 is now implemented):

```powershell
.\RL_venv\Scripts\python.exe -m Inference.Mystic_Live.server --checkpoint PATH_TO_CHECKPOINT
.\RL_venv\Scripts\python.exe -m Automation.Bridge.ws_zmq_bridge --single-run
```

Use separate terminals. Add `--sampled` for categorical sampling; greedy is the
default. `--device` defaults to `cpu`. In checkpoint mode `--input-size` is
inferred, and a conflicting override fails. The final JSON records the selected
checkpoint path, metadata, sampling choice and progression mode.

**Step 2 enables continuous snapshot inference for compatible 56-input policies.**
See [STEP_2.md](STEP_2.md). Existing 80-input checkpoints require explicit
`--confirmed-feedback`; this flag does not install or emulate a producer. They
also require authoritative observation context and outcomes as specified in
PHASE_8C. [Step 3](STEP_3.md) adds explicit commands, extension updates and the
optional 71-input recurrent contract; reload the extension before deployment.
Restart both bridge and backend for a new run.

## Validation

```powershell
.\RL_venv\Scripts\python.exe -m unittest Tests.test_mystic_live_policy Tests.test_mystic_live_inputs Tests.test_mystic_live_lifecycle Tests.test_mystic_live_evidence
```

The new tests create real checkpoint files with synthetic initialized weights,
then compare normalized inputs, logits, greedy actions, probabilities and
recurrent state against simulator inference across multiple observations and all
supported architecture/head combinations. They check sampled selection, frozen
weights, no live critic call, contract/buffer rejection, CLI mode selection and
loaded-actor lifecycle behavior across feedback, duplicates and death.
These are local tests, not a trained-policy game evaluation.

Validation result: all 53 tests in the command above passed on 2026-10-23.
