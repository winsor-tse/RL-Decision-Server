# Watch a checkpoint on a chosen seed

From the repository root, with the training and Pygame dependencies installed:

```powershell
.\Watch_Mystic_Policy.bat runs/mystic_sim/full/full80_7p2m_cp04_continue_1m_lr1e4_seed42/checkpoints/checkpoint_10.pt --seed 91000 --mode sampled
```

This example watches the starting 8.2M CP10. Any compatible checkpoint can be
supplied; no checkpoint path is hard-coded in the launcher. Change `--seed` to
the episode seed from a comparison report, or use `--mode greedy` for argmax.
Sampled mode uses the same seed-derived action RNG as `compare`/`evaluate`.
CPU inference is the default; `--device cuda` is optional, but device/kernel
differences can affect exact replay. Match the evaluation device for comparisons.

The portable module equivalent is:

```powershell
python -m Training.Mystic_Sim.watch PATH_TO_CHECKPOINT --seed 91000 --mode sampled
```

Relative checkpoint paths resolve against the repository root. The Windows
launcher uses the active virtual environment, then RL_venv, then .venv.

- Space: start/pause (starts paused).
- R: reset the entire environment and replay the same seed, resetting LSTM
  hidden states, observation history and policy-sampling RNG.
- Shift+R: reset with the next seed.
- Minus/plus: playback speed; this does not change the simulation decision size.
- Tab: map overview. Mouse wheel: zoom. B: spawn boxes. C: casting range.
- Esc: exit. Manual movement and spell keys are disabled.

The viewer retains the checkpoint's scenario via the existing evaluation loader.
It advances one policy action per 200 ms simulation decision, stops on the actual
environment termination/truncation, displays steps/reward, and prints the episode
outcome to the console. It never trains or writes checkpoint files. No free-play
override is used. Existing `Play_Mystic_Sim.bat` remains the manual viewer.

Headless rendering smoke check (no interactive window):

```powershell
python -m Training.Mystic_Sim.watch PATH_TO_CHECKPOINT --seed 91000 --smoke-test
```
