# Headless Yugen simulation scaffold

`YugenSimEnv.hpp` is a standalone C++17 class using only the standard library.
It owns a 32x32 row-major map, player, enemy vector, absolute-step cooldowns,
and an integer simulation clock. No graphics, sockets, sleeps, or game client
are required. Separate instances own separate state.

`Reset()` restores a deterministic scenario with a wall boundary, a player at
(16,16), two enemies, and all cooldowns ready. Class choice persists across
resets. Stats, cooldown durations, and spawns are illustrative, not measured
Yugen Saga values. All classes currently have identical placeholder stats.

`Step(int action)` is deliberately a stub: it validates an action, advances the
clock once, and returns zero reward with both end flags false. It does not yet
move, attack, spend mana, activate cooldowns, or implement enemy AI. It cannot
train useful combat behavior yet. Read state through `GetState()`; returned
references are live views, not snapshots.

Action IDs match `Custom_enviornments/Test_Env/Env_16.py`: 0..3 are up, down,
left, right; 4 is attack; 5..10 are spell slots 1, 2, 3, 5, 6, 7. Cooldown array
indices are separate: 0 is attack and 1..7 are spell slots. A successful cast
at step t should set `ready_at_step = t + duration_steps`; it becomes available
when `current_step >= ready_at_step`. Define a fixed tick duration when adding
combat and convert measured game timings into steps consistently.

## Build

With CMake and a C++17 compiler installed, from the repository root:

```powershell
cmake -S Simulation -B Simulation/build
cmake --build Simulation/build --config Release
```

Run `Simulation/build/Release/yugen_sim_example.exe` for a Visual Studio build,
or `Simulation/build/yugen_sim_example.exe` for a single-configuration build.

## Practical development outline

1. Measure the live game's movement cadence, collision rules, targeting,
   damage, mana costs, regeneration, cooldowns, aggro, and respawn behavior.
   Use recorded trajectories to check one-step simulator predictions.
2. Implement one class and one enemy type: movement and collision first,
   then range, attacks, cooldowns, mana, enemy pursuit, and death. Choose and
   document event ordering within each fixed tick.
3. Add observations, rewards, action masks, death/objective termination, and
   separate time-limit truncation. Keep observations limited to information
   available through the live client. Match the existing Python environment's
   feature order, normalization, action mapping, and decision interval before
   attempting policy transfer; matching dimensions alone is insufficient.
4. Expose reset/step through a Python binding and Gymnasium wrapper, then batch
   independent instances. Benchmark steps per second after combat exists;
   C++ and headless execution alone do not establish a throughput figure.
5. Increase difficulty from one stationary enemy to pursuit, kiting, multiple
   enemies, obstacles, and resource management. Randomize seeded spawn layouts
   and plausible timing/damage parameters to reduce dependence on one scenario.
6. Evaluate against held-out layouts and recorded live transitions. Validate
   policies in the real environment, identify model errors, and refine the
   simulator before considering limited live fine-tuning.

The existing live Gymnasium/ZMQ trainer remains the eventual transfer target;
this scaffold does not yet include a Python binding or integrate with training.
