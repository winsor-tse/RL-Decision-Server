# Phase 8A scripted capture fixtures

`phase8a_scenarios.json` specifies constructed map53 inputs and stub commands.
`phase8a_capture.jsonl` is their saved capture-format example. These are synthetic
transport/parser evidence, never a recorded live game or mechanics oracle.

The nested state follows `../simulation/example_full_state.json`; map ID, stats,
positions and timestamps are explicitly constructed. A label such as
`blocked_movement` or `cooldown_rejection` describes the scenario to investigate,
not proof of a collision or rejection. No authoritative cast/action outcomes,
server execution times, runtime extension identity or game build are invented.

The fixtures cover movement, intended blocked movement, all three spells,
no-target and cooldown attempts, death, map change and disconnect. Automated
tests additionally send the inputs over actual local WebSocket/ZMQ sockets with
a stub backend, checking that the optional recorder preserves raw messages.
Neither the fixture nor a stub backend executes game actions.
