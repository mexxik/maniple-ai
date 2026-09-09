# Unreal plugins

Two plugins, both built with the project (symlink or copy them into `<Project>/Plugins/`):

| plugin | what it is |
|---|---|
| `ManipleInference` | Triton gRPC client and the Maniple agent protocol. No game code, reusable in any project. |
| `ManipleLyra` | Reference integration for the Lyra Starter Game: bots trained and served from Triton, all driven by launch arguments. |

The gRPC library and Triton protos are prebuilt into `ManipleInference/Source/ThirdParty/ManipleGrpc`
(see the root README, *Build*).

## ManipleInference

- `FManipleTritonClient` — one persistent bidirectional `ModelStreamInfer` stream per client, requests matched
  by id, reconnects on failure, completions delivered on the game thread. Raw tensors in `FManipleTensor`
  (FP32, INT64, BOOL, BYTES helpers).
- `FManipleAgentClient` — the C++ twin of `gym/triton_agent.py`: one named policy on one server.
  `Register(spec)`, `Act(obs, rows, explore, channel)`, `Observe(transitions)`, `Status()`, `Report`,
  `Promote`, `Export`. Channel `latest` goes to the trainer (`ppo_train`, the weights being trained);
  `best`, `stable` or a version number go to `ppo_infer` (exported ONNX / TensorRT).
- `FManipleAgentSpec` — obs dim, action space, network preset, PPO overrides; serialised to the same JSON
  the gym scripts send.
- `FManipleBatchInferer` — generic client-side batching for plain models (obs in, action out).

Tests (`Automation RunTests Maniple`, Triton must be up): `Maniple.Triton.Smoke` registers a throwaway policy
`uesmoke`, acts, observes and reads status; `Maniple.Triton.Batch` measures round trips by batch size.

## ManipleLyra

Server-side only. `UManipleBotSubsystem` (a tickable world subsystem, created only when `-ManipleBrain` is
set) finds Lyra's bot controllers, attaches a `UManipleAgentComponent` to every bot pawn (again after each
respawn), optionally spawns extra bots, and runs the decision loop: one `act` request per tick for all agents,
one `observe` request per tick with the transitions of the previous tick.

`UManipleAgentComponent` stops the behaviour tree, keeps its own aim rotation (Lyra's AI controller would reset
pitch every tick), builds the observation, applies the action every frame, reloads when the magazine is empty (fire input is
suppressed until it refills, since firing cancels the reload),
and accumulates reward between decisions. One component is one episode: the pawn's death ends it.

Fire presses both `InputTag.Weapon.Fire` (pistol) and `InputTag.Weapon.FireAuto` (rifle, shotgun): bots pick up the
map's weapon spawners by walking over them and the new weapon becomes active. A watchdog logs a gun that wants to fire
and cannot (`agent N dry (...)` with item, ammo, gameplay tags and abilities), forces a reload once, and cancels stuck
input abilities; `dry=` in the `train:` line counts such bots (0 in a healthy run).

### Observation (32 floats, `ManipleLyraSchema.h`)

| slot | content |
|---|---|
| 0-2 | own velocity in the aim frame / 600 |
| 3 | health fraction |
| 4-21 | 3 nearest enemies: rel x, y, z / 2000, distance / 2000, visible (weapon capsule trace), health |
| 22 | aim pitch / 90 |
| 23 | constant 1 |
| 24-31 | 8 wall rays at eye height every 45 deg, hit distance / 1500 (1 = free) |

### Action (5 floats, continuous in [-1, 1])

move forward, move right, yaw rate (x120 deg/s), pitch rate (x60 deg/s), fire (> 0).

### Reward (per decision step)

kill +1, death -1, team kill -1, damage dealt +dmg/100, damage taken -0.5 dmg/100, living cost -0.002,
aim shaping +0.01 * (0.5 (1 - angle / 90 deg) + 0.5 exp(-(angle / 6 deg)^2)) to the nearest visible enemy,
+0.03 per shot actually fired (magazine went down) with a visible enemy within 5 deg (half while moving), -0.005 per shot, approach shaping +0.5 * (distance closed to the
nearest enemy / 20 m, only beyond 10 m) in curriculum stages 0-1. Damage and kills come from Lyra's `Lyra.Damage.Message` /
`Lyra.Elimination.Message` gameplay messages. Episodes end on death or after 60 s of game time; spare ammo is topped
up to 60 rounds at every episode start (a pawn that never dies would otherwise run dry for good).

### Curriculum

`-ManipleCurriculum=auto` (train default) starts at stage 0. After at least 30 game-minutes in a stage it advances
when the policy's kills per game-minute over the last 5 game-minutes pass the stage threshold, and goes back one
stage when the rate drops below a quarter of the previous stage's threshold. 30 % of placements replay a random
earlier stage so early skills do not decay. `-ManipleCurriculum=N` pins a stage, `off` disables it.

| stage | start position | opponents | shaping | promote at |
|---|---|---|---|---|
| 0 duel | teleported 5-10 m from a random enemy, both facing, line of sight | self-play | approach | 2.0 kills/min |
| 1 close | 8-25 m, random facing | self-play | approach | 1.5 |
| 2 mid | random reachable point within 40 m of an enemy | self-play | none | 1.0 |
| 3 full | Lyra's spawn points | self-play | none | final |

The schedule is self-play throughout: the built-in heuristic (perfect aim) is far above an early policy and stalls
learning when mixed in. Use it as an opponent deliberately with `-ManipleOpponents=mixed|heuristic` (heuristic bots
never send transitions), e.g. in infer mode as a benchmark for a trained policy. `-ManipleEndless=1` (train default) raises Lyra's kill and time limits so the
match never restarts.

### Launch arguments

```
-ManipleBrain=none|random|heuristic|triton   none = Lyra bots untouched (default)
                                             random = random actions; heuristic = turn to the nearest visible enemy and shoot
                                             triton = actions from the policy
-ManipleMode=infer|train                     train = act on the trainer's latest weights with exploration, send transitions back
-ManipleModel=lyra                           policy name
-ManipleTritonUrl=localhost:8001
-ManipleChannel=best|stable|latest|<n>       infer mode only (train always uses latest)
-ManipleExplore=0|1                          default 1 in train, 0 in infer
-ManipleBots=all|N                           how many bots to take over
-ManipleSpawnBots=N                          extra bots
-ManipleHz=15                                decision rate, in game time (15 in train, 10 in infer)
-ManipleNet=auto|small|medium|large          network preset for the first registration
-ManipleHidden=256,256                       custom layers
-ManipleActivation=tanh|relu|elu|gelu
-ManipleLayerNorm=0|1
-ManipleEntropy=0.01                         PPO entropy bonus, first registration only
-ManipleLogStd=-1.0                          initial action noise (log std), first registration only
-ManipleTimeScale=X                          world time dilation (windowed runs)
-ManipleSpectate                             local player follows a bot; add ?SpectatorOnly=1 to the map URL
-ManipleCurriculum=auto|off|N                see Curriculum
-ManipleOpponents=stage|self|mixed|heuristic|lyra   lyra = the other team keeps Lyra's behaviour trees (benchmark)
-ManipleEndless=0|1                          no score / time limit (train default 1)
-ManipleSync=0|1                             wait for the policy reply inside the tick, deterministic timing (train default 1)
-ManipleScore=kills|stat:<Tag>|team:<Tag>    what "score" means in this game mode, see Score and evaluation
-ManipleEval=N                               evaluation: N game-seconds after the warmup, one "eval:" line, quit
-ManipleEvalReport=0|1                       also send the evaluation score to the trainer (default 0)
```

Faster than real time: run headless (`-nullrhi -unattended -nosound`) with `-benchmark -fps=30`; the game steps
fixed 33 ms frames as fast as the CPU allows (about 9x real time with 11 bots on one core).

### Score and evaluation

The plugin turns the game mode into one number, the **score**, with `-ManipleScore`:

| `-ManipleScore` | number | game modes |
|---|---|---|
| `kills` (default) | policy kills per agent-minute, from `Lyra.Elimination.Message` | elimination, deathmatch |
| `stat:<Tag>` | a player-state stat tag summed over the policy bots, per agent-minute, e.g. `stat:ShooterGame.Score.ControlPointCapture` | anything that counts per player |
| `team:<Tag>` | a team tag stack, our team minus the best other team, per game-minute, e.g. `team:ShooterGame.ControlPoint.TeamScore` | team objectives |

Time only counts while a bot can take damage (Lyra's warmup immunity is excluded), and only bots in the final
curriculum stage or on Lyra's own spawns count, so the score always means "in the real game".

- **Training**: every 5 game-minutes the subsystem sends the score with `report` and the policy is registered
  with `versioning.score = "report"`, so `best` is the exported version with the highest game score, not the
  highest shaped return. Earlier curriculum stages send nothing; the first export happens once the policy plays
  the final stage. The `train:` line shows the window's `score=` and its `score_min=` (agent-minutes behind it).
- **Evaluation**: `-ManipleEval=300` waits for the warmup, runs 300 game-seconds, prints
  `eval: mode=.. version=.. channel=.. opponents=.. stage=.. score=.. value=.. agents=.. game_min=.. agent_min=.. kills=.. deaths=.. hits=.. shots=..`
  and quits. Works with every brain, so the heuristic and random bots give reference values. The standard
  benchmark is infer mode against Lyra's own bots: `-ManipleMode=infer -ManipleChannel=<best|stable|n>
  -ManipleOpponents=lyra -ManipleEval=300`. `-ManipleEvalReport=1` stores the value as that version's eval score
  on the trainer.

### Training

```
UnrealEditor <Project>.uproject /ShooterMaps/Maps/L_Expanse -game -Experience=B_ShooterGame_Elimination \
  -nullrhi -unattended -nosound -benchmark -fps=30 \
  -ManipleBrain=triton -ManipleMode=train -ManipleModel=lyra -ManipleBots=all -ManipleSpawnBots=8
```

Log lines: `registered policy ...`, then every 5 s (game time) `bench: ...` (latency, served version) and
`train: stage=.. rows=.. reward_sum=.. fire=.. shots=.. hits=.. episodes=.. placed=.. kills=.. heuristic_kills=.. deaths=.. kills_per_min=.. score=.. score_min=.. dry=.. trainer=[.. std=..]`,
every 5 game-minutes `report: version=.. score=..`
(fire = fraction of decisions with the fire action on, shots = rounds that left the gun, hits = damage events dealt).
`-LogCmds="LogManipleLyra Verbose"` adds one line per hit and per death.

Then play against it with `-ManipleMode=infer -ManipleChannel=best` (needs an exported version: the trainer
exports on score improvement, or `gym/play.py --export` / `--promote`), or watch training in a window with
`?SpectatorOnly=1 -ManipleSpectate -ManipleTimeScale=3`.
