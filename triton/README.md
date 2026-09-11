# Triton model server

    docker compose up -d                       # builds maniple-triton (tritonserver + torch) on first run
    curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/v2/health/ready     # 200
    curl -s -X POST localhost:8000/v2/repository/index
    docker compose logs -f triton
    docker compose down
    docker compose run --rm reset && docker compose restart triton   # wipe all policies + checkpoints

## Models
| model | what |
|---|---|
| `ppo_train` | the algorithm (Python backend, `common/maniple`). `register` a named policy with an `AgentSpec`, `observe` transitions, `act` = the `latest` channel from the current weights; trains in the background; exports on events (`export`, `promote`, score improvement) |
| `ppo_infer` | inference entry point: `name` + the policy's inputs (+ `explore`, `channel`) → `action`, `action_index`, `logp`, `policy_version`. `latest` → `ppo_train`; `best`/`stable`/`<n>` → exported models, loaded on first use, unloaded when idle |
| `policy_<name>` | ONNX exports of a policy (several versions), manifest `versions.json` next to it. `best` is ranked by the mean training return, or by the client's `report` scores when the spec says `versioning.score = "report"` (a game-defined number such as kills per agent-minute) |
| `policy_<name>_trt` | TensorRT engine built on `promote` for the `stable` version |

The server runs in explicit model-control mode: only `ppo_train` and `ppo_infer` load at start.
Protocol details: docstrings in `common/maniple/triton_model.py` and `common/maniple/infer_model.py`.
A new algorithm = `common/maniple/algorithms/<x>.py` + `model_repository/<x>_train/1/model.py` (subclass, 5 lines).

## The spec: named inputs, action groups

A policy is described once, at `register`, by an `AgentSpec` (`common/maniple/spec.py`):

    {"inputs":  {"obs": {"shape": [32]}, "frame": {"shape": [84, 84, 4], "dtype": "uint8", "encoder": "cnn"}},
     "actions": {"move": {"type": "continuous", "dim": 4}, "fire": {"type": "discrete", "n": 2}},
     "net": {...}, "ppo": {...}, "versioning": {...}}

- Inputs are the observation tensors on the wire, all optional, fixed names: `obs` FP32 `[N, D]`, `frame` UINT8
  `[N, H, W, C]`, `audio` FP32 `[N, S]`, `text` STRING `[N]`. `obs` gets an `mlp` encoder (normalisation, optional
  layers), `frame` a `cnn` (Nature-DQN); `audio` and `text` are accepted on the wire but have no encoder yet.
- The encoder outputs are concatenated, go through the torso (`net.preset` / `net.hidden`, `none` = no torso)
  and into one head per action group.
- `action` is one flat row per agent, groups in spec order: continuous values, one-hot per discrete group.
  `action_index` is `[N, number of discrete groups]`, `logp` is summed over groups.
- The v1 shorthand `{"obs": {"dim": 32}, "action": {...}}` still works: one input `obs`, one group `action`.
  Checkpoints from before the generic pipe load unchanged (keys are migrated), old exports are served as v1.
- Each export writes `policy_<name>/spec.json` next to the model so `ppo_infer` knows the layout.

## Tests

    docker compose exec triton python -m unittest discover -s /common/tests -v

Spec parsing, networks, buffer, one PPO update, ONNX export and the v1 checkpoint migration; no game needed.

## Gotchas
- `common/` is mounted at `/common`; Triton only reloads a model when files under its own directory change, so after
  editing `common/` run `docker compose restart triton`.
- Explicit model control: nothing is polled; `ppo_infer` loads exported models on demand.
- The container runs as your uid (`user:` in compose) so exported models and checkpoints stay deletable.
- Model artifacts (`*.onnx`, checkpoints) are gitignored.
