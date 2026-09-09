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
| `ppo_infer` | inference entry point: `name` + `obs` (+ `explore`, `channel`) → `action`, `action_index`, `logp`, `policy_version`. `latest` → `ppo_train`; `best`/`stable`/`<n>` → exported models, loaded on first use, unloaded when idle |
| `policy_<name>` | ONNX exports of a policy (several versions), manifest `versions.json` next to it. `best` is ranked by the mean training return, or by the client's `report` scores when the spec says `versioning.score = "report"` (a game-defined number such as kills per agent-minute) |
| `policy_<name>_trt` | TensorRT engine built on `promote` for the `stable` version |

The server runs in explicit model-control mode: only `ppo_train` and `ppo_infer` load at start.
Protocol details: docstrings in `common/maniple/triton_model.py` and `common/maniple/infer_model.py`.
A new algorithm = `common/maniple/algorithms/<x>.py` + `model_repository/<x>_train/1/model.py` (subclass, 5 lines).

## Gotchas
- `common/` is mounted at `/common`; Triton only reloads a model when files under its own directory change, so after
  editing `common/` run `docker compose restart triton`.
- Explicit model control: nothing is polled; `ppo_infer` loads exported models on demand.
- The container runs as your uid (`user:` in compose) so exported models and checkpoints stay deletable.
- Model artifacts (`*.onnx`, checkpoints) are gitignored.
