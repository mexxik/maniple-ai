# Triton model server

    docker compose up -d                       # builds maniple-triton (tritonserver + torch) on first run
    curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/v2/health/ready     # 200
    curl -s -X POST localhost:8000/v2/repository/index
    docker compose logs -f triton
    docker compose down

## Models
| model | what |
|---|---|
| `ppo_train` | the algorithm (Python backend, `common/maniple`). `register` a named policy with an `AgentSpec`, `observe` transitions; trains in the background, exports every update |
| `ppo_infer` | inference entry point: `name` + `obs` (+ `explore`) → `action`, `action_index`, `logp`, `policy_version`. BLS to `<name>_policy` |
| `<name>_policy` | static ONNX exported by the trainer (`obs` → `action` [, `log_std`], `policy_version`). Hot path if you want to skip `ppo_infer` |
| `lyra_policy` | placeholder for the Lyra demo (see python/export_policy.py in the dev repo) |

Protocol details: docstrings in `common/maniple/triton_model.py` and `common/maniple/infer_model.py`.
A new algorithm = `common/maniple/algorithms/<x>.py` + `model_repository/<x>_train/1/model.py` (subclass, 5 lines).

## Gotchas
- `common/` is mounted at `/common`; Triton only reloads a model when files under its own directory change, so after
  editing `common/` run `docker compose restart triton`.
- The repository is polled every 1 s; the trainer writes `<name>_policy/<version>` atomically and keeps the last 3.
- The container runs as your uid (`user:` in compose) so exported models and checkpoints stay deletable.
- Model artifacts (`*.onnx`, checkpoints) are gitignored.
