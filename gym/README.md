# gym — train and play Gymnasium environments through Triton

Proves the train/infer loop without a game engine. The client only steps environments and moves tensors;
the algorithm runs inside Triton (`triton/model_repository/<algo>_train`), actions come from `<algo>_infer`.

## Setup

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install "gymnasium[box2d]"              # optional: LunarLander, BipedalWalker
    # requirements.txt includes gymnasium[atari] (ale-py with ROMs) for the pixel environments (ALE/Pong-v5, ...)

    cd ../triton && docker compose up -d && cd ../gym     # server on localhost:8001 (gRPC)

## Algorithms

| file | algorithm | server models |
|---|---|---|
| [PPO.md](PPO.md) | PPO, on-policy, discrete + continuous | `ppo_train`, `ppo_infer` |

Each file has one section per environment with the commands to train, evaluate, promote and play it,
plus the settings that are known to work.

## Concepts (apply to every algorithm)

- **Policy name** (`--name`): one trained agent on the server, with its own spec, checkpoint and versions.
  A name keeps the spec it was created with; different network/PPO flags need a new name, `--resume` reuses
  the stored spec.
- **Version** = the trainer's update count.
- **Channels**: `latest` (trainer's current weights, what `run.py` uses), `best` (highest-scoring export),
  `stable` (promoted by you; default for `play.py`; gets a TensorRT engine).
- **Exports** happen on events only: training score beats `best`, `play.py --export`, `--promote`, or
  `command=export`. Exported models are loaded by the server on first request and unloaded when idle.
- **Manifest**: `../triton/model_repository/policy_<name>/versions.json` — channels, exported versions, scores, TRT state.
- **Inputs and action groups**: an environment's observation becomes a named input (`obs` for vectors, `frame`
  for images, see `envs.py`) and its action space one action group (`--groups per-dim` splits a continuous
  space into one group per dimension). Actions come back as one flat row plus the index per discrete group.

## Scripts

| script | purpose |
|---|---|
| `run.py` | train: register → loop { act on `latest`, env.step, observe }; `--help` lists network and PPO flags |
| `play.py` | play a channel or version greedily; `--render human\|rgb\|none`, `--video`, `--report`, `--promote`, `--export` |
| `triton_agent.py` | the client class both use (`register`, `act`, `observe`, `report`, `promote`, `export`, `status`) |
| `envs.py` | environment ↔ spec glue: which input an observation space becomes, Atari preprocessing, action groups, `env.step` translation |

## Inspect the server

    curl -s -X POST localhost:8000/v2/repository/index                         # models and what is loaded
    ls ../triton/model_repository/policy_<name>/                                # exported versions
    ls ../triton/checkpoints/<name>/                                            # trainer state + spec
    docker compose -f ../triton/docker-compose.yml logs -f triton | grep -v "HTTP request"

To start one policy from scratch, delete `triton/model_repository/policy_<name>*` and `triton/checkpoints/<name>`
and restart the server. To wipe everything: `cd ../triton && docker compose run --rm reset && docker compose restart triton`.
