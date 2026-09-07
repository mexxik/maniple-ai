# gym — train and play Gymnasium environments through Triton

Proves the train/infer loop without a game engine. The client only steps environments and moves tensors;
PPO runs inside Triton (`triton/model_repository/ppo_train`), actions come from `ppo_infer`.

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

    cd ../triton && docker compose up -d && cd ../gym     # server on localhost:8001 (gRPC)

## Train

    # discrete actions, learns in ~30 s
    .venv/bin/python run.py --env CartPole-v1 --name cartpole --agents 16 --steps 8000

    # continuous actions
    .venv/bin/python run.py --env Pendulum-v1 --name pendulum --agents 16 --steps 20000

    # network presets and options (LunarLander needs: .venv/bin/pip install "gymnasium[box2d]")
    .venv/bin/python run.py --env LunarLander-v3 --name lander --agents 32 --steps 50000 --net medium --rollout 8192
    .venv/bin/python run.py --env LunarLander-v3 --name lander2 --hidden 512,512,256 --activation relu --layernorm

    # keep the cart near the centre: quadratic reward penalty, 0 at the centre, --center-penalty at the rail
    .venv/bin/python run.py --env CartPole-v1 --name cartpole_centered --steps 20000 --center-penalty 0.1

    # keep training an existing policy with the spec it was created with (network/PPO flags are ignored)
    .venv/bin/python run.py --env CartPole-v1 --name cartpole --steps 4000 --resume

    # remote server
    .venv/bin/python run.py --url gpubox:8001 --env CartPole-v1 --name cartpole

Options (`run.py --help` lists them all):

| group | flags |
|---|---|
| run | `--agents` parallel envs, `--steps` per env, `--no-explore` greedy actions, `--center-penalty` (CartPole) |
| network | `--net auto\|small\|medium\|large\|custom`, `--hidden 512,512` (custom), `--activation tanh\|relu\|elu\|gelu`, `--layernorm`, `--shared-critic`, `--no-normalize-obs` |
| ppo | `--rollout`, `--epochs`, `--minibatch`, `--lr`, `--gamma`, `--entropy`, `--max-lag` |

`--net auto` (default) sizes two hidden layers from the observation size; presets are 64x64 / 256x256 / 512x512x256.
A policy name keeps the spec it was created with: changing network or PPO options for an existing `--name` is
rejected, use a new name (or delete the policy, see below). The resolved network is printed at start and is in
every `status`.

`--center-penalty` shapes the reward on the client, before it is sent to the trainer: the cost is
`coef * (position / 2.4)^2`, so at `0.1` a cart parked at the rail earns 0.9 per step instead of 1.0. Reported
returns stay on the true environment reward, so they remain comparable to unshaped runs.

## Play

    # live window on your display, greedy policy
    DISPLAY=:0 .venv/bin/python play.py --env CartPole-v1 --name cartpole_centered --episodes 5

    # headless, write an mp4
    .venv/bin/python play.py --env CartPole-v1 --name cartpole --episodes 3 --video cartpole.mp4

    # scores only
    .venv/bin/python play.py --env CartPole-v1 --name cartpole --episodes 20 --render none

    # stochastic actions (as during training), slower playback
    DISPLAY=:0 .venv/bin/python play.py --env Pendulum-v1 --name pendulum --explore --fps 30

## Versions and channels

Every PPO update exports a new version of `<name>_policy`. Clients never pick numbers, they pick a channel:

| channel | meaning |
|---|---|
| `latest` | newest export; what `run.py` uses while training |
| `best` | highest-scoring version with at least `min_episodes` training episodes, or a reported eval score; default for `play.py` |
| `stable` | manually promoted; never moves on its own |

    .venv/bin/python play.py --name cartpole --episodes 20 --render none --report     # greedy eval, score fed back -> 'best'
    .venv/bin/python play.py --name cartpole --channel latest --episodes 3            # watch the newest one
    .venv/bin/python play.py --name cartpole --channel 97 --episodes 3 --promote      # pin 'stable' to v97
    cat ../triton/model_repository/cartpole_policy/versions.json                       # the manifest: channels + scores

Retention: the last `keep_latest` versions plus `best` and `stable` stay on disk; everything else is pruned.

## Inspect the server

    curl -s -X POST localhost:8000/v2/repository/index                         # models and versions
    ls ../triton/model_repository/cartpole_policy/                              # exported versions (last 3 kept)
    ls ../triton/checkpoints/cartpole/                                          # trainer state + spec
    docker compose -f ../triton/docker-compose.yml logs -f triton | grep -v "HTTP request"

Each policy name gets its own `<name>_policy` model in the repository and its own checkpoint. To start a policy
from scratch, delete both directories (`triton/model_repository/<name>_policy`, `triton/checkpoints/<name>`)
and restart the server.
