# gym — train and play Gymnasium environments through Triton

Proves the train/infer loop without a game engine. The client only steps environments and moves tensors;
PPO runs inside Triton (`triton/model_repository/ppo_train`), actions come from `ppo_infer`.

## Setup

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt

    cd ../triton && docker compose up -d && cd ../gym     # server on localhost:8001 (gRPC)

## Train

    # discrete actions, learns in ~30 s
    .venv/bin/python run.py --env CartPole-v1 --name cartpole --agents 16 --steps 8000

    # continuous actions
    .venv/bin/python run.py --env Pendulum-v1 --name pendulum --agents 16 --steps 20000

    # bigger net, larger rollout per update (LunarLander needs: .venv/bin/pip install "gymnasium[box2d]")
    .venv/bin/python run.py --env LunarLander-v3 --name lander --agents 32 --steps 50000 --hidden 128,128 --rollout 8192

    # keep training an existing policy: same --name resumes from the checkpoint and the last exported version
    .venv/bin/python run.py --env CartPole-v1 --name cartpole --steps 4000

    # remote server
    .venv/bin/python run.py --url gpubox:8001 --env CartPole-v1 --name cartpole

Options: `--agents` parallel envs, `--steps` env steps per env, `--rollout` transitions per PPO update,
`--hidden` layer sizes, `--no-explore` greedy actions (for evaluation runs only).

## Play

    # live window on your display, greedy policy
    DISPLAY=:0 .venv/bin/python play.py --env Pendulum-v1 --name pendulum --episodes 5

    # headless, write an mp4
    .venv/bin/python play.py --env CartPole-v1 --name cartpole --episodes 3 --video cartpole.mp4

    # scores only
    .venv/bin/python play.py --env CartPole-v1 --name cartpole --episodes 20 --render none

    # stochastic actions (as during training), slower playback
    DISPLAY=:0 .venv/bin/python play.py --env Pendulum-v1 --name pendulum --explore --fps 30

## Inspect the server

    curl -s -X POST localhost:8000/v2/repository/index                         # models and versions
    ls ../triton/model_repository/cartpole_policy/                              # exported versions (last 3 kept)
    ls ../triton/checkpoints/cartpole/                                          # trainer state + spec
    docker compose -f ../triton/docker-compose.yml logs -f triton | grep -v "HTTP request"

Each policy name gets its own `<name>_policy` model in the repository and its own checkpoint. To start a policy
from scratch, delete both directories (`triton/model_repository/<name>_policy`, `triton/checkpoints/<name>`)
and restart the server.
