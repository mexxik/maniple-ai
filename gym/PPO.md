# PPO

Server models `ppo_train` / `ppo_infer`. All commands run from this directory with the venv from the README.
Flags not shown keep their defaults (`run.py --help`). Times are for one RTX 3090 with 16 parallel envs.

Common flow for any environment:

    .venv/bin/python run.py  --env <Env> --name <name> [network/ppo flags]   # train (latest channel, no exports)
    .venv/bin/python play.py --env <Env> --name <name> --channel latest --episodes 3   # watch the live policy
    .venv/bin/python play.py --env <Env> --name <name> --channel latest --episodes 5 --promote   # export + stable + TRT
    .venv/bin/python play.py --env <Env> --name <name> --episodes 20 --render none --report      # greedy eval of stable
    .venv/bin/python play.py --env <Env> --name <name> --episodes 5                              # play stable (window)
    .venv/bin/python run.py  --env <Env> --name <name> --resume --steps <n>                       # continue training

`DISPLAY=:0` in front of `play.py` for a window on this machine; `--video out.mp4` for a headless recording.

---

## CartPole-v1 (discrete, 4 obs, 2 actions)

Solved at return 500. Learns in ~20 s.

    .venv/bin/python run.py  --env CartPole-v1 --name cartpole --agents 16 --steps 6000
    .venv/bin/python play.py --env CartPole-v1 --name cartpole --channel latest --episodes 5 --promote
    .venv/bin/python play.py --env CartPole-v1 --name cartpole --episodes 20 --render none --report
    DISPLAY=:0 .venv/bin/python play.py --env CartPole-v1 --name cartpole --episodes 3

Variants worth comparing (new names):

    .venv/bin/python run.py --env CartPole-v1 --name cartpole_relu --activation relu --layernorm --steps 6000
    .venv/bin/python run.py --env CartPole-v1 --name cartpole_big --net large --steps 6000

## Acrobot-v1 (discrete, 6 obs, 3 actions)

Return is negative (−steps to swing up); good policies reach about −80. Needs more samples than CartPole.

    .venv/bin/python run.py  --env Acrobot-v1 --name acrobot --agents 16 --steps 20000 --rollout 4096
    .venv/bin/python play.py --env Acrobot-v1 --name acrobot --channel latest --episodes 5 --promote
    DISPLAY=:0 .venv/bin/python play.py --env Acrobot-v1 --name acrobot --episodes 3

## MountainCar-v0 (discrete, 2 obs, 3 actions)

Hard exploration: reward is −1 per step until the goal. Plain PPO often never sees the goal; use a larger
entropy bonus and long runs, or treat it as a stress test.

    .venv/bin/python run.py  --env MountainCar-v0 --name mcar --agents 32 --steps 50000 --rollout 8192 --entropy 0.05
    .venv/bin/python play.py --env MountainCar-v0 --name mcar --channel latest --episodes 3

## Pendulum-v1 (continuous, 3 obs, 1 action in [−2, 2])

Return per episode is in [−1600, 0]; a good policy stays above −200. Continuous actions exercise the
`log_std` path and the `--net` options matter here.

    .venv/bin/python run.py  --env Pendulum-v1 --name pendulum --agents 16 --steps 30000 --rollout 4096 --lr 1e-4 --gamma 0.95
    .venv/bin/python play.py --env Pendulum-v1 --name pendulum --channel latest --episodes 5 --promote
    .venv/bin/python play.py --env Pendulum-v1 --name pendulum --episodes 10 --render none --report
    DISPLAY=:0 .venv/bin/python play.py --env Pendulum-v1 --name pendulum --episodes 3 --fps 30

## LunarLander-v3 (discrete, 8 obs, 4 actions) — needs `gymnasium[box2d]`

Solved at return 200 averaged over 100 episodes. Expect a few minutes.

    .venv/bin/python run.py  --env LunarLander-v3 --name lander --agents 32 --steps 60000 --rollout 8192 --net medium
    .venv/bin/python play.py --env LunarLander-v3 --name lander --channel latest --episodes 5 --promote
    .venv/bin/python play.py --env LunarLander-v3 --name lander --episodes 20 --render none --report
    DISPLAY=:0 .venv/bin/python play.py --env LunarLander-v3 --name lander --episodes 3

Continuous variant:

    .venv/bin/python run.py  --env LunarLanderContinuous-v3 --name lander_c --agents 32 --steps 80000 --rollout 8192 --net medium --lr 1e-4

## BipedalWalker-v3 (continuous, 24 obs, 4 actions) — needs `gymnasium[box2d]`

Long-horizon locomotion; the first real test of network size and normalisation. Solved at 300.

    .venv/bin/python run.py  --env BipedalWalker-v3 --name walker --agents 32 --steps 200000 --rollout 16384 --net large --activation relu --lr 1e-4 --entropy 0.0
    .venv/bin/python play.py --env BipedalWalker-v3 --name walker --channel latest --episodes 3 --promote
    DISPLAY=:0 .venv/bin/python play.py --env BipedalWalker-v3 --name walker --episodes 2

---

## Tuning notes

- `--rollout` is transitions per PPO update across all agents; bigger = steadier gradients, fewer versions.
- `--agents` multiplies throughput; each step is one `act` and one `observe` call for the whole batch.
- `--entropy` keeps exploration alive on sparse rewards; lower it for continuous control once learning starts.
- `--net auto` sizes two layers from the observation size; use presets or `--hidden` for anything image-like.
- Observation normalisation is on by default (`--no-normalize-obs` to disable); it is baked into exports.
- Nothing is exported while training unless the score improves; promote what you like from `latest`.
