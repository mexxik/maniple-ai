"""Train a Gymnasium environment through Triton.

The loop is: register the policy once, then for every step
    ppo_infer  -> actions for all environments
    env.step
    ppo_train  <- the transitions
No ML code runs here; PPO lives in Triton (triton/common/maniple).

    .venv/bin/python run.py --env CartPole-v1 --name cartpole --agents 16 --steps 8000
    .venv/bin/python run.py --env Pendulum-v1 --name pendulum --agents 16 --steps 20000
"""

import argparse
import time

import gymnasium as gym
import numpy as np
from triton_agent import TritonAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium environment id")
    parser.add_argument("--name", default="cartpole", help="policy name on the server")
    parser.add_argument("--agents", type=int, default=16, help="parallel environments")
    parser.add_argument("--steps", type=int, default=3000, help="environment steps per agent")
    parser.add_argument("--rollout", type=int, default=2048, help="transitions per PPO update")
    parser.add_argument("--hidden", default="64,64", help="hidden layer sizes")
    parser.add_argument("--no-explore", action="store_true", help="act greedily (evaluation only)")
    return parser.parse_args()


def make_spec(envs, hidden, rollout):
    """Build the AgentSpec the trainer needs from the environment's spaces."""
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    space = envs.single_action_space

    if isinstance(space, gym.spaces.Discrete):
        action = {"type": "discrete", "n": int(space.n)}
    else:
        action = {
            "type": "continuous",
            "dim": int(np.prod(space.shape)),
            "low": float(space.low.min()),
            "high": float(space.high.max()),
        }

    ppo = {
        "rollout": rollout,
        "epochs": 4,
        "minibatch": 256,
        "lr": 3e-4,
        "entropy_coef": 0.01,
        "max_policy_lag": 4,
    }
    return {"obs": {"dim": obs_dim}, "action": action, "hidden": hidden, "ppo": ppo}


def to_env_action(action, action_index, action_space):
    """Translate the server's action tensor into what env.step() expects."""
    if action_space["type"] == "discrete":
        return action_index
    return np.clip(action, action_space["low"], action_space["high"])


def main():
    args = parse_args()

    envs = gym.vector.SyncVectorEnv([lambda: gym.make(args.env) for _ in range(args.agents)])
    hidden = [int(h) for h in args.hidden.split(",")]
    spec = make_spec(envs, hidden, args.rollout)

    agent = TritonAgent(args.url, args.name)
    print("register:", agent.register(spec))
    agent.wait_until_ready()

    # per-environment bookkeeping the trainer needs to stitch trajectories
    agent_ids = np.arange(args.agents, dtype=np.int64)
    episode_ids = np.zeros(args.agents, dtype=np.int64)

    # for the progress line
    episode_returns = np.zeros(args.agents, dtype=np.float32)
    finished_returns = []
    started = time.time()

    obs, _ = envs.reset(seed=0)

    for step in range(1, args.steps + 1):
        obs_batch = obs.reshape(args.agents, -1).astype(np.float32)

        action, action_index, logp, served_version = agent.act(obs_batch, explore=not args.no_explore)
        next_obs, reward, terminated, truncated, _ = envs.step(
            to_env_action(action, action_index, spec["action"])
        )
        done = np.logical_or(terminated, truncated)

        status = agent.observe(
            obs=obs_batch,
            action=action,
            reward=reward,
            done=done,
            agent_id=agent_ids,
            episode_id=episode_ids,
            policy_version=np.full(args.agents, served_version),
            logp=logp,
        )

        # episode accounting
        episode_returns += reward
        for i in np.where(done)[0]:
            finished_returns.append(float(episode_returns[i]))
            episode_returns[i] = 0.0
            episode_ids[i] += 1

        obs = next_obs

        if step % 100 == 0:
            recent = finished_returns[-20:]
            mean_return = np.mean(recent) if recent else float("nan")
            print(
                f"step {step:5d}  served v{served_version} trained v{status['version']}  "
                f"updates {status['updates']}  buffered {status['buffered']:5d}  stale {status['dropped_stale']}  "
                f"episodes {len(finished_returns):4d}  mean return(last 20) {mean_return:8.2f}  "
                f"kl {status['stats'].get('kl', 0):.4f}  {time.time() - started:5.0f}s"
            )


if __name__ == "__main__":
    main()
