"""Train a Gymnasium environment through Triton.

The loop is: register the policy once, then for every step
    ppo_infer  -> actions for all environments
    env.step
    ppo_train  <- the transitions
No ML code runs here; PPO lives in Triton (triton/common/maniple).

    .venv/bin/python run.py --env CartPole-v1 --name cartpole --agents 16 --steps 8000
    .venv/bin/python run.py --env Pendulum-v1 --name pendulum --agents 16 --steps 20000
    .venv/bin/python run.py --env ALE/Pong-v5 --name pong --agents 16 --steps 60000 --net none   # pixels

Observations become the policy's inputs (a vector -> "obs", an image -> "frame", see envs.py); the action
space becomes one action group, or one per dimension with --groups per-dim.
"""

import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
from envs import make_env, spec_actions, spec_inputs, to_env_action, to_inputs
from triton_agent import TritonAgent

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "triton", "common"))
from maniple.recording import RecordingWriter  # noqa: E402  (numpy only)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium environment id (ALE/... = Atari)")
    parser.add_argument("--name", default="cartpole", help="policy name on the server")
    parser.add_argument("--agents", type=int, default=16, help="parallel environments")
    parser.add_argument("--steps", type=int, default=3000, help="environment steps per agent")
    parser.add_argument("--no-explore", action="store_true", help="act greedily (evaluation only)")
    parser.add_argument(
        "--async-envs", action="store_true", help="step the environments in worker processes (Atari)"
    )
    parser.add_argument(
        "--groups",
        default="single",
        choices=["single", "per-dim"],
        help="continuous actions as one group, or one group per dimension (several heads)",
    )
    parser.add_argument(
        "--center-penalty",
        type=float,
        default=0.0,
        help="CartPole only: subtract this much reward when the cart sits at the rail (0 = off)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue an existing policy with its stored spec"
    )
    parser.add_argument(
        "--record",
        default=None,
        help="write every transition to this directory (tools/recording.py reads it)",
    )

    net = parser.add_argument_group("network (see triton/common/maniple/spec.py)")
    net.add_argument("--net", default="auto", help="auto | small | medium | large | custom | none")
    net.add_argument("--hidden", default=None, help="layer sizes for --net custom, e.g. 512,512,256")
    net.add_argument("--activation", default="tanh", choices=["tanh", "relu", "elu", "gelu"])
    net.add_argument("--layernorm", action="store_true", help="LayerNorm after every hidden layer")
    net.add_argument("--shared-critic", action="store_true", help="value head on the actor torso")
    net.add_argument(
        "--no-normalize-obs", action="store_true", help="disable running observation normalisation"
    )

    ppo = parser.add_argument_group("ppo")
    ppo.add_argument("--rollout", type=int, default=2048, help="transitions per update")
    ppo.add_argument("--epochs", type=int, default=4)
    ppo.add_argument("--minibatch", type=int, default=256)
    ppo.add_argument("--lr", type=float, default=3e-4)
    ppo.add_argument("--gamma", type=float, default=0.99)
    ppo.add_argument("--entropy", type=float, default=0.01, help="entropy bonus coefficient")
    ppo.add_argument("--max-lag", type=int, default=4, help="drop rows older than this many policy versions")
    return parser.parse_args()


def make_spec(envs, args):
    """Build the AgentSpec the trainer needs from the environment's spaces and the CLI options."""
    net = {
        "preset": "custom" if args.hidden else args.net,
        "hidden": [int(h) for h in args.hidden.split(",")] if args.hidden else [],
        "activation": args.activation,
        "layernorm": args.layernorm,
        "separate_critic": not args.shared_critic,
        "normalize_obs": not args.no_normalize_obs,
    }
    ppo = {
        "rollout": args.rollout,
        "epochs": args.epochs,
        "minibatch": args.minibatch,
        "lr": args.lr,
        "gamma": args.gamma,
        "entropy_coef": args.entropy,
        "max_policy_lag": args.max_lag,
    }
    return {
        "inputs": spec_inputs(envs.single_observation_space),
        "actions": spec_actions(envs.single_action_space, args.groups),
        "net": net,
        "ppo": ppo,
    }


def center_penalty(obs, coef, threshold):
    """Quadratic cost on how far the cart has drifted: 0 at the centre, `coef` at the rail."""
    return coef * (obs[:, 0] / threshold) ** 2


def main():
    args = parse_args()

    factories = [lambda: make_env(args.env) for _ in range(args.agents)]
    envs = gym.vector.AsyncVectorEnv(factories) if args.async_envs else gym.vector.SyncVectorEnv(factories)
    agent = TritonAgent(args.url, args.name)
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space

    # the rail the cart is allowed to reach before the episode ends (CartPole: 2.4)
    x_threshold = 2.4
    if not args.async_envs:
        x_threshold = float(getattr(envs.envs[0].unwrapped, "x_threshold", 2.4))

    spec = make_spec(envs, args)
    if args.resume:
        stored = agent.stored_spec()
        if stored is None:
            print(f"no policy '{args.name}' on the server yet, creating it")
        else:
            spec = stored  # network and PPO flags are ignored on resume
    status = agent.register(spec)
    print(
        f"policy '{args.name}' v{status['version']} ({status['updates']} updates so far), net: {status['net']}"
    )

    # per-environment bookkeeping the trainer needs to stitch trajectories
    agent_ids = np.arange(args.agents, dtype=np.int64)
    episode_ids = np.zeros(args.agents, dtype=np.int64)

    # for the progress line
    episode_returns = np.zeros(args.agents, dtype=np.float32)
    finished_returns = []
    started = time.time()

    recorder = None
    if args.record:
        meta = {"game": args.env, "policy": args.name, "spec": spec, "kinds": {"0": "policy"}}
        recorder = RecordingWriter(args.record, meta)
        print(f"recording to {args.record}")

    obs, _ = envs.reset(seed=0)

    try:
        for step in range(1, args.steps + 1):
            inputs = to_inputs(obs, obs_space)

            action, action_index, logp, served_version = agent.act(
                inputs, explore=not args.no_explore, channel="latest"
            )
            next_obs, reward, terminated, truncated, _ = envs.step(
                to_env_action(action, action_index, act_space)
            )
            done = np.logical_or(terminated, truncated)

            # the trainer learns from the shaped reward; the progress line below stays on the true one
            shaped_reward = reward
            if args.center_penalty:
                position = next_obs.reshape(args.agents, -1)
                shaped_reward = reward - center_penalty(position, args.center_penalty, x_threshold)

            status = agent.observe(
                inputs=inputs,
                action=action,
                reward=shaped_reward,
                done=done,
                agent_id=agent_ids,
                episode_id=episode_ids,
                policy_version=np.full(args.agents, served_version),
                logp=logp,
            )
            if recorder:
                recorder.append(
                    **inputs,
                    action=action.astype(np.float32),
                    reward=shaped_reward.astype(np.float32),
                    done=done.astype(bool),
                    logp=logp.astype(np.float32),
                    policy_version=np.full(args.agents, served_version, dtype=np.int64),
                    agent_id=agent_ids,
                    episode_id=episode_ids.copy(),
                    time=np.full(args.agents, time.time() - started, dtype=np.float32),
                    kind=np.zeros(args.agents, dtype=np.uint8),
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
                elapsed = time.time() - started
                print(
                    f"step {step:5d}  served v{served_version} trained v{status['version']}  "
                    f"updates {status['updates']}  buffered {status['buffered']:5d}  stale {status['dropped_stale']}  "
                    f"episodes {len(finished_returns):4d}  mean return(last 20) {mean_return:8.2f}  "
                    f"kl {status['stats'].get('kl', 0):.4f}  {step * args.agents / elapsed:6.0f} steps/s  {elapsed:5.0f}s"
                )

    finally:
        if recorder:
            recorder.close()
            print(f"recorded {recorder.rows} rows to {args.record}")


if __name__ == "__main__":
    main()
