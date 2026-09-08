"""Play a policy trained through Triton, with visualization. No training happens here.

    DISPLAY=:0 .venv/bin/python play.py --env CartPole-v1 --name cartpole --episodes 5           # live window
    .venv/bin/python play.py --env CartPole-v1 --name cartpole --episodes 3 --video cartpole.mp4   # headless mp4
    .venv/bin/python play.py --env CartPole-v1 --name cartpole --episodes 20 --render none        # scores only

Add --explore to sample actions like during training instead of acting greedily.
--channel best|latest|stable|<n> picks the version (default best); --report feeds the score back to the trainer;
--promote pins 'stable' to the version you just watched.
"""

import argparse
import time

import gymnasium as gym
import numpy as np
from triton_agent import TritonAgent

RENDER_MODES = {"human": "human", "rgb": "rgb_array", "none": None}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium environment id")
    parser.add_argument("--name", default="cartpole", help="policy name on the server")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--render", choices=list(RENDER_MODES), default=None, help="default: human, or rgb with --video"
    )
    parser.add_argument(
        "--video", default=None, help="write an mp4 (implies --render rgb, needs imageio[ffmpeg])"
    )
    parser.add_argument("--fps", type=int, default=50, help="playback / video frame rate")
    parser.add_argument("--explore", action="store_true", help="sample actions instead of greedy")
    parser.add_argument("--max-steps", type=int, default=1000, help="safety cap per episode")
    parser.add_argument("--channel", default="stable", help="stable | best | latest | <version number>")
    parser.add_argument(
        "--report", action="store_true", help="send the mean return to the trainer as an eval score"
    )
    parser.add_argument(
        "--promote", action="store_true", help="after playing, pin 'stable' to the version played"
    )
    parser.add_argument(
        "--export", action="store_true", help="before playing, export the trainer's current version"
    )
    return parser.parse_args()


def play_episode(env, agent, explore, max_steps, render, frames, fps, channel):
    """Run one episode. Returns (return, steps, served policy version)."""
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    obs, _ = env.reset()
    episode_return = 0.0
    version = 0

    for step in range(1, max_steps + 1):
        action, action_index, _, version = agent.act(obs[None], explore=explore, channel=channel)

        if discrete:
            env_action = int(action_index[0])
        else:
            env_action = np.clip(action[0], env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, _ = env.step(env_action)
        episode_return += float(reward)

        if render == "rgb":
            frames.append(env.render())
        elif render == "human":
            time.sleep(1.0 / fps)

        if terminated or truncated:
            return episode_return, step, version

    return episode_return, max_steps, version


def main():
    args = parse_args()
    render = args.render or ("rgb" if args.video else "human")

    env = gym.make(args.env, render_mode=RENDER_MODES[render])
    agent = TritonAgent(args.url, args.name)

    if args.export:
        status = agent.export()
        print(f"exported v{status['exported_version']}")

    frames = []
    returns = []

    for episode in range(1, args.episodes + 1):
        episode_return, steps, version = play_episode(
            env, agent, args.explore, args.max_steps, render, frames, args.fps, args.channel
        )
        returns.append(episode_return)
        print(f"episode {episode}: return {episode_return:8.1f}  steps {steps:4d}  policy v{version}")

    mode = "explore" if args.explore else "greedy"
    mean_return = float(np.mean(returns))
    print(
        f"mean return {mean_return:.1f} over {len(returns)} episodes ({mode}, channel {args.channel} -> v{version})"
    )
    env.close()

    if args.report and not args.explore:
        status = agent.report(version, mean_return, len(returns))
        print(f"reported: best is now v{status['versions']['best']}")
    if args.promote:
        status = agent.promote(version)
        print(f"promoted: stable is now v{status['versions']['stable']}")

    if args.video:
        import imageio

        imageio.mimwrite(args.video, frames, fps=args.fps)
        print(f"wrote {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
