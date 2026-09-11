"""Gymnasium environments as the trainer sees them: which inputs they produce, which action groups they take.

Two kinds of observation are handled:
  vectors (Box of floats, any shape)            -> input "obs"   [N, D]
  images  (Box of uint8, 3 dims, e.g. Atari)    -> input "frame" [N, H, W, C]
Atari environments (ids starting with "ALE/") get the standard preprocessing: 4-frame skip, 84x84 grayscale,
4 frames stacked into the channels, so "frame" is [84, 84, 4].

Action spaces map to groups: Discrete(n) -> one discrete group "action"; Box -> one continuous group "action",
or one group per dimension with groups="per-dim" (exercises several heads on a plain environment).
"""

import gymnasium as gym
import numpy as np

ATARI_PREFIX = "ALE/"
ATARI_SCREEN = 84
ATARI_STACK = 4


def is_atari(env_id):
    return env_id.startswith(ATARI_PREFIX)


def make_env(env_id, render_mode=None):
    """One environment with the wrappers the trainer's inputs assume."""
    if not is_atari(env_id):
        return gym.make(env_id, render_mode=render_mode)

    import ale_py

    gym.register_envs(ale_py)
    env = gym.make(env_id, render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=ATARI_SCREEN, grayscale_obs=True, scale_obs=False
    )
    return gym.wrappers.FrameStackObservation(env, ATARI_STACK)  # obs [4, 84, 84] uint8


def input_layout(observation_space):
    """(input name, row shape) for an observation space."""
    if observation_space.dtype == np.uint8 and len(observation_space.shape) == 3:
        shape = list(observation_space.shape)
        if shape[0] in (1, 3, 4, ATARI_STACK) and shape[0] < shape[-1]:  # stacked frames come channels-first
            shape = shape[1:] + shape[:1]
        return "frame", shape
    return "obs", [int(np.prod(observation_space.shape))]


def to_inputs(obs, observation_space):
    """A batch of raw observations [N, ...] -> the inputs dict for act() / observe()."""
    name, shape = input_layout(observation_space)
    n = obs.shape[0]
    if name == "frame":
        frames = np.asarray(obs, dtype=np.uint8)
        if list(frames.shape[1:]) != shape:  # [N, C, H, W] -> [N, H, W, C]
            frames = np.transpose(frames, (0, 2, 3, 1))
        return {"frame": np.ascontiguousarray(frames)}
    return {"obs": np.asarray(obs, dtype=np.float32).reshape(n, -1)}


def spec_inputs(observation_space):
    name, shape = input_layout(observation_space)
    if name == "frame":
        return {"frame": {"shape": shape, "dtype": "uint8", "encoder": "cnn"}}
    return {"obs": {"shape": shape, "dtype": "fp32", "encoder": "mlp"}}


def spec_actions(action_space, groups="single"):
    """Action groups for an action space. groups: single | per-dim (continuous only)."""
    if isinstance(action_space, gym.spaces.Discrete):
        return {"action": {"type": "discrete", "n": int(action_space.n)}}
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError(f"unsupported action space {action_space}")
    low = action_space.low.reshape(-1)
    high = action_space.high.reshape(-1)
    if groups == "per-dim":
        return {
            f"a{i}": {"type": "continuous", "dim": 1, "low": float(low[i]), "high": float(high[i])}
            for i in range(low.shape[0])
        }
    return {
        "action": {
            "type": "continuous",
            "dim": int(low.shape[0]),
            "low": float(low.min()),
            "high": float(high.max()),
        }
    }


def to_env_action(action, action_index, action_space):
    """Translate the server's action row / indices into what env.step() expects (batched)."""
    if isinstance(action_space, gym.spaces.Discrete):
        return action_index[:, 0]
    return np.clip(action, action_space.low.reshape(-1), action_space.high.reshape(-1)).reshape(
        (action.shape[0], *action_space.shape)
    )
