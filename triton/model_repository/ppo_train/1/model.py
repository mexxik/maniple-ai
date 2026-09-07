"""ppo_train: PPO algorithm endpoint. All logic lives in /common/maniple (triton/common in the repo)."""

from maniple.algorithms.ppo import PPO
from maniple.triton_model import AlgorithmModel


class TritonPythonModel(AlgorithmModel):
    algorithm = PPO
