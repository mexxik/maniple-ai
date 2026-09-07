"""maniple: shared code for the Triton-hosted training/inference models (mounted into the server as /common).

spec.py          AgentSpec — the contract between game, trainer and exported policy
nets.py          networks built from a spec (MLP actor / actor-critic)
buffer.py        transition buffer with trajectory stitching and policy-lag filtering
export.py        write a policy as a static Triton model '<name>_policy/<version>'
registry.py      named policies + background training threads + checkpoints
triton_model.py  AlgorithmModel — base class for '<algo>_train/<v>/model.py'
infer_model.py   InferModel — base class for '<algo>_infer/<v>/model.py' (BLS to '<name>_policy', sampling)
algorithms/      one class per algorithm (ppo.py, ...), all implementing algorithms.base.Algorithm
"""
