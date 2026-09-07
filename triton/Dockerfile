# Triton + PyTorch for the ppo_train Python-backend model.
FROM nvcr.io/nvidia/tritonserver:26.07-py3
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu130 && \
    pip install --no-cache-dir --ignore-installed typing_extensions && pip install --no-cache-dir onnx numpy
