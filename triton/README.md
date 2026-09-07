# Triton model server

    docker compose up -d
    curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/v2/health/ready     # 200
    curl -s localhost:8000/v2/models/policy_mlp | jq
    # model artifacts are produced by external scripts (not in this repo)
    docker compose logs -f triton
    docker compose down

Layout: `model_repository/<model>/<version>/model.onnx` + `model_repository/<model>/config.pbtxt`.
The server polls the repository every 5 s; adding a new version directory hot-swaps the model.
