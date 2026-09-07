#!/usr/bin/env bash
#
# Format and lint the repo.
#   Python: ruff (pyproject.toml)          triton/common, triton/model_repository/*/1/model.py, gym/
#   C++   : clang-format (.clang-format)   ue/Plugins/*/Source/*/{Public,Private}   (generated + third-party excluded)
#
# Usage:
#   scripts/lint.sh          check only (exit 1 on findings)   -- what CI runs
#   scripts/lint.sh --fix    rewrite files in place
#
# Needs `uv` (https://docs.astral.sh/uv/): tools run through `uvx`, nothing is installed globally.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUFF="uvx ruff@0.14.14"
CLANG_FORMAT="uvx clang-format@20.1.8"

FIX=0
if [ "${1:-}" = "--fix" ]; then
  FIX=1
fi

# ---------------------------------------------------------------- python

PY_PATHS=(triton/common triton/model_repository/*/1/model.py gym)

if [ "$FIX" = 1 ]; then
  $RUFF format "${PY_PATHS[@]}"
  $RUFF check --fix "${PY_PATHS[@]}"
else
  $RUFF format --check "${PY_PATHS[@]}"
  $RUFF check "${PY_PATHS[@]}"
fi

# ---------------------------------------------------------------- c++

mapfile -t CPP_FILES < <(find ue/Plugins -path "*/Source/*" \( -name "*.h" -o -name "*.cpp" \) \
  -not -path "*/ThirdParty/*" | sort)

if [ "$FIX" = 1 ]; then
  $CLANG_FORMAT -i "${CPP_FILES[@]}"
else
  $CLANG_FORMAT --dry-run --Werror "${CPP_FILES[@]}"
fi

echo "lint ok (${#PY_PATHS[@]} python paths, ${#CPP_FILES[@]} c++ files)"
