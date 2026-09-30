#!/usr/bin/env bash
# CI gate: lint, strict typing, tests with coverage thresholds on core packages.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ruff check agent tests evals
python -m mypy --strict agent
python -m pytest -q --cov=agent --cov-report=term-missing:skip-covered
python -m coverage report --include='agent/adapters/*,agent/tools/*,agent/context/*,agent/mcp/*' --fail-under=85
