# Model-agnostic coding agent. Build: docker build -t agent .
# Run:   docker run --rm -it -v "$PWD":/work -v ~/.agent:/home/agent/.agent agent
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# git (checkpoints), ripgrep (grep tool), bubblewrap (optional bash sandbox), node/npm (npx MCP servers)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ripgrep bubblewrap nodejs npm ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash agent

WORKDIR /opt/agent
COPY pyproject.toml README.md ./
COPY agent ./agent
RUN pip install . && agent --version

USER agent
WORKDIR /work
ENTRYPOINT ["agent"]
