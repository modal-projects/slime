# syntax=docker/dockerfile:1

# Use official Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variable to avoid interactive prompts during apt-get installation
ENV DEBIAN_FRONTEND=noninteractive

# --- 1. Install base dependencies, language toolchains and Node.js ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    jq \
    git \
    unzip \
    zip \
    build-essential \
    pkg-config \
    python3 \
    python3-pip \
    pypy3 \
    openjdk-17-jdk \
    # Install Node.js 20.x
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    # Clean apt cache
    && rm -rf /var/lib/apt/lists/*

# --- 2. Update npm (pinned to a major compatible with Node 20; npm@latest is
# npm 12+ which requires Node 22+ and breaks the build) ---
RUN npm install -g npm@11

# --- 3. Install go-judge (auto-detect architecture) ---
# v1.11.1 assets are named go-judge_<ver-without-v>_linux_<arch>.tar.gz, with
# amd64 published per microarch level (amd64v2 = safe modern x86 baseline). The
# tarball holds the `go-judge` binary at top level.
ARG GO_JUDGE_VERSION=v1.11.1
RUN set -eux; \
  ver="${GO_JUDGE_VERSION#v}"; \
  arch="$(uname -m)"; \
  case "$arch" in \
    x86_64)  goarch="amd64v2" ;; \
    aarch64) goarch="arm64" ;; \
    *) echo "Unsupported arch: $arch" && exit 1 ;; \
  esac; \
  url="https://github.com/criyle/go-judge/releases/download/${GO_JUDGE_VERSION}/go-judge_${ver}_linux_${goarch}.tar.gz"; \
  echo "Downloading go-judge ${GO_JUDGE_VERSION} ($goarch) from $url"; \
  curl -fsSL "$url" | tar -xz -C /usr/local/bin go-judge; \
  chmod +x /usr/local/bin/go-judge; \
  /usr/local/bin/go-judge --help >/dev/null 2>&1 || true

# --- 4. Set application working directory and install Node.js dependencies ---
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --omit=dev --ignore-scripts

# --- 5. Copy application code and entry script ---
COPY server.js entrypoint.sh ./
COPY judge/src/ ./src/
COPY judge/include/ ./include/
COPY judge/config/ ./config/
COPY judge/include/ /lib/testlib/

# Ensure entrypoint.sh is executable and fix possible Windows line ending issues
RUN chmod +x entrypoint.sh && sed -i 's/\r$//' entrypoint.sh

# --- 6. Set default environment variables ---
# Note: JUDGE_WORKERS and GJ_PARALLELISM default to $(nproc) at runtime in entrypoint.sh
ENV PORT=8081 \
    GJ_ADDR=http://127.0.0.1:5050

# --- 7. Set container entry point ---
ENTRYPOINT ["/app/entrypoint.sh"]