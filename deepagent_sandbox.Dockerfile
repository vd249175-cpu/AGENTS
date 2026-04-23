# DeepAgent sandbox base image, preinstalled with common shell tools and Python.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        curl \
        file \
        findutils \
        git \
        grep \
        less \
        procps \
        ripgrep \
        sed \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["tail", "-f", "/dev/null"]
