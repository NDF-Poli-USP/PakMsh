FROM ubuntu:24.04 AS pakmsh_base

USER root
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        git && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# pakmsh dependencies
RUN python -m pip install --upgrade pip setuptools wheel

FROM pakmsh_base AS pakmsh_user

RUN python -m pip install --no-cache-dir "pakmsh[examples] @ git+https://github.com/NDF-Poli-USP/PakMsh.git"

FROM pakmsh_base AS pakmsh_development

WORKDIR /home/shared/pakmsh
COPY . .
RUN python -m pip install --no-cache-dir \
    -e ".[dev]"
WORKDIR /home/