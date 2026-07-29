FROM ubuntu:24.04 AS pakmsh_base

USER root
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pakmsh dependencies
RUN pip3 install segyio
RUN pip3 install meshio


FROM pakmsh_base AS pakmsh_user

RUN pip install git+https://github.com/NDF-Poli-USP/PakMsh.git

FROM pakmsh_base AS pakmsh_development

# Profiling tools
RUN pip3 install -U memory_profiler

RUN pip3 install pytest pytest-cov
WORKDIR /home/
RUN echo "/home/shared/pakmsh" >> /usr/local/lib/python3.12/site-packages/shared.pth
