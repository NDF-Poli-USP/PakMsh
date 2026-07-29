# PakMsh Docker Environment

This directory contains the Docker configuration used to build reproducible environments for **PakMsh**.

The Dockerfile has two diferent build targets, for different kind of users:

* **`pakmsh_user`** – A lightweight image intended for users who simply want to install and use PakMsh.
* **`pakmsh_development`** – A base image for development, intended to be used with or without VS Code Dev Containers.

## Requirements

* Docker
* (Optional) Visual Studio Code with the **Dev Containers** extension

---

# Docker Images

## User Image

The `pakmsh_user` target creates an image with PakMsh installed directly from the current main GitHub repository and includes the optional `examples` dependencies.

Build the image:

```bash
docker build \
    --target pakmsh_user \
    -t pakmsh:user \
    docker/
```

Run an interactive shell:

```bash
docker run --rm -it pakmsh:user bash
```

---

## Developmer Image

The `pakmsh_development` target is intended for local development.

Unlike the user image, it **does not install PakMsh during the Docker build**. If the user is using VScode, the package is installed in editable mode after the development container has been created. If the user isn't using VS code they have to install the package after building the container as explained below.

This approach provides was chosen because:
* Changes to the source code are immediately visible.
* No Docker image rebuild is required after modifying Python files.
* The development environment always uses the local checkout (so you can just use git to switch branches and git pull will imeadiatly update your code).

This is the recommended workflow for Python development 

---

# Using VS Code Dev Containers

Open the PakMsh repository in Visual Studio Code and select the command (usually by pressing f1):

```
Dev Containers: Reopen in Container
```

The Dev Container performs the following steps automatically:
1. Builds the `pakmsh_development` image.
2. Mounts the local repository into the container.
3. Executes a installation with (you don't need to do this, VS code handles it):
```bash
python -m pip install -e ".[dev]"
```
via the `postCreateCommand` defined in `.devcontainer/devcontainer.json`.

This installs PakMsh in editable mode together with all development dependencies defined in `pyproject.toml`.

---

# Developing Without VS Code

The development image can also be used without Visual Studio Code.

## Build the development image

```bash
docker build \
    --target pakmsh_development \
    -t pakmsh:dev \
    docker/
```

## Start a container

From the root of the PakMsh repository, mount the local source tree into the container:

```bash
docker run --rm -it \
    -v "$(pwd):/workspace" \
    -w /workspace \
    pakmsh:dev \
    bash
```

## Install PakMsh in editable mode

Inside the container, install the project together with its development dependencies:

```bash
python -m pip install -e ".[dev]"
```

This only needs to be done once per container.

After the installation, any modifications to the local source code are immediately visible inside the container because the repository is mounted as a Docker volume. Therefore you can use any branch and modify your code at will.

---

# Dockerfile Structure

```
pakmsh_base
├── Ubuntu 24.04
├── Python 3.12
├── Virtual environment (/opt/venv)
└── Updated packaging tools
    │
    ├──► pakmsh_user
    │       Installs PakMsh from GitHub
    │
    └──► pakmsh_development
            Base image for Dev Containers
```

---

# Virtual Environment

All Python packages are installed inside a virtual environment located at

```
/opt/venv
```

The environment is activated automatically through the `PATH` environment variable, so all `python` and `pip` commands executed inside the container use the virtual environment by default.
