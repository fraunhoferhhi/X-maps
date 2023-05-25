# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.231.6/containers/ubuntu/.devcontainer/base.Dockerfile

FROM mambaorg/micromamba:1.4-jammy

USER root

# set timezone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ./metavision.list is not included in the repository, you need to manually create .devcontainer/metavision.list from metavision.list.template
COPY ./metavision.list /tmp/metavision.list
# check on a temporary copy, because we don't have ca-certificates yet, and apt-get update would fail
RUN grep -q '^[^#]' /tmp/metavision.list || (echo "ERROR: metavision.list is empty. Add Metavision SDK repository URL to .devcontainer/metavision.list!" && exit 1)

USER root

# intalling basic tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt autoremove -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    git \
    gpg \
    libgtk2.0-dev \
    locales \
    lsb-release \
    parallel \
    sudo \
    unzip \
    wget

# X Error of failed request:  BadShmSeg (invalid shared segment parameter) on Jammy
# https://github.com/RobotLocomotion/drake/issues/18726
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades \
    xvfb \
    libegl1 \
    libegl-mesa0=22.0.1-1ubuntu2 \
    libgbm1=22.0.1-1ubuntu2 \
    libgl1-mesa-dri=22.0.1-1ubuntu2 \
    libglapi-mesa=22.0.1-1ubuntu2 \
    libglx-mesa0=22.0.1-1ubuntu2

RUN DEBIAN_FRONTEND=noninteractive apt-mark hold \
    libegl-mesa0 \
    libgbm1 \
    libgl1-mesa-dri \
    libglapi-mesa \
    libglx-mesa0

# installing MetavisionSDK
COPY ./metavision.list /etc/apt/sources.list.d/metavision.list

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    metavision-hal-bin \
    metavision-hal-prophesee-plugins \
    metavision-sdk-core-bin \
    metavision-sdk-core-python \
    metavision-sdk-core-python3.10 \
    metavision-sdk-python3.10

RUN export LC_ALL=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN locale-gen en_US.UTF-8

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

USER root

ARG USER="vscode"
ARG USER_UID=1001
ARG USER_GID=${USER_UID}

RUN DEBIAN_FRONTEND=noninteractive groupadd --gid ${USER_GID} ${USER} && \
    DEBIAN_FRONTEND=noninteractive useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER} && \
    DEBIAN_FRONTEND=noninteractive echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER} && \
    DEBIAN_FRONTEND=noninteractive chmod 0440 /etc/sudoers.d/${USER} && \
    DEBIAN_TRONTEND=noninteractive usermod --shell /bin/bash ${USER} && \
    DEBIAN_TRONTEND=noninteractive usermod -aG plugdev ${USER} && \
    DEBIAN_TRONTEND=noninteractive usermod -aG root ${USER} && \
    # create directories to hold vscode extensions
    DEBIAN_FRONTEND=noninteractive mkdir -p \
    /home/${USER}/.vscode-server/extensions \
    /home/${USER}/.vscode-server-insiders/extensions && \
    DEBIAN_FRONTEND=noninteractive chown -R ${USER} \
    /home/${USER}/.vscode-server \
    /home/${USER}/.vscode-server-insiders

USER ${USER}
