# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.231.6/containers/ubuntu/.devcontainer/base.Dockerfile

FROM ubuntu:focal

# set timezone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ./metavision.list is not included in the repository, you need to manually create .devcontainer/metavision.list from metavision.list.template
COPY ./metavision.list /tmp/metavision.list
# check on a temporary copy, because we don't have ca-certificates yet, and apt-get update would fail
RUN grep -q '^[^#]' /tmp/metavision.list || (echo "ERROR: metavision.list is empty. Add Metavision SDK repository URL to .devcontainer/metavision.list!" && exit 1)

# intalling basic tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt autoremove -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install \
    ca-certificates \
    git \
    gpg \
    libgtk2.0-dev \
    locales \
    lsb-release \
    parallel \
    sudo \
    unzip \
    wget \
    -y

# install python3 and packages
# pylops API changes rapidly, 1.18.3 seems to have the one that is used in ESL code
RUN DEBIAN_FRONTEND=noninteractive apt -y install python3-pip python3-tk && \
    DEBIAN_FRONTEND=noninteractive python3 -m pip install pip --upgrade && \
    DEBIAN_FRONTEND=noninteractive python3 -m pip install \
    black \
    click \
    fire \
    funcy \
    h5py \
    ipython \
    jupyter \
    matplotlib \
    numba \
    numpy \
    open3d \
    opencv-python \
    pandas \
    pylops==1.18.3 \
    pyntcloud \
    pyyaml \
    scikit-video \
    scipy \
    seaborn

# installing MetavisionSDK
COPY ./metavision.list /etc/apt/sources.list.d/metavision.list

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install 'metavision-*' -y

RUN export LC_ALL=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN locale-gen en_US.UTF-8

# adding user with UID/GID 1001 and add user to sudo
ARG USER="vscode"
ARG USER_UID=1000
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
