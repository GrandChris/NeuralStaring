# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.148.1/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version: bionic, focal
ARG VARIANT="groovy"
FROM ubuntu:groovy

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends --reinstall ca-certificates \
    && apt-get -y install --no-install-recommends git build-essential curl locate
    
# Get Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup component add rust-analysis --toolchain stable-x86_64-unknown-linux-gnu \
    && rustup component add rust-src --toolchain stable-x86_64-unknown-linux-gnu \
    && rustup component add rls --toolchain stable-x86_64-unknown-linux-gnu

RUN updatedb

