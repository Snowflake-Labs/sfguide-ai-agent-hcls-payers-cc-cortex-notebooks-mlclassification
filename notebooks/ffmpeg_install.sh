#!/bin/bash
export DEBIAN_FRONTEND=noninteractive \
    && apt-get upgrade -y && apt-get update -y \
    && apt-get -qq install --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*