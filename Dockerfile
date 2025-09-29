FROM nvcr.io/nvidia/pytorch:25.09-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
 && rm -rf /var/lib/apt/lists/*

# Create workdir
RUN mkdir /app
WORKDIR /app

# Upgrade pip before installing
RUN python3 -m pip install --upgrade pip

# Copy requirements and install
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt \
    && python3 -m pip install ipykernel

# Set NVIDIA environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Default command (override if needed)
CMD ["bash"]
