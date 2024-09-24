# Use the NVIDIA CUDA runtime base image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set the maintainer label (optional)
LABEL maintainer="stevesteverulez@gmail.com"

# Install dependencies, including Python 3.10, and Ollama
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev curl && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/

# Expose the port `ollama` uses
EXPOSE 11434

# Make entrypoint script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the default command
CMD ["bash"]
