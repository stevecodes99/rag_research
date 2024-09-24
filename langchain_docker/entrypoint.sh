#!/bin/bash

# Start ollama serve in the background and redirect output to a log file
ollama serve > /var/log/ollama.log 2>&1 &

ollama pull llama3

# Execute the container's main process (what's set as CMD in the Dockerfile)
exec "$@"
