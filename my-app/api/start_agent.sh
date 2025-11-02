#!/bin/bash
# Startup script for the agent server

# Set default model if not provided
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
export PORT="${PORT:-5000}"

echo "Starting Agent Server..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Run the server
python3 agent_server.py

