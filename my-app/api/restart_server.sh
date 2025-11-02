#!/bin/bash
# Restart script for agent server

echo "Stopping any existing agent server..."
pkill -f "agent_server.py" 2>/dev/null
sleep 2

echo "Starting agent server..."
cd "$(dirname "$0")"
python3 agent_server.py

