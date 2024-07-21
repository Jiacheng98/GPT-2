#!/bin/sh

# Check the first command-line argument
if [ "$1" = "training" ]; then
    # Run the training script
    exec python main.py
elif [ "$1" = "service" ]; then
    # Run the backend service script
    exec python backend_server.py
else
    echo "Invalid argument. Please use 'training' or 'service'."
    exit 1
fi