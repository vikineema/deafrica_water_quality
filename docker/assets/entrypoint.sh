#!/bin/sh
set -e

if [ "$BUILD_ENV" = "dev" ]; then
    echo "Starting Jupyter Lab in development mode..."
    exec jupyter lab --config="$PYTHON_ENV/etc/jupyter/jupyter_lab_config.py"
else
    echo "Running wq-version..."
    exec wq-version
fi
