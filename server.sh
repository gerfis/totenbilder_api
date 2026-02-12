#!/bin/bash

# Configuration
PORT=8000
VENV_PYTHON="venv/bin/python"
MAIN_SCRIPT="main.py"

case "$1" in
    start)
        echo "Starting server on port $PORT..."
        if [ -f "$VENV_PYTHON" ]; then
            $VENV_PYTHON $MAIN_SCRIPT
        else
            python3 $MAIN_SCRIPT
        fi
        ;;
    stop)
        echo "Stopping server on port $PORT..."
        PID=$(lsof -ti :$PORT)
        if [ -n "$PID" ]; then
            kill -9 $PID
            echo "Server stopped (PID: $PID)."
        else
            echo "No server running on port $PORT."
        fi
        ;;
    restart)
        $0 stop
        sleep 1
        $0 start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac
