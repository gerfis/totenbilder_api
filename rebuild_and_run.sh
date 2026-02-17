#!/bin/bash
echo "Building Docker image..."
docker build -t totenbilder-api .

echo "Stopping existing container..."
docker stop totenbilder-api 2>/dev/null || true
docker rm totenbilder-api 2>/dev/null || true

echo "Starting new container with .env configuration..."
docker run -d \
  --name totenbilder-api \
  -p 8000:8000 \
  --env-file .env \
  totenbilder-api

echo "Container started. Logs:"
sleep 2
docker logs totenbilder-api
