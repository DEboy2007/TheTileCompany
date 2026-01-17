#!/bin/bash

echo "🛑 Stopping NexHacks PostgreSQL Database..."

# Navigate to the database directory
cd "$(dirname "$0")"

# Stop the database
docker-compose down

echo "✅ Database stopped successfully!"
