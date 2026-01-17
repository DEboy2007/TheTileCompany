#!/bin/bash

echo "🚀 Starting NexHacks PostgreSQL Database..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Navigate to the database directory
cd "$(dirname "$0")"

# Start the database
docker-compose up -d

# Wait for the database to be ready
echo ""
echo "⏳ Waiting for database to be ready..."
sleep 3

# Check if the database is healthy
if docker-compose ps | grep -q "healthy"; then
    echo ""
    echo "✅ Database is running!"
    echo ""
    echo "📊 Connection Details:"
    echo "  Host:     localhost"
    echo "  Port:     5432"
    echo "  Database: nexhacks"
    echo "  Username: postgres"
    echo "  Password: postgres"
    echo ""
    echo "🔗 Connection String:"
    echo "  postgresql://postgres:postgres@localhost:5432/nexhacks"
    echo ""
    echo "📝 To stop the database, run: ./stop.sh"
    echo "📝 To view logs, run: docker-compose logs -f"
else
    echo ""
    echo "⚠️  Database started but may still be initializing..."
    echo "   Run 'docker-compose logs -f' to check the status"
fi
