# PostgreSQL Database Setup

This directory contains scripts to spin up a local PostgreSQL database using Docker.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)

## Quick Start

### 1. Start the Database

```bash
cd database
chmod +x start.sh stop.sh
./start.sh
```

### 2. Connection Details

Once the database is running, use these credentials:

```
Host:     localhost
Port:     5432
Database: nexhacks
Username: postgres
Password: postgres
```

**Connection String:**
```
postgresql://postgres:postgres@localhost:5432/nexhacks
```

### 3. Stop the Database

```bash
./stop.sh
```

## Sample Data

The database is initialized with two tables:

### `sample_data` Table
- Contains 3 sample items for testing
- Columns: `id`, `name`, `description`, `created_at`

### `users` Table
- Contains 2 sample users
- Columns: `id`, `username`, `email`, `created_at`

## Manual Docker Commands

If you prefer to use Docker commands directly:

```bash
# Start the database
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the database
docker-compose down

# Stop and remove all data
docker-compose down -v
```

## Connecting from the App

In your Next.js app, you can connect using the credentials above. The app includes a UI to save these credentials in the browser.

## Troubleshooting

**Port already in use:**
If port 5432 is already in use, you can either:
1. Stop the existing PostgreSQL service
2. Edit `docker-compose.yml` to use a different port (e.g., `5433:5432`)

**Container won't start:**
Check Docker Desktop is running and has sufficient resources allocated.

**Reset database:**
To start fresh, run:
```bash
docker-compose down -v
./start.sh
```
