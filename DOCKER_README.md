# Containerized Predictive Maintenance System

This document provides instructions for running the Predictive Maintenance System using Docker containers.

## Architecture Overview

The system consists of 5 containers:

- **postgres**: PostgreSQL database
- **data-agent**: Handles database operations and data fetching (port 5001)
- **prediction-agent**: Runs ML predictions using inference models (port 5002)
- **simulation-agent**: Generates synthetic data and scenarios (port 5003)
- **supervisor**: Orchestrates communication between agents (port 5000)

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for containers
- At least 2GB disk space

## Quick Start

### 1. Build and Start All Services

```bash
# Build and start all containers
docker-compose up --build

# Or run in detached mode (background)
docker-compose up --build -d
```

### 2. Check Service Health

```bash
# Check all container status
docker-compose ps

# Check logs for all services
docker-compose logs

# Check logs for specific service
docker-compose logs supervisor
docker-compose logs data-agent
```

### 3. Test the System

```bash
# Test supervisor health
curl http://localhost:5000/health

# Test data agent
curl http://localhost:5001/health

# Test prediction agent
curl http://localhost:5002/health

# Test simulation agent
curl http://localhost:5003/health
```

## Development Mode

For development with hot-reload capabilities:

```bash
# Start in development mode (uses docker-compose.override.yml automatically)
docker-compose up --build

# The override file enables:
# - Source code volume mounts for hot reload
# - Flask debug mode
# - Development environment variables
```

## Production Deployment

For production deployment:

```bash
# Use production configuration only
docker-compose -f docker-compose.yml up --build -d

# Or create a production-specific override
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Database Setup

The PostgreSQL database is automatically initialized with:

- Database: `predictive_maintenance`
- User: `postgres`
- Password: `postgres`
- Port: `5432` (exposed for development)

### Connecting to Database

```bash
# Connect using docker exec
docker-compose exec postgres psql -U postgres -d predictive_maintenance

# Or connect from host (development only)
psql -h localhost -p 5432 -U postgres -d predictive_maintenance
```

## Volume Management

### Persistent Data

- `postgres_data`: Database data (persisted)
- `./logs`: Application logs (shared across containers)

### Development Volumes

- `./agents/*/`: Source code (mounted for hot reload)
- `./inference/`: ML models and inference data
- `./training/`: Training data and models
- `./data/`: Application data files

## Service Communication

Agents communicate using Docker's internal DNS:

- Data Agent: `http://data-agent:5001`
- Prediction Agent: `http://prediction-agent:5002`
- Simulation Agent: `http://simulation-agent:5003`

External access uses localhost:

- Supervisor: `http://localhost:5000`
- Data Agent: `http://localhost:5001`
- Prediction Agent: `http://localhost:5002`
- Simulation Agent: `http://localhost:5003`

## Common Commands

### Container Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart specific service
docker-compose restart supervisor

# View real-time logs
docker-compose logs -f

# Execute command in container
docker-compose exec data-agent bash
```

### Database Operations

```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres predictive_maintenance > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres predictive_maintenance < backup.sql

# Reset database (WARNING: destroys all data)
docker-compose down -v
docker-compose up --build
```

### Scaling Services

```bash
# Scale simulation agent to 3 instances
docker-compose up --scale simulation-agent=3 -d

# Note: Only stateless services should be scaled
```

## Troubleshooting

### Common Issues

1. **Port conflicts**

   ```bash
   # Check what's using the ports
   netstat -tulpn | grep :5000

   # Stop conflicting services or change ports in docker-compose.yml
   ```

2. **Database connection issues**

   ```bash
   # Check database logs
   docker-compose logs postgres

   # Verify database is ready
   docker-compose exec postgres pg_isready -U postgres
   ```

3. **Agent communication failures**

   ```bash
   # Check network connectivity between containers
   docker-compose exec supervisor ping data-agent

   # Verify all services are healthy
   docker-compose ps
   ```

4. **Volume mount issues**

   ```bash
   # Check volume mounts
   docker-compose exec data-agent ls -la /app/data

   # Verify file permissions
   ls -la ./data/
   ```

### Debugging

```bash
# Enter container for debugging
docker-compose exec supervisor bash

# Check environment variables
docker-compose exec supervisor env

# Monitor resource usage
docker stats

# View detailed container info
docker-compose exec supervisor cat /proc/1/environ | tr '\0' '\n'
```

## Environment Variables

Key environment variables (defined in docker-compose.yml):

### Database

- `DB_HOST=postgres`
- `DB_NAME=predictive_maintenance`
- `DB_USER=postgres`
- `DB_PASSWORD=postgres`

### Agent URLs

- `DATA_AGENT_URL=http://data-agent:5001`
- `PREDICTION_AGENT_URL=http://prediction-agent:5002`
- `SIMULATION_AGENT_URL=http://simulation-agent:5003`

### Flask Configuration

- `FLASK_HOST=0.0.0.0`
- `FLASK_ENV=production` (development in override)

## Security Considerations

### Development

- Database credentials are hardcoded for simplicity
- All ports are exposed for easy access
- Debug mode is enabled

### Production

- Change default database credentials
- Use Docker secrets for sensitive data
- Limit exposed ports
- Enable SSL/TLS
- Use non-root users in containers
- Implement proper logging and monitoring

## Performance Tuning

### Resource Limits

Add to docker-compose.yml:

```yaml
services:
  prediction-agent:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2G
        reservations:
          cpus: "1.0"
          memory: 1G
```

### Database Optimization

```yaml
postgres:
  environment:
    - POSTGRES_SHARED_PRELOAD_LIBRARIES=pg_stat_statements
    - POSTGRES_MAX_CONNECTIONS=200
  command: >
    postgres
    -c shared_preload_libraries=pg_stat_statements
    -c max_connections=200
    -c shared_buffers=256MB
```

## Monitoring

### Health Checks

All services include health checks that can be monitored:

```bash
# Check health status
docker-compose ps

# View health check logs
docker inspect --format='{{json .State.Health}}' container_name
```

### Logs

```bash
# Centralized logging
docker-compose logs -f --tail=100

# Service-specific logs
docker-compose logs -f supervisor
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres predictive_maintenance > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="./backups"
mkdir -p $BACKUP_DIR
docker-compose exec postgres pg_dump -U postgres predictive_maintenance > $BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql
```

### Volume Backup

```bash
# Backup volumes
docker run --rm -v predictive-maintanence_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_data_backup.tar.gz -C /data .
```

## Migration from Local Setup

1. **Export existing database**

   ```bash
   pg_dump -h localhost -U your_user predictive_maintenance > migration.sql
   ```

2. **Start containerized system**

   ```bash
   docker-compose up -d postgres
   ```

3. **Import data**

   ```bash
   docker-compose exec -T postgres psql -U postgres predictive_maintenance < migration.sql
   ```

4. **Start remaining services**
   ```bash
   docker-compose up -d
   ```

## Support

For issues related to containerization:

1. Check this README
2. Review docker-compose logs
3. Verify system requirements
4. Check Docker and Docker Compose versions
