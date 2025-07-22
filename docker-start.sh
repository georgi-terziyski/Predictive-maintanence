#!/bin/bash

# Predictive Maintenance System - Docker Startup Script
# This script provides easy commands to manage the containerized system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services (build if needed)"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show status of all services"
    echo "  logs        Show logs for all services"
    echo "  logs [service]  Show logs for specific service"
    echo "  build       Build all images"
    echo "  clean       Stop and remove all containers and volumes"
    echo "  health      Check health of all services"
    echo "  shell [service]  Open shell in service container"
    echo "  db          Connect to PostgreSQL database"
    echo "  backup      Backup database"
    echo "  help        Show this help message"
    echo ""
    echo "Services: supervisor, data-agent, prediction-agent, simulation-agent, postgres"
}

# Function to start services
start_services() {
    print_status "Starting Predictive Maintenance System..."
    docker-compose up --build -d
    print_success "All services started successfully!"
    print_status "Waiting for services to be ready..."
    sleep 10
    show_health
}

# Function to stop services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped successfully!"
}

# Function to restart services
restart_services() {
    print_status "Restarting all services..."
    docker-compose restart
    print_success "All services restarted successfully!"
}

# Function to show status
show_status() {
    print_status "Service Status:"
    docker-compose ps
}

# Function to show logs
show_logs() {
    if [ -z "$1" ]; then
        print_status "Showing logs for all services (press Ctrl+C to exit):"
        docker-compose logs -f --tail=50
    else
        print_status "Showing logs for $1 (press Ctrl+C to exit):"
        docker-compose logs -f --tail=50 "$1"
    fi
}

# Function to build images
build_images() {
    print_status "Building all Docker images..."
    docker-compose build --no-cache
    print_success "All images built successfully!"
}

# Function to clean up
clean_up() {
    print_warning "This will stop and remove all containers and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up all containers and volumes..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        print_success "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Function to check health
show_health() {
    print_status "Checking service health..."
    echo ""
    
    services=("supervisor:5000" "data-agent:5001" "prediction-agent:5002" "simulation-agent:5003")
    
    for service in "${services[@]}"; do
        name=$(echo "$service" | cut -d: -f1)
        port=$(echo "$service" | cut -d: -f2)
        
        if curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
            print_success "$name is healthy"
        else
            print_error "$name is not responding"
        fi
    done
    
    # Check database
    if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        print_success "postgres is healthy"
    else
        print_error "postgres is not responding"
    fi
}

# Function to open shell
open_shell() {
    if [ -z "$1" ]; then
        print_error "Please specify a service name"
        echo "Available services: supervisor, data-agent, prediction-agent, simulation-agent, postgres"
        exit 1
    fi
    
    print_status "Opening shell in $1 container..."
    docker-compose exec "$1" bash
}

# Function to connect to database
connect_db() {
    print_status "Connecting to PostgreSQL database..."
    docker-compose exec postgres psql -U postgres -d predictive_maintenance
}

# Function to backup database
backup_db() {
    backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"
    print_status "Creating database backup: $backup_file"
    docker-compose exec postgres pg_dump -U postgres predictive_maintenance > "$backup_file"
    print_success "Database backup created: $backup_file"
}

# Main script logic
main() {
    # Check prerequisites
    check_docker
    check_docker_compose
    
    # Handle commands
    case "${1:-help}" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$2"
            ;;
        build)
            build_images
            ;;
        clean)
            clean_up
            ;;
        health)
            show_health
            ;;
        shell)
            open_shell "$2"
            ;;
        db)
            connect_db
            ;;
        backup)
            backup_db
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
