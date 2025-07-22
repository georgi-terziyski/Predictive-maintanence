#!/bin/bash

# Test script for containerized Predictive Maintenance System
# This script tests all the main endpoints to ensure the system is working correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to test endpoint
test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_status="${3:-200}"
    
    print_status "Testing $name..."
    
    if response=$(curl -s -w "%{http_code}" -o /tmp/test_response "$url" 2>/dev/null); then
        status_code="${response: -3}"
        if [ "$status_code" = "$expected_status" ]; then
            print_success "$name - Status: $status_code"
            ((TESTS_PASSED++))
            return 0
        else
            print_error "$name - Expected: $expected_status, Got: $status_code"
            ((TESTS_FAILED++))
            return 1
        fi
    else
        print_error "$name - Connection failed"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to test POST endpoint
test_post_endpoint() {
    local name="$1"
    local url="$2"
    local data="$3"
    local expected_status="${4:-200}"
    
    print_status "Testing $name..."
    
    if response=$(curl -s -w "%{http_code}" -o /tmp/test_response -X POST -H "Content-Type: application/json" -d "$data" "$url" 2>/dev/null); then
        status_code="${response: -3}"
        if [ "$status_code" = "$expected_status" ]; then
            print_success "$name - Status: $status_code"
            ((TESTS_PASSED++))
            return 0
        else
            print_error "$name - Expected: $expected_status, Got: $status_code"
            echo "Response: $(cat /tmp/test_response)"
            ((TESTS_FAILED++))
            return 1
        fi
    else
        print_error "$name - Connection failed"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:5000/health" > /dev/null 2>&1; then
            print_success "Services are ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "Services did not become ready within expected time"
    return 1
}

# Main test function
run_tests() {
    echo "=========================================="
    echo "  Predictive Maintenance System Tests"
    echo "=========================================="
    echo ""
    
    # Wait for services
    if ! wait_for_services; then
        print_error "Services are not ready. Please ensure the system is running with: ./docker-start.sh start"
        exit 1
    fi
    
    echo ""
    print_status "Starting endpoint tests..."
    echo ""
    
    # Test health endpoints
    test_endpoint "Supervisor Health" "http://localhost:5000/health"
    test_endpoint "Data Agent Health" "http://localhost:5001/health"
    test_endpoint "Prediction Agent Health" "http://localhost:5002/health"
    test_endpoint "Simulation Agent Health" "http://localhost:5003/health"
    
    echo ""
    print_status "Testing data endpoints..."
    
    # Test data endpoints
    test_endpoint "Machine List" "http://localhost:5000/machine_list"
    test_endpoint "Live Data" "http://localhost:5000/live_data"
    
    echo ""
    print_status "Testing prediction endpoints..."
    
    # Test prediction endpoint (this might fail if no data is available)
    test_post_endpoint "Prediction Request" "http://localhost:5000/predict" '{"machine_id":"M001"}' "200"
    
    echo ""
    print_status "Testing simulation endpoints..."
    
    # Test simulation endpoint
    test_post_endpoint "Simulation Request" "http://localhost:5000/simulate" '{
        "machine_id": "M001",
        "duration_hours": 24,
        "initial_values": {
            "temperature": 75.0,
            "pressure": 4.5
        },
        "fixed_parameters": {
            "temperature": 85.0
        }
    }' "200"
    
    echo ""
    echo "=========================================="
    echo "  Test Results"
    echo "=========================================="
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        print_success "All tests passed! ($TESTS_PASSED/$((TESTS_PASSED + TESTS_FAILED)))"
        echo ""
        echo "✅ Your containerized Predictive Maintenance System is working correctly!"
        echo ""
        echo "You can now:"
        echo "  - Access the supervisor at: http://localhost:5000"
        echo "  - View logs with: ./docker-start.sh logs"
        echo "  - Check status with: ./docker-start.sh status"
        echo "  - Stop the system with: ./docker-start.sh stop"
        echo ""
        exit 0
    else
        print_error "Some tests failed. ($TESTS_PASSED passed, $TESTS_FAILED failed)"
        echo ""
        echo "❌ Please check the logs for more information:"
        echo "  ./docker-start.sh logs"
        echo ""
        echo "Common issues:"
        echo "  - Database not initialized (wait longer or check postgres logs)"
        echo "  - Missing data files (ensure data/ directory has required files)"
        echo "  - Port conflicts (check if ports 5000-5003 are available)"
        echo ""
        exit 1
    fi
}

# Cleanup function
cleanup() {
    rm -f /tmp/test_response
}

# Set trap for cleanup
trap cleanup EXIT

# Check if system is running
if ! curl -s -f "http://localhost:5000/health" > /dev/null 2>&1; then
    print_warning "System doesn't appear to be running."
    echo "Starting the system first..."
    ./docker-start.sh start
    echo ""
fi

# Run the tests
run_tests
