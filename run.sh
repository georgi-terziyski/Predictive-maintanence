#!/bin/bash
# Run script for the Predictive Maintenance System

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=====================================================${NC}"
echo -e "${YELLOW}  Predictive Maintenance System - Run Tool           ${NC}"
echo -e "${YELLOW}=====================================================${NC}"
echo

# Ensure pids directory exists
mkdir -p agents/pids

# Source the environment variables
echo -e "${YELLOW}Loading environment variables...${NC}"
if [ -f ".env" ]; then
    source .env
else
    echo -e "${RED}Error: .env file not found. Please run setup.sh first.${NC}"
    exit 1
fi

# Function to check if an agent is already running
is_agent_running() {
    local agent_name=$1
    local pid_file="agents/pids/${agent_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat $pid_file)
        if ps -p $pid > /dev/null; then
            return 0 # Agent is running
        else
            # PID file exists but process is not running, clean up
            rm -f $pid_file
        fi
    fi
    return 1 # Agent is not running
}

# Function to start an agent
start_agent() {
    local agent_name=$1
    local agent_script=$2
    local agent_port=$3
    
    if is_agent_running $agent_name; then
        echo -e "${YELLOW}${agent_name} is already running.${NC}"
        return
    fi
    
    echo -e "${GREEN}Starting ${agent_name} on port ${agent_port}...${NC}"
    python $agent_script &
    local pid=$!
    echo $pid > "agents/pids/${agent_name}.pid"
    echo -e "${BLUE}${agent_name} started with PID: ${pid}${NC}"
}

# Function to start all agents
start_all_agents() {
    # Start in the correct order: data_agent -> prediction_agent -> simulation_agent -> supervisor
    start_agent "data_agent" "agents/data_agent/app.py" $DATA_AGENT_PORT
    sleep 2 # Give data agent time to start
    
    start_agent "prediction_agent" "agents/prediction_agent/app.py" $PREDICTION_AGENT_PORT
    sleep 1
    
    start_agent "simulation_agent" "agents/simulation_agent/app.py" $SIMULATION_AGENT_PORT
    sleep 1
    
    start_agent "supervisor" "agents/supervisor/app.py" $SUPERVISOR_PORT
    
    echo -e "${GREEN}All agents started. Use ./stop_reboot.sh to stop or reboot agents.${NC}"
}

# Display agent statuses
display_status() {
    echo -e "${YELLOW}Current agent status:${NC}"
    
    if is_agent_running "data_agent"; then
        echo -e "${GREEN}Data Agent: Running (PID: $(cat agents/pids/data_agent.pid), Port: $DATA_AGENT_PORT)${NC}"
    else
        echo -e "${RED}Data Agent: Not running${NC}"
    fi
    
    if is_agent_running "prediction_agent"; then
        echo -e "${GREEN}Prediction Agent: Running (PID: $(cat agents/pids/prediction_agent.pid), Port: $PREDICTION_AGENT_PORT)${NC}"
    else
        echo -e "${RED}Prediction Agent: Not running${NC}"
    fi
    
    if is_agent_running "simulation_agent"; then
        echo -e "${GREEN}Simulation Agent: Running (PID: $(cat agents/pids/simulation_agent.pid), Port: $SIMULATION_AGENT_PORT)${NC}"
    else
        echo -e "${RED}Simulation Agent: Not running${NC}"
    fi
    
    if is_agent_running "supervisor"; then
        echo -e "${GREEN}Supervisor: Running (PID: $(cat agents/pids/supervisor.pid), Port: $SUPERVISOR_PORT)${NC}"
    else
        echo -e "${RED}Supervisor: Not running${NC}"
    fi
}

# Show menu of options
echo -e "${YELLOW}What would you like to do?${NC}"
echo "1. Run all agents"
echo "2. Run data agent only"
echo "3. Run prediction agent only"
echo "4. Run simulation agent only"
echo "5. Run supervisor only"
echo "6. Display agent status"
echo "7. Exit"
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        start_all_agents
        ;;
    2)
        start_agent "data_agent" "agents/data_agent/app.py" $DATA_AGENT_PORT
        ;;
    3)
        start_agent "prediction_agent" "agents/prediction_agent/app.py" $PREDICTION_AGENT_PORT
        ;;
    4)
        start_agent "simulation_agent" "agents/simulation_agent/app.py" $SIMULATION_AGENT_PORT
        ;;
    5)
        start_agent "supervisor" "agents/supervisor/app.py" $SUPERVISOR_PORT
        ;;
    6)
        display_status
        ;;
    7)
        echo -e "${GREEN}Exiting.${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "${YELLOW}You can use ./stop_reboot.sh to stop or reboot agents.${NC}"
