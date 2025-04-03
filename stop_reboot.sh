#!/bin/bash
# Stop and Reboot script for the Predictive Maintenance System

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=====================================================${NC}"
echo -e "${YELLOW}  Predictive Maintenance System - Stop/Reboot Tool   ${NC}"
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

# Function to check if an agent is running
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

# Function to stop an agent
stop_agent() {
    local agent_name=$1
    local pid_file="agents/pids/${agent_name}.pid"
    
    if is_agent_running $agent_name; then
        local pid=$(cat $pid_file)
        echo -e "${YELLOW}Stopping ${agent_name} (PID: ${pid})...${NC}"
        kill $pid
        sleep 1
        
        # Check if process is still running
        if ps -p $pid > /dev/null; then
            echo -e "${YELLOW}Process still running, using force kill...${NC}"
            kill -9 $pid
            sleep 1
        fi
        
        # Remove PID file
        rm -f $pid_file
        echo -e "${GREEN}${agent_name} stopped successfully.${NC}"
    else
        echo -e "${YELLOW}${agent_name} is not running.${NC}"
    fi
}

# Function to start an agent (similar to run.sh)
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

# Function to reboot an agent
reboot_agent() {
    local agent_name=$1
    local agent_script=$2
    local agent_port=$3
    
    echo -e "${YELLOW}Rebooting ${agent_name}...${NC}"
    stop_agent $agent_name
    sleep 2
    start_agent $agent_name $agent_script $agent_port
    echo -e "${GREEN}${agent_name} rebooted successfully.${NC}"
}

# Function to stop all agents
stop_all_agents() {
    # Stop in reverse order: supervisor -> simulation_agent -> prediction_agent -> data_agent
    stop_agent "supervisor"
    sleep 1
    stop_agent "simulation_agent"
    sleep 1
    stop_agent "prediction_agent"
    sleep 1
    stop_agent "data_agent"
    echo -e "${GREEN}All agents stopped.${NC}"
}

# Function to reboot all agents
reboot_all_agents() {
    # First stop all agents
    stop_all_agents
    
    # Start in the correct order
    sleep 2
    start_agent "data_agent" "agents/data_agent/app.py" $DATA_AGENT_PORT
    sleep 2
    start_agent "prediction_agent" "agents/prediction_agent/app.py" $PREDICTION_AGENT_PORT
    sleep 1
    start_agent "simulation_agent" "agents/simulation_agent/app.py" $SIMULATION_AGENT_PORT
    sleep 1
    start_agent "supervisor" "agents/supervisor/app.py" $SUPERVISOR_PORT
    echo -e "${GREEN}All agents rebooted.${NC}"
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

# Main Menu
while true; do
    clear
    echo -e "${YELLOW}=====================================================${NC}"
    echo -e "${YELLOW}  Predictive Maintenance System - Stop/Reboot Tool   ${NC}"
    echo -e "${YELLOW}=====================================================${NC}"
    echo
    
    display_status
    echo
    
    echo -e "${YELLOW}What would you like to do?${NC}"
    echo "1. Stop all agents"
    echo "2. Reboot all agents"
    echo "3. Stop data agent"
    echo "4. Reboot data agent"
    echo "5. Stop prediction agent"
    echo "6. Reboot prediction agent"
    echo "7. Stop simulation agent"
    echo "8. Reboot simulation agent"
    echo "9. Stop supervisor"
    echo "10. Reboot supervisor"
    echo "11. Display agent status"
    echo "12. Exit"
    read -p "Enter your choice (1-12): " choice
    
    case $choice in
        1)
            stop_all_agents
            ;;
        2)
            reboot_all_agents
            ;;
        3)
            stop_agent "data_agent"
            ;;
        4)
            reboot_agent "data_agent" "agents/data_agent/app.py" $DATA_AGENT_PORT
            ;;
        5)
            stop_agent "prediction_agent"
            ;;
        6)
            reboot_agent "prediction_agent" "agents/prediction_agent/app.py" $PREDICTION_AGENT_PORT
            ;;
        7)
            stop_agent "simulation_agent"
            ;;
        8)
            reboot_agent "simulation_agent" "agents/simulation_agent/app.py" $SIMULATION_AGENT_PORT
            ;;
        9)
            stop_agent "supervisor"
            ;;
        10)
            reboot_agent "supervisor" "agents/supervisor/app.py" $SUPERVISOR_PORT
            ;;
        11)
            display_status
            ;;
        12)
            echo -e "${GREEN}Exiting.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
done
