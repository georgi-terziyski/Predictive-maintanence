#!/bin/bash
# view_logs.sh - View logs for any agent

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Available agents
AGENTS=("data_agent" "prediction_agent" "simulation_agent" "supervisor" "synthetic_data" "chat_agent")

# Function to show usage
show_usage() {
    echo -e "${YELLOW}Usage: $0 [agent_name]${NC}"
    echo -e "${YELLOW}Available agents:${NC}"
    for agent in "${AGENTS[@]}"; do
        echo "  - $agent"
    done
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 data_agent    # View data_agent logs"
    echo "  $0 all           # View all logs combined"
    exit 1
}

# Check if an agent name was provided
if [ $# -eq 0 ]; then
    show_usage
fi

AGENT=$1

# Handle "all" option
if [ "$AGENT" = "all" ]; then
    echo -e "${GREEN}==================================================${NC}"
    echo -e "${GREEN}  Viewing All Agent Logs                          ${NC}"
    echo -e "${GREEN}==================================================${NC}"
    
    cat /dev/null > logs/combined.log
    for agent in "${AGENTS[@]}"; do
        if [ -f "logs/${agent}.log" ]; then
            echo -e "\n==== ${agent} log ====" >> logs/combined.log
            cat "logs/${agent}.log" >> logs/combined.log
        fi
    done
    
    tail -n 50 logs/combined.log
    echo -e "${BLUE}Now following all logs (Ctrl+C to exit)...${NC}"
    tail -f logs/*.log
    exit 0
fi

# Check if the agent is valid
VALID_AGENT=0
for agent in "${AGENTS[@]}"; do
    if [ "$AGENT" = "$agent" ]; then
        VALID_AGENT=1
        break
    fi
done

if [ $VALID_AGENT -eq 0 ]; then
    echo -e "${RED}Invalid agent name: $AGENT${NC}"
    show_usage
fi

# Ensure logs directory exists
LOG_FILE="logs/${AGENT}.log"

if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}${AGENT} log file not found. Is the agent running?${NC}"
    exit 1
fi

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  ${AGENT} Log Viewer                             ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo
echo -e "${YELLOW}Showing last 20 lines of the log (Ctrl+C to exit):${NC}"
echo

# First show the last 20 lines
tail -n 20 "$LOG_FILE"

echo
echo -e "${BLUE}Now following log updates (Ctrl+C to exit)...${NC}"
echo

# Then follow the log
tail -f "$LOG_FILE"
