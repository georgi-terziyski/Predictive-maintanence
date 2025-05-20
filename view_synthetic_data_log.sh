#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure logs directory exists
LOG_FILE="logs/synthetic_data.log"

if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}Synthetic Data log file not found. Is the Synthetic Data agent running?${NC}"
    exit 1
fi

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Synthetic Data Generator Log Viewer             ${NC}"
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
