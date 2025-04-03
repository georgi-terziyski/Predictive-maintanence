#!/bin/bash
# Setup and run script for the Predictive Maintenance System

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=====================================================${NC}"
echo -e "${YELLOW}  Predictive Maintenance System - Setup and Run Tool  ${NC}"
echo -e "${YELLOW}=====================================================${NC}"
echo

# Check if PostgreSQL is installed
command -v psql >/dev/null 2>&1 || { 
    echo -e "${RED}PostgreSQL is not installed. Please install PostgreSQL before continuing.${NC}" 
    exit 1
}

# Ensure required Python packages are installed
echo -e "${YELLOW}Checking and installing required Python packages...${NC}"
pip install -r agents/requirements.txt

# Check if .env file exists, create if not
if [ ! -f "agents/.env" ]; then
    echo -e "${YELLOW}Creating .env file from .env.example...${NC}"
    cp agents/.env.example agents/.env
    echo -e "${GREEN}Created .env file. Please edit agents/.env to set your database credentials.${NC}"
    echo -e "${YELLOW}Press Enter to continue after editing the .env file, or Ctrl+C to cancel.${NC}"
    read
fi

# Source the environment variables
echo -e "${YELLOW}Loading environment variables...${NC}"
source agents/.env

# Check database connection
echo -e "${YELLOW}Checking database connection...${NC}"
if psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1" >/dev/null 2>&1; then
    echo -e "${GREEN}Database connection successful.${NC}"
else
    echo -e "${RED}Cannot connect to database. Please check your credentials in agents/.env${NC}"
    echo -e "${YELLOW}Do you want to create the database? (y/n)${NC}"
    read create_db
    if [ "$create_db" = "y" ]; then
        echo -e "${YELLOW}Creating database '${DB_NAME}'...${NC}"
        createdb -h $DB_HOST -U $DB_USER $DB_NAME
        
        echo -e "${YELLOW}Initializing database schema...${NC}"
        psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f data/schema.sql
        psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f data/maintenance_table.sql
        
        echo -e "${GREEN}Database setup complete.${NC}"
    else
        echo -e "${RED}Exiting. Please set up the database manually.${NC}"
        exit 1
    fi
fi

# Create placeholder ML model if needed
if [ ! -f "agents/models/failure_prediction_model.pkl" ]; then
    echo -e "${YELLOW}Creating placeholder machine learning model...${NC}"
    python agents/models/create_placeholder_model.py
    echo -e "${GREEN}Placeholder model created.${NC}"
fi

# Show menu of options
echo -e "${YELLOW}What would you like to do?${NC}"
echo "1. Import sensor data from CSV"
echo "2. Run all agents"
echo "3. Run data agent only"
echo "4. Run prediction agent only"
echo "5. Exit"
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo -e "${YELLOW}Running data import script...${NC}"
        python data/import_data.py
        ;;
    2)
        echo -e "${YELLOW}Starting all agents...${NC}"
        echo -e "${GREEN}Starting data agent on port ${DATA_AGENT_PORT}...${NC}"
        python agents/data_agent/app.py &
        DATA_AGENT_PID=$!
        
        echo -e "${GREEN}Starting prediction agent on port ${PREDICTION_AGENT_PORT}...${NC}"
        python agents/prediction_agent/app.py &
        PREDICTION_AGENT_PID=$!
        
        echo -e "${GREEN}Starting supervisor on port ${SUPERVISOR_PORT}...${NC}"
        python agents/supervisor/app.py &
        SUPERVISOR_PID=$!
        
        echo -e "${GREEN}All agents started. Press Ctrl+C to stop.${NC}"
        wait
        ;;
    3)
        echo -e "${YELLOW}Starting data agent only...${NC}"
        python agents/data_agent/app.py
        ;;
    4)
        echo -e "${YELLOW}Starting prediction agent only...${NC}"
        python agents/prediction_agent/app.py
        ;;
    5)
        echo -e "${GREEN}Exiting.${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"
