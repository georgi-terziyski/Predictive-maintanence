#!/bin/bash
# Setup script for the Predictive Maintenance System

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=====================================================${NC}"
echo -e "${YELLOW}  Predictive Maintenance System - Setup Tool         ${NC}"
echo -e "${YELLOW}=====================================================${NC}"
echo

# Check if PostgreSQL is installed
command -v psql >/dev/null 2>&1 || { 
    echo -e "${RED}PostgreSQL is not installed. Please install PostgreSQL before continuing.${NC}" 
    exit 1
}

# Ensure required Python packages are installed
echo -e "${YELLOW}Checking and installing required Python packages...${NC}"
pip install -r requirements.txt

# Check if .env file exists, create if not
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from .env.example...${NC}"
    cp .env.example .env
    
    # Prompt for database credentials
    echo -e "${YELLOW}Please enter your database credentials:${NC}"
    read -p "Database Username: " db_user
    read -sp "Database Password: " db_pass
    echo
    
    # Update the .env file with the provided credentials
    echo -e "${YELLOW}Updating .env file with provided credentials...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS uses BSD sed which requires a different syntax
        sed -i '' "s/DB_USER=your_username_here/DB_USER=$db_user/" .env
        sed -i '' "s/DB_PASSWORD=your_password_here/DB_PASSWORD=$db_pass/" .env
    else
        # Linux and other systems use GNU sed
        sed -i "s/DB_USER=your_username_here/DB_USER=$db_user/" .env
        sed -i "s/DB_PASSWORD=your_password_here/DB_PASSWORD=$db_pass/" .env
    fi
    
    echo -e "${GREEN}Credentials saved to .env file.${NC}"
fi

# Source the environment variables
echo -e "${YELLOW}Loading environment variables...${NC}"
source .env

# Check database connection
echo -e "${YELLOW}Checking database connection...${NC}"
if psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT 1" >/dev/null 2>&1; then
    echo -e "${GREEN}Database connection successful.${NC}"
else
    echo -e "${RED}Cannot connect to database. Please check your credentials in .env${NC}"
    echo -e "${YELLOW}Do you want to create the database? (y/n)${NC}"
    read create_db
    if [ "$create_db" = "y" ]; then
        echo -e "${YELLOW}Creating database '${DB_NAME}'...${NC}"
        createdb -h $DB_HOST -U $DB_USER $DB_NAME
        
        echo -e "${YELLOW}Initializing database schema...${NC}"
        psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f data/schema.sql
        psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f data/defaults.sql
        
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

# Create PIDs directory if it doesn't exist
if [ ! -d "agents/pids" ]; then
    echo -e "${YELLOW}Creating directory for PID files...${NC}"
    mkdir -p agents/pids
    echo -e "${GREEN}PID directory created.${NC}"
fi

# Show menu of additional setup options
echo -e "${YELLOW}What would you like to do?${NC}"
echo "1. Import sensor data from CSV"
echo "2. Exit"
read -p "Enter your choice (1-2): " choice

case $choice in
    1)
        echo -e "${YELLOW}Running data import script...${NC}"
        python data/import_data.py
        ;;
    2)
        echo -e "${GREEN}Setup complete. Exiting.${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Setup complete! You can now run the system using ./run.sh${NC}"
