# Makefile for Predictive Maintenance System
# This file allows running the system scripts without execute permissions

.PHONY: all setup run stop reboot status stop-reboot logs-all logs-data logs-prediction logs-simulation logs-supervisor logs-synthetic-data logs-chat

# Default target - shows available commands
all:
	@echo "Predictive Maintenance System Commands:"
	@echo "  make setup           - Set up the system"
	@echo "  make run             - Run the system"
	@echo "  make stop            - Stop all agents"
	@echo "  make reboot          - Reboot all agents"
	@echo "  make status          - Display agent status"
	@echo "  make stop-reboot     - Open the stop/reboot interface"
	@echo "  make logs-all        - View all logs combined"
	@echo "  make logs-data       - View data agent logs"
	@echo "  make logs-prediction - View prediction agent logs"
	@echo "  make logs-simulation - View simulation agent logs"
	@echo "  make logs-supervisor - View supervisor logs"
	@echo "  make logs-synthetic-data - View synthetic data generator logs"
	@echo "  make logs-chat       - View chat agent logs"

# Setup the system
setup:
	bash setup.sh

# Run the system
run:
	bash run.sh

# Stop all agents
stop:
	bash stop_reboot.sh 1

# Reboot all agents
reboot:
	bash stop_reboot.sh 2

# Display agent status
status:
	bash stop_reboot.sh 13

# Open the stop/reboot interface
stop-reboot:
	bash stop_reboot.sh

# Log viewing commands
logs-all:
	@bash view_logs.sh all

logs-data:
	@bash view_logs.sh data_agent

logs-prediction:
	@bash view_logs.sh prediction_agent

logs-simulation:
	@bash view_logs.sh simulation_agent

logs-supervisor:
	@bash view_logs.sh supervisor

logs-synthetic-data:
	@bash view_logs.sh synthetic_data

logs-chat:
	@bash view_logs.sh chat_agent
