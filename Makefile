# Makefile for Predictive Maintenance System
# This file allows running the system scripts without execute permissions

.PHONY: all setup run stop reboot status stop-reboot

# Default target - shows available commands
all:
	@echo "Predictive Maintenance System Commands:"
	@echo "  make setup       - Set up the system"
	@echo "  make run         - Run the system"
	@echo "  make stop        - Stop all agents"
	@echo "  make reboot      - Reboot all agents"
	@echo "  make status      - Display agent status"
	@echo "  make stop-reboot - Open the stop/reboot interface"

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
	bash stop_reboot.sh 11

# Open the stop/reboot interface
stop-reboot:
	bash stop_reboot.sh
