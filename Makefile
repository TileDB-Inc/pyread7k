.ONESHELL:
.PHONY: help lint format check

help:
	@echo "Makefile for the pyread7k package"
	@echo "Uses poetry as a package manager"
	@echo "Valid make targets are:"
	@echo "  check            -  Format the code and run linting"
	@echo "  lint             -  Check code quality and security"
	@echo "  format           -  Format the code"

BLUE='\033[0;34m'
NC='\033[0m'

format:
	@poetry run isort *.py pyread7k/

lint: format
	@echo "\n${BLUE}Running Pylint against source files...${NC}\n"
	@poetry run pylint --rcfile=setup.cfg pyread7k/*.py --exit-zero
	@echo "\n${BLUE}Running Flake8 against source files...${NC}\n"
	@poetry run flake8 pyread7k/*.py --exit-zero
	@echo "\n${BLUE}Running Black against source files...${NC}\n"
	@poetry run black pyread7k/*.py
	@echo "\n${BLUE}Running Bandit against source files...${NC}\n"
	@poetry run bandit -r --ini setup.cfg

check: lint