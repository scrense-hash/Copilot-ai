#!/bin/bash
# Script to run Autorouter tests

set -e  # Exit on error

echo "=================================="
echo "Autorouter Test Suite Runner"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -d ".venv" ]]; then
        echo -e "${GREEN}Activating virtual environment...${NC}"
        source .venv/bin/activate
    elif [[ -d "venv" ]]; then
        echo -e "${GREEN}Activating virtual environment...${NC}"
        source venv/bin/activate
    else
        echo -e "${YELLOW}Warning: Virtual environment not found${NC}"
        echo "Create one with: python -m venv .venv"
        echo "Then install dependencies: pip install -r tests/requirements-dev.txt"
        echo ""
    fi
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install it with: pip install -r tests/requirements-dev.txt"
    exit 1
fi

# Parse arguments
COVERAGE=false
VERBOSE=false
PARALLEL=false
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --parallel|-p)
            PARALLEL=true
            shift
            ;;
        --test|-t)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --coverage, -c    Run with coverage report"
            echo "  --verbose, -v     Run with verbose output"
            echo "  --parallel, -p    Run tests in parallel"
            echo "  --test, -t FILE   Run specific test file"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Run all tests"
            echo "  $0 -c                   # Run with coverage"
            echo "  $0 -v -c                # Verbose with coverage"
            echo "  $0 -t test_config.py    # Run specific file"
            echo "  $0 -p                   # Run in parallel"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests"

if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=. --cov-config=tests/.coveragerc --cov-report=term-missing --cov-report=html"
fi

if [[ "$PARALLEL" == true ]]; then
    # Check if pytest-xdist is installed
    if python -c "import xdist" 2>/dev/null; then
        PYTEST_CMD="$PYTEST_CMD -n auto"
    else
        echo -e "${YELLOW}Warning: pytest-xdist not installed, running sequentially${NC}"
        echo "Install with: pip install pytest-xdist"
        echo ""
    fi
fi

if [[ -n "$SPECIFIC_TEST" ]]; then
    # If specific test doesn't have 'tests/' prefix, add it
    if [[ ! "$SPECIFIC_TEST" =~ ^tests/ ]]; then
        SPECIFIC_TEST="tests/$SPECIFIC_TEST"
    fi
    PYTEST_CMD="pytest $SPECIFIC_TEST"
    if [[ "$VERBOSE" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD -vv"
    else
        PYTEST_CMD="$PYTEST_CMD -v"
    fi
    if [[ "$COVERAGE" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD --cov=. --cov-config=tests/.coveragerc --cov-report=term-missing --cov-report=html"
    fi
fi

# Run tests
echo -e "${GREEN}Running tests...${NC}"
echo "Command: $PYTEST_CMD"
echo ""

$PYTEST_CMD

# Check exit code
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}=================================="
    echo "All tests passed! ✓"
    echo -e "==================================${NC}"

    if [[ "$COVERAGE" == true ]]; then
        echo ""
        echo "Coverage report generated:"
        echo "  - Terminal: above"
        echo "  - HTML: htmlcov/index.html"
    fi
else
    echo ""
    echo -e "${RED}=================================="
    echo "Some tests failed! ✗"
    echo -e "==================================${NC}"
    exit 1
fi
