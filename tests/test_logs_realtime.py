#!/usr/bin/env python3
"""Test script to verify real-time log updates."""

import time
import logging

# Setup logging to the Copilot AI log file
logging.basicConfig(
    filename='copilot-ai.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s - %(message)s'
)

logger = logging.getLogger('test_realtime')


def main() -> None:
    print("Starting real-time log test...")
    print("Open http://localhost:8000/logs in your browser")
    print("You should see new log entries appearing every 2 seconds")
    print("Press Ctrl+C to stop")

    try:
        counter = 0
        while True:
            counter += 1
            logger.info(f"Test log entry #{counter} - timestamp: {time.time()}")
            print(f"Wrote log entry #{counter}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nTest stopped")
        logger.info("Test stopped by user")


if __name__ == "__main__":
    main()
