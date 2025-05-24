```python
import argparse
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to execute the script.
    """
    try:
        # Argument parsing (optional - can be extended)
        parser = argparse.ArgumentParser(
            description="A simple script for demonstration purposes."
        )
        # Add arguments here if needed
        args = parser.parse_args()

        # Your script logic here
        logger.info("Script started.")
        
        # Example: Simulate some processing
        result = {"message": "Hello from the AI!", "timestamp": json.dumps(list(datetime.datetime.now()))}
        
        logger.info("Script finished successfully.")
        
        # Return results in JSON format
        print(json.dumps(result))

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        # Handle the error appropriately (e.g., exit with an error code)
        sys.exit(1)


if __name__ == "__main__":
    main()
```