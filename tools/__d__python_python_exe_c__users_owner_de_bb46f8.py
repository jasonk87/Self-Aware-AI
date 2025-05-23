import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="A simple console AI tool.")
    parser.add_argument("query", nargs='?', default='test query', help="The query to process.")
    
    args = parser.parse_args()

    if args.query is None:
        print("No query provided. Using default: 'test query'")
        query = 'test query'
    else:
        query = args.query
    
    try:
        print(f"Processing query: {query}")
        # Simulate some processing
        result = f"Result for '{query}'"
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()