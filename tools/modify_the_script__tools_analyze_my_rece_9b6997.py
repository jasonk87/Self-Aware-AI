import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Analyze performance data.")
    parser.add_argument("query", nargs='?', default='test query', help="The query to analyze.")
    parser.add_argument("--filename", help="The filename to use.")
    args = parser.parse_args()

    if args.filename:
        try:
            with open(args.filename, 'r') as f:
                data = f.read()
                print(f"Data from {args.filename}:\n{data}")
        except FileNotFoundError:
            print(f"Error: File not found: {args.filename}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        print(f"Analyzing query: {args.query}")

if __name__ == "__main__":
    main()