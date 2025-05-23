import argparse
import duckducksearch
import json

def main():
    parser = argparse.ArgumentParser(description="Search the internet using DuckDuckSearch and format the results.")
    parser.add_argument("query", nargs='?', default='test query', help="The search query.")
    parser.add_argument("-o", "--output", help="Output file name for JSON results.")
    args = parser.parse_args()

    try:
        results = duckducksearch.ddg(args.query, max_results=5)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=4))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()