import argparse
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_duckduckgo_response(response_text):
    """
    Parses a DuckDuckGo search API response.
    Handles potential JSON decoding errors and returns None on failure.
    """
    try:
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON response.")
        return None

def analyze_response(query, api_url):
    """
    Fetches data from the DuckDuckGo API, parses the response, and performs schema validation.
    """
    try:
        response = requests.get(api_url, params={'q': query})
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        parsed_data = parse_duckduckgo_response(response.text)
        if parsed_data:
            logging.info("Response parsed successfully.")
            return parsed_data
        else:
            logging.warning("Failed to parse response data.")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        return None

def main(query=None, api_url="https://api.duckduckgo.com/?q={}&format=json"):
    """
    Main function to orchestrate the analysis.
    """
    if not query:
        query = "test query"
        logging.info(f"Using default query: {query}")

    parsed_data = analyze_response(query, api_url)

    if parsed_data:
        print(json.dumps(parsed_data, indent=4))
    else:
        print("Failed to retrieve and parse data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DuckDuckGo search API responses.")
    parser.add_argument("query", nargs="?", default=None, help="The search query.")
    args = parser.parse_args()

    if args.query is None:
        main()
    else:
        main(args.query)