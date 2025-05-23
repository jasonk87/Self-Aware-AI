import argparse
import requests
from bs4 import BeautifulSoup

def main():
    parser = argparse.ArgumentParser(description="Identify the winners of the Eastern and Western Conference playoffs.")
    parser.add_argument("--year", type=int, default=2023, help="The year for which to retrieve playoff data. Defaults to 2023.")
    args = parser.parse_args()

    try:
        url = f"https://www.nba.com/playoffs/{args.year}"
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        eastern_winner = soup.find("div", class_="playoffs-conference-winner eastern").text.strip()
        western_winner = soup.find("div", class_="playoffs-conference-winner western").text.strip()

        print(f"Eastern Conference Winner ({args.year}): {eastern_winner}")
        print(f"Western Conference Winner ({args.year}): {western_winner}")

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
    except AttributeError as e:
        print(f"Error parsing HTML: {e}.  The website structure may have changed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()