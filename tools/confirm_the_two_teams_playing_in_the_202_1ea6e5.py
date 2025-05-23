import argparse
import requests
from bs4 import BeautifulSoup

def main():
    parser = argparse.ArgumentParser(description="Confirm the two teams playing in the 2023 NBA Finals.")
    # No arguments are strictly required for this specific goal.
    # Using a default query for demonstration.
    parser.add_argument('--query', type=str, default='2023 NBA Finals', help='Query for the NBA Finals.')
    args = parser.parse_args()

    try:
        url = f"https://www.nba.com/finals/{args.query.lower().replace(' ', '-')}"
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract team names - this is a simplified approach and may need adjustment
        # based on the actual HTML structure of the NBA.com website.
        teams = soup.find_all('h1', class_='title')
        if teams:
            team_names = [team.text for team in teams]
            print(f"The teams playing in the {args.query} are: {', '.join(team_names)}")
        else:
            print(f"Could not find team names for the {args.query} on NBA.com.")

    except requests.exceptions.RequestException as e:
        print(f"Error during request to NBA.com: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()