import argparse
import requests
from bs4 import BeautifulSoup

def main():
    parser = argparse.ArgumentParser(description="Summarize the results of the 2023 NBA Finals.")
    parser.add_argument("--query", type=str, default="test query", help="Query to search for game results.")
    args = parser.parse_args()

    url = f"https://www.nba.com/finals/2023/results/{args.query}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract game results - This is a placeholder, adapt to the actual HTML structure
        game_results = []
        for game in soup.find_all('div', class_='game-result'):
            date = game.find('span', class_='date').text
            team1 = game.find('span', class_='team team-name').text
            team2 = game.find('span', class_='team team-name').text
            score1 = game.find('span', class_='score').text if game.find('span', class_='score') else "N/A"
            score2 = game.find('span', class_='score').text if game.find('span', class_='score') else "N/A"
            game_results.append(f"{date}: {team1} vs {team2} - {score1} vs {score2}")

        if game_results:
            print("2023 NBA Finals Game Results:")
            for result in game_results:
                print(result)
        else:
            print("No game results found for the given query.")

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()