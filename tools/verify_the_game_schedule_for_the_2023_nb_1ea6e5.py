import argparse
import requests
from bs4 import BeautifulSoup

def main():
    parser = argparse.ArgumentParser(description="Verify the game schedule for the 2023 NBA Finals.")
    parser.add_argument("--url", type=str, default="https://www.nba.com/finals/2023/schedule/", help="URL to scrape the schedule from. Defaults to the 2023 NBA Finals schedule.")
    args = parser.parse_args()

    try:
        response = requests.get(args.url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table containing the schedule.  This is highly dependent on the NBA website structure.
        # This is a placeholder - adjust the selector based on the actual HTML.
        schedule_table = soup.find('table', {'class': 'schedule-table'})

        if schedule_table:
            print("2023 NBA Finals Schedule:")
            for row in schedule_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) > 0:
                    print(cells[0].text.strip())
        else:
            print("Could not find the schedule table on the specified URL.")

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()