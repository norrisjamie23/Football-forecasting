import http.client
import json
import os
from dotenv import load_dotenv
import json

# Load environment variables from a .env file
load_dotenv()

class DataFetcher:
    """
    Class for fetching football data from a given API.

    Attributes:
    - conn (https.client.HTTPSConnection): Connection to the API server.
    - api_key (str): API key for authentication, fetched from environment variables.
    - league (int): League ID for which data is to be fetched.
    - season (int): Season year for which data is to be fetched.
    - headers (dict): Headers required for making API requests.

    Methods:
    - make_request: Makes an API request and returns the JSON response.
    - get_standings: Fetches the standings data for the given league and season.
    - get_results: Fetches the fixtures data for the given league, season, and status.
    """
    
    def __init__(self, league=43, season=2023):
        """
        Initialize the DataFetcher class with optional league and season parameters.
        
        Parameters:
        - league (int): League ID for which data is to be fetched (default is 43).
        - season (int): Season year for which data is to be fetched (default is 2023).
        """
        self.conn = http.client.HTTPSConnection("v3.football.api-sports.io")
        self.api_key = os.getenv("FOOTBALL_API_KEY")  # Get API key from environment variable
        self.league = league
        self.season = season
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "v3.football.api-sports.io"
        }

    def make_request(self, request_url):
        """
        Make a request to a given URL and return the JSON response.
        
        Parameters:
        - request_url (str): URL to which the API request is to be made.
        
        Returns:
        dict: The JSON response from the API, or None if an error occurs.
        """
        try:
            self.conn.request("GET", request_url, headers=self.headers)
            res = self.conn.getresponse()
            data = res.read()
            return json.loads(data.decode("utf-8"))["response"]
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_standings(self):
        """
        Fetch the standings data for the given league and season.
        
        Returns:
        dict: The JSON response from the API.
        """
        request_url = f"/standings?league={self.league}&season={self.season}"

        return self.make_request(request_url)
    
    def get_results(self, status="FT"):
        """
        Fetch the fixtures data for the given league, season, and status.
        
        Parameters:
        - status (str): The status of the fixtures to fetch (default is "FT" for Full Time).
        
        Returns:
        dict: The JSON response from the API.
        """
        request_url = f"/fixtures?league={self.league}&season={self.season}&status={status}"

        return self.make_request(request_url)    
