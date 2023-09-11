# National League 2023/24 Predictions
This project allows you to forecast the rest of a league season in Python/NumPy ⚽️

Although this project can be modified for any league supported by the [Api-Football](https://www.api-football.com) API, it is currently setup to forecast the remainder of the 2023/24 National League season. After simulating, you can produce a table like so:
![Table showing NL 2023/24 forecasts as of 11/09/2023](/images/national_league_table.jpg "Table showing NL 2023/24 forecasts as of 11/09/2023")

Furthermore, there is code to produce a plot of point distributions:
![Table showing point distributions of NL 2023/24 as of 11/09/2023](/images/national_league.png "Table showing point distributions of NL 2023/24 as of 11/09/2023")

## Setup
* Install the necessary modules in requirements.txt:
    ```
    pip install -r requirements.txt
    ```
* Get an API key from [Api-Football](https://www.api-football.com). This is free for up to 100 requests per day.
* Replace _your_actual_api_key_ in .env.sample with your API key, and change the file name to .env
* Simulate the remainder of the season in notebooks/main.ipynb. The _simulations_ parameter determines the number of simulations to perform.