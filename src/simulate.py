# Data Manipulation Libraries
import numpy as np
import pandas as pd

# Utilities
from tqdm import tqdm
from functools import partial 
import datetime

# Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns


tqdm = partial(tqdm, position=0, leave=True)

class Season:
    """
    The `Season` class is responsible for managing and simulating a league season, 
    primarily focused on maintaining team Elo ratings, league standings, future fixtures, 
    and postponed fixtures.

    Attributes:
    - ELOs (dict): A dictionary mapping team names to their respective ELO ratings.
    - data_fetcher (Object): An object responsible for fetching various types of data such as
                             current league standings and match results.
    - model (model): Model to simulate with.
    """

    def __init__(self, elo_csv_path, data_fetcher, model):
        """
        Initialize the class with initial ELO ratings and a data fetcher object.
        
        Parameters:
        - elo_csv_path (str): The path to the CSV file containing initial ELO ratings.
        - data_fetcher (Object): An object responsible for fetching data.
        """
        # Initialize ELO ratings from the given CSV file
        self.ELOs = self.get_init_ELOs(elo_csv_path)
        
        # Initialize data fetcher
        self.data_fetcher = data_fetcher

        # Initialize data fetcher
        self.model = model

    def get_init_ELOs(self, elo_csv_path):
        """
        Fetch the initial ELO ratings for teams from a CSV file.
        
        Parameters:
        - elo_csv_path (str): The path to the CSV file containing initial ELO ratings.
        
        Returns:
        - ELOs_init_dict (dict): A dictionary mapping team names to their respective initial ELO ratings.
        """
        
        # Load data from the CSV file
        ELOs_init = np.loadtxt(elo_csv_path, delimiter=',', dtype=str)
        
        # Extract team names and their ELO ratings
        teams = ELOs_init[:, 1]
        team_ELOs = ELOs_init[:, 2].astype(int)
        
        # Create a dictionary to store team names and their ELO ratings
        ELOs_init_dict = {team: team_ELO for team, team_ELO in zip(teams, team_ELOs)}

        return ELOs_init_dict
    
    def get_future_fixtures(self):
        """
        Fetch and organize future fixtures by their date.

        Returns:
        - future_fixtures (List): A sorted list of tuples where each tuple contains a date and the corresponding list of fixtures.
        """
        
        # Fetch fixtures with status "Not Started"
        fixtures = self.data_fetcher.get_results(status="NS")
        
        # Dictionary to store future fixtures organized by date
        future_fixtures = {}

        # Populate the future_fixtures dictionary
        for fixture in fixtures:
            # Extract date from fixture information
            date = datetime.datetime.strptime(fixture['fixture']['date'][:10], '%Y-%m-%d')
            
            # Add fixture to the appropriate date entry in the dictionary
            if date not in future_fixtures:
                future_fixtures[date] = [[fixture['teams']['home']['name'], fixture['teams']['away']['name']]]
            else:
                future_fixtures[date].append([fixture['teams']['home']['name'], fixture['teams']['away']['name']])
        
        # Sort future fixtures by date
        future_fixtures = sorted(future_fixtures.items())

        return future_fixtures
    
    def get_postponed_fixtures(self):
        """
        Fetch and organize fixtures that have been postponed.

        Returns:
        - postponed_fixtures_to_simulate (List): A list containing the names of the home and away teams for each postponed fixture.
        """
        
        # Fetch fixtures with the status "PST" for postponed fixtures
        fixtures = self.data_fetcher.get_results(status="PST")
        
        # Initialize a list to store fixtures that have been postponed
        postponed_fixtures_to_simulate = []

        # Populate the list with postponed fixtures
        for postponed_fixture in fixtures:
            postponed_fixtures_to_simulate.append([postponed_fixture['teams']['home']['name'], 
                                                postponed_fixture['teams']['away']['name']])

        return postponed_fixtures_to_simulate

    def get_table_df(self):
        """
        Generate a DataFrame representing the league table along with ELO scores.

        Parameters:
        - data_fetcher (Object): An object with methods to fetch results and standings.
        
        Returns:
        - all_teams_df (DataFrame): A DataFrame containing team names, points, goals for, goals against, and ELO scores.
        """
        ELOs_copy = self.ELOs.copy()
        
        # Fetch match results and current league standings
        results = self.data_fetcher.get_results()
        standings = self.data_fetcher.get_standings()

        # Lists to store today's games and scores
        todays_games = []
        todays_scores = []

        # Iterate through match results to update ELOs and identify today's matches
        for result in results:
            curr_results = np.array([[result['score']['fulltime']['home'], result['score']['fulltime']['away']]])
            ELOs = np.array([[self.ELOs[result['teams']['home']['name']], self.ELOs[result['teams']['away']['name']]]])
            Ws = np.array(get_Ws(curr_results, ELOs))
            
            # Check if the game is from today
            date = datetime.datetime.strptime(result['fixture']['date'][:10], '%Y-%m-%d')
            if str(date)[:10] == str(datetime.datetime.today())[:10]:
                todays_games.append([result['teams']['home']['name'], result['teams']['away']['name']])
                todays_scores.append([result['score']['fulltime']['home'], result['score']['fulltime']['away']])
            
            # Update ELO scores
            elo_adjustments = get_elo_adjustments(ELOs, Ws, curr_results)
            self.ELOs[result['teams']['home']['name']] += elo_adjustments[0, 0]
            self.ELOs[result['teams']['away']['name']] += elo_adjustments[0, 1]

        # Initialize DataFrame to store league table and ELOs
        df_list = [[team, 0, 0, 0, elo] for team, elo in self.ELOs.items()]
        all_teams_df = pd.DataFrame(df_list, columns=['Team', 'Points', 'GF', 'GA', 'ELO'])

        # Update DataFrame based on current league standings
        for team in standings[0]['league']['standings'][0]:
            team_name = team['team']['name']
            all_teams_df.loc[all_teams_df.Team == team_name, 'Points'] = team['points']
            all_teams_df.loc[all_teams_df.Team == team_name, 'GF'] = team['all']['goals']['for']
            all_teams_df.loc[all_teams_df.Team == team_name, 'GA'] = team['all']['goals']['against']

        # Update table based on today's games
        for todays_game, todays_score in zip(todays_games, todays_scores):
            if todays_score[0] > todays_score[1]:
                home_points = 3
                away_points = 0
            elif todays_score[0] < todays_score[1]:
                home_points = 0
                away_points = 3
            else:
                home_points = 1
                away_points = 1
            
            # Update for home team
            all_teams_df.loc[all_teams_df.Team == todays_game[0], 'Points'] += home_points
            all_teams_df.loc[all_teams_df.Team == todays_game[0], 'GF'] += todays_score[0]
            all_teams_df.loc[all_teams_df.Team == todays_game[0], 'GA'] += todays_score[1]

            # Update for away team
            all_teams_df.loc[all_teams_df.Team == todays_game[1], 'Points'] += away_points
            all_teams_df.loc[all_teams_df.Team == todays_game[1], 'GF'] += todays_score[1]
            all_teams_df.loc[all_teams_df.Team == todays_game[1], 'GA'] += todays_score[0]
        
        return all_teams_df
    
    def simulate_season(self, simulations=10000):
        """
        Simulate the outcomes of a season for a specified number of times.
        
        This function uses existing data about teams' performance and future fixtures
        to simulate the outcome of the season. It uses Elo ratings to update teams' performance
        and applies the model for the simulations.
        
        Parameters:
        - simulations (int): The number of times the season is to be simulated. Default is 10,000.
        
        Returns:
        - all_simulated_dfs (List[pd.DataFrame]): A list containing pandas DataFrames for each simulation. 
        Each DataFrame represents the final league table for that simulation.
        
        Notes:
        - The function uses tqdm for a progress bar, offering visual feedback during simulations.
        - The function relies on the methods `get_future_fixtures`, `get_postponed_fixtures`, and `get_table_df` 
        for initial data. 
        - The function also uses another method, `simulate_future_fixtures`, to perform the actual simulations 
        for future and postponed fixtures.
        """
        
        all_simulated_dfs = []

        # Fetch data for future fixtures, postponed fixtures, and current team standings
        future_fixtures = self.get_future_fixtures()
        postponed_fixtures_to_simulate = self.get_postponed_fixtures()
        all_teams_df = self.get_table_df()
        
        # Loop through each simulation
        for _ in tqdm(range(simulations)):
            all_teams_df_copy = all_teams_df.copy()

            # Simulate future fixtures and update ELO ratings
            simulate_future_fixtures(all_teams_df_copy, future_fixtures, update_elos=True, model=self.model)
            
            # Create a list of postponed fixtures to simulate
            day_fixtures = [[0, postponed_fixtures_to_simulate]]
            
            # Simulate postponed fixtures without updating ELO ratings
            simulate_future_fixtures(all_teams_df_copy, day_fixtures, update_elos=False, model=self.model)
            
            # Calculate goal difference for each team
            all_teams_df_copy.loc[:, 'GD'] = all_teams_df_copy.loc[:, 'GF'] - all_teams_df_copy.loc[:, 'GA']
            
            # Sort the DataFrame based on Points, Goal Difference, and Goals For, in that order
            all_teams_df_copy = all_teams_df_copy.sort_values(['Points', 'GD', 'GF'], ascending=[False, False, False]).reset_index(drop=True)
            
            # Append the sorted DataFrame to the list
            all_simulated_dfs.append(all_teams_df_copy)

        return all_simulated_dfs


def get_Ws(results, ELOs):
    """Returns result (1 for W, 0 for L, 0.5 for D) for the team with a higher ELO

    Parameters
    ----------
    results : ndarray (int)
        A 2D array of scores.

    ELOs : ndarray (float)
        A 2D array of ELOs for the teams in results.

    Returns
    ------
    Ws : ndarray
        The result for the "strongest" team in each fixture.
    """

    # Initialise Ws array
    Ws = np.zeros(ELOs.shape[:1])
    
    # Mask for games in which "home" and "away" teams won respectively
    team1_wins = results[:, 0] > results[:, 1] #+ #eps
    team2_wins = results[:, 1] > results[:, 0] #+ #eps
    
    wins = np.logical_or(team1_wins, team2_wins)
    
    # Mask for games that ended in draws
    draws = np.logical_not(wins)
        
    # Matches with an "away" team that is stronger
    away_better = np.argmax(ELOs, axis=-1).astype(bool)

    # As above for "home"
    home_better = np.logical_not(away_better)
    
    # If "home" team won and they had a higher ELO, W is 1
    Ws[np.logical_and(home_better, team1_wins)] = 1
    
    # If "away" team won and they had a higher ELO, W is 1
    Ws[np.logical_and(away_better, team2_wins)] = 1
    
    # For draws, W is 1. Remaining values are already 0 so no need to change
    Ws[draws] = 0.5
    
    return Ws


def get_elo_adjustments(ELOs, Ws, results):
    """Get adjustment for ELOs of each team given the final score in each game.

    Parameters
    ----------
    results : ndarray (int)
        A 2D array of scores.

    ELOs : ndarray (float)
        A 2D array of ELOs for the teams in results.

    Ws : ndarray (float)
        A 1D array of results for the "strongest" team in each fixture.

    Returns
    ------
    ELO_changes_arr : ndarray
        Of the same shape as results, this says how much to adjust the ELO by.
        Each element corresponds to the same team in ELOs/results.
    """

    # Get difference in ELOs for each fixture
    drs = np.abs(ELOs[:, 0] - ELOs[:, 1] + 90)
    
    # Get probability of higher ELO team winning
    Ws_e = 1 / (10**(-drs/400) + 1)
    
    # Work out weight constant
    Ks = np.ones(ELOs.shape[0]) * 20
    
    score_diffs = np.abs(results[:, 0] - results[:, 1])
    Ks[np.where(score_diffs == 2)] *= 1.5
    Ks[np.where(score_diffs > 2)] *= 1 + (3/4 + (score_diffs[score_diffs > 2]-3)/8)
    
    # How much will ELOs change?
    ELO_changes = Ks * (Ws - Ws_e)
    
    # Put this into array showing changes for each team
    ELO_changes_arr = np.zeros((ELOs.shape[0], 2))
    ELO_changes_arr[np.arange(ELOs.shape[0]), np.argmax(ELOs, axis=1)] = ELO_changes
    ELO_changes_arr[np.arange(ELOs.shape[0]), 1 - np.argmax(ELOs, axis=1)] = -ELO_changes
            
    return ELO_changes_arr


def simulate_future_fixtures(all_teams_df, future_fixtures, update_elos, model):
    """
    Simulate future fixtures for a league based on teams' current ELO ratings and update the league table accordingly.
    
    This function updates the goals scored, goals against, and points columns in the league table. It optionally updates
    ELO ratings for each team based on the simulation results.
    
    Parameters:
    - all_teams_df (pd.DataFrame): DataFrame containing the current league table with columns 'Team', 'GF', 'GA', 'ELO', and 'Points'.
    - future_fixtures (List): List of fixtures to be simulated.
    - update_elos (bool): Flag to indicate whether to update ELO ratings after simulation.
    - model (Model): A model object to simulate match results based on ELO ratings.

    Notes:
    - The function expects 'all_teams_df' to have columns: 'Team', 'GF', 'GA', 'ELO', and 'Points'.
    - The model object passed should have a 'simulate_matches' method that takes ELO ratings and returns simulation results.
    - Function uses NumPy for numerical operations for efficiency.
    - Handles cases where no fixtures are available for a given day.
    """
    
    for day_fixtures in future_fixtures:
        try:
            # Extract ELO ratings for each team in each fixture
            ELOs = np.vectorize(dict(zip(all_teams_df.Team, all_teams_df.ELO)).get)(day_fixtures[1]).astype(np.float32)
        except ValueError:
            # Handle the case where no fixtures are available
            break
        
        fixtures = np.array(day_fixtures[1])
        
        results = model.simulate_matches(ELOs, lam_multiplier=1, home_advantage=111)
                
        # Get column indices for GF and GA
        GF_col = all_teams_df.columns.get_loc("GF")
        GA_col = all_teams_df.columns.get_loc("GA")

        # Update goals scored/conceded for "home" team
        all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[:, 0]), GF_col] += results[:, 0]
        all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[:, 1]), GA_col] += results[:, 0]

        # Update goals scored/conceded for "away" team
        all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[:, 1]), GF_col] += results[:, 1]
        all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[:, 0]), GA_col] += results[:, 1]

        if update_elos:

            # Index of ELO column
            ELO_col = all_teams_df.columns.get_loc("ELO")

            # Get ELO for "home" and "away" teams respectively
            team1_ELOs = np.array(all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[:, 0]), ELO_col])
            team2_ELOs = np.array(all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[:, 1]), ELO_col])

            # Combine into one ELO array
            ELOs = np.stack((team1_ELOs, team2_ELOs), axis=1).astype(np.int32)

        # Index of points column
        points_col = all_teams_df.columns.get_loc("Points")

        # Mask for games in which "home" team won
        team1_wins = results[:, 0] > results[:, 1]

        # Add 3 points for these teams
        all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[team1_wins, 0]), points_col] += 3

        # Mask for games in which "away" team won
        team2_wins = results[:, 0] < results[:, 1]

        # Add 3 points for these teams
        all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[team2_wins, 1]), points_col] += 3

        # Mask for games in which "away" team won
        draws = results[:, 0] == results[:, 1]
        
        try:
            # Add 1 point to each of these teams
            all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[draws].flatten()), points_col] += 1

        except KeyError as e:
            
            for draw_team in fixtures[draws].flatten():
                # Add 1 point to each of these teams
                all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer([draw_team]), points_col] += 1

        if update_elos:

            # Use get_Ws to get W value for ELO updating
            Ws = get_Ws(results, ELOs)

            # How much is to be added to ELO, elementwise matching teams in fixtures away
            elo_adjustments = get_elo_adjustments(ELOs, Ws, results)

            # Update ELO for "home" and "away" teams respectively
            all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[:, 0]), ELO_col] += elo_adjustments[:, 0]
            all_teams_df.iloc[pd.Index(all_teams_df.Team).get_indexer(fixtures[:, 1]), ELO_col] += elo_adjustments[:, 1]


def summarise_forecasts(all_simulated_dfs):
    """
    Generate a summary DataFrame that consolidates the results of all simulated seasons.
    The function computes average points for each team over multiple simulations and calculates
    probabilities for specific events like winning the title, finishing in the top 7, or being relegated.

    Parameters:
    all_simulated_dfs (list of pandas.DataFrame): List of DataFrames, each containing the standings of a single simulated season.

    Returns:
    pandas.DataFrame: A DataFrame containing the summary statistics for each team, sorted by average points.
    - Team: Name of the team
    - Points: Average points over all simulations
    - Title: Probability (%) of winning the title
    - Top 7: Probability (%) of finishing in the top 7 (play offs and title)
    - Relegated: Probability (%) of being relegated

    Note:
    - The function assumes that each DataFrame in all_simulated_dfs has a 'Team' column and a 'Points' column.
    - The 'Team' column contains unique identifiers for each team.
    - It also assumes that the index of each DataFrame reflects the ranking of the teams.
    """

    # Concatenate all simulated DataFrames for overall statistics
    concatenated_simulated_dfs = pd.concat((all_simulated_dfs))

    # Compute the mean points for each team and sort them
    summary_df = concatenated_simulated_dfs.groupby(['Team']).mean().sort_values('Points', ascending=False)

    # Determine the number of simulations for probability calculations
    total_simulations = len(all_simulated_dfs)

    # Iterate over each unique team
    for team in all_simulated_dfs[0].Team.unique():
        team_placings = [0, 0, 0]  # Initialize counters for [Titles, Top 7 finishes, Relegations]
        
        # Count the occurrences of each type of placing for the team across all simulations
        for df in all_simulated_dfs:
            placing = df.Team[df.Team == team].index.tolist()[0]
            if placing == 0:
                team_placings[0] += 1  # Count titles
            if placing <= 6:
                team_placings[1] += 1  # Count top 7 finishes
            elif placing > 19:
                team_placings[2] += 1  # Count relegations

        # Calculate and store the probabilities in the summary DataFrame
        summary_df.loc[summary_df.index == team, 'Title'] = 100 * team_placings[0] / total_simulations
        summary_df.loc[summary_df.index == team, 'Top 7'] = 100 * team_placings[1] / total_simulations
        summary_df.loc[summary_df.index == team, 'Relegated'] = 100 * team_placings[2] / total_simulations

    return summary_df


def plot_simulated_points_distribution(all_simulated_dfs, save_path=None):
    """
    Generate and display a plot showing the distribution of simulated points for each team.
    
    Parameters:
    all_simulated_dfs (list of pandas.DataFrame): List of DataFrames, each containing the standings of a single simulated season.
    save_path (str, optional): File path to save the generated plot. If None, the plot is not saved.
    
    Note:
    - Assumes each DataFrame in all_simulated_dfs has a 'Team' and 'Points' column.
    - The 'Team' column should contain unique identifiers for each team.
    """

    # Concatenate all simulated DataFrames for overall statistics
    concatenated_simulated_dfs = pd.concat((all_simulated_dfs))

    # Create a plot for each team's simulated points distribution
    fig, axes = plt.subplots(nrows=len(all_simulated_dfs[0]), ncols=1, figsize=(6, 8), sharex=True)
    
    # Sort teams by their average points
    teams_sorted = list(concatenated_simulated_dfs.groupby(['Team']).mean().Points.sort_values(ascending=False).index)

    # Plot KDE for each team
    for team, ax in zip(teams_sorted, axes):
        sns.kdeplot(concatenated_simulated_dfs[concatenated_simulated_dfs.Team == team], x='Points', fill=True, ax=ax,
                    linewidth=0, bw_adjust=4)
        
        # Configure axis labels and appearance
        ax.yaxis.set_label_position('left')
        ax.set_ylabel(f"{team}", rotation=0, horizontalalignment='right', y=0)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.bottom.set_edgecolor('000')
        ax.spines.bottom.set_linewidth(0.2)
        ax.tick_params(left=False, labelleft=False, bottom=False)
        ax.patch.set_visible(False)

    # Add global labels and title
    plt.xlabel('Points')
    plt.title("Simulated Points Distribution", y=28.7, fontweight=590)

    # Save or show the plot
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()
