import numpy as np

def create_dataset():
    """
    This function reads CSV files for WC matches played in specific years and processes the data to create datasets (X, y).
    
    The function specifically reads files from 1990 to 2018 and extracts information on ELO ratings and scores for home and away teams.
    
    X: A dataset containing the ELO rating differences for home and away teams.
    y: A dataset containing the actual scores for home and away teams.
    
    Returns:
    X (numpy.ndarray): The features for the model. Each row contains the ELO rating difference for home and away teams.
    y (numpy.ndarray): The labels for the model. Each row contains the score for home and away teams.
    """
    
    # Initialize empty lists to store the features and labels
    X = []
    y = []

    # Loop over the specified years to read and process the respective CSV files
    for year in ['1990', '1994', '1998', '2002', '2006', '2010', '2014', '2018']:
        
        # Load the data from the CSV file and replace any '−' characters with '-'
        data = np.loadtxt(f"../data/{year}.csv", delimiter=',', dtype=str, usecols=0)
        data = np.char.replace(data, '−', '-')
        
        # Calculate the home and away ELO rating differences
        home_elo = data[10::16].astype(int) - data[8::16].astype(int)
        away_elo = data[11::16].astype(int) - data[9::16].astype(int)
        
        # Extract the home and away scores
        home_score = data[4::16].astype(int)
        away_score = data[5::16].astype(int)

        # Append the processed ELO rating differences and scores to the feature and label lists, respectively
        X.append(np.stack((home_elo, away_elo), axis=-1))
        y.append(np.stack((home_score, away_score), axis=-1))

    # Concatenate the lists into numpy arrays for the final feature and label datasets
    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y
