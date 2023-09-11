from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold

import numpy as np

class Model:
    """
    This class provides functionals for fitting a multi-output model using Poisson regression.
    The class employs grid search with k-fold cross-validation for hyperparameter tuning.

    Attributes:
    single_model (estimator object): A scikit-learn compatible estimator object used as the base model.
    model (MultiOutputRegressor): The trained MultiOutputRegressor model.
    """

    def __init__(self, single_model=PoissonRegressor(fit_intercept=True)):
        """
        Initialize an instance of the class with a single model, which defaults to a Poisson Regressor.

        Parameters:
        single_model (estimator object, optional): A scikit-learn compatible estimator object to be used as the base model.
                                                Defaults to PoissonRegressor with fit_intercept set to True.

        Attributes:
        single_model (estimator object): Stores the estimator object passed as an argument or the default PoissonRegressor.
        """
        self.single_model = single_model


    def fit_model(self, X, y, n_splits=5):
        """
        Perform grid search with k-fold cross-validation to tune hyperparameters for a MultiOutputRegressor model
        with Poisson regression as the base estimator.

        Parameters:
        X (numpy.ndarray): The features for the model.
        y (numpy.ndarray): The labels for the model.
        n_splits (int): The number of folds in k-fold cross-validation.

        Returns:
        best_model (MultiOutputRegressor): The best trained model based on the grid search.
        best_params (dict): The best hyperparameters found in the grid search.
        """

        # Define the parameter grid for the Poisson Regressor
        param_grid = {
            'estimator__alpha': [0.1, 0.5, 1, 2],  # L2 regularization term
            'estimator__tol': [1e-3, 1e-4, 1e-5]   # Tolerance for stopping criteria
        }

        # Initialize the MultiOutput Regressor model
        model = MultiOutputRegressor(self.single_model)

        # Initialize k-fold cross-validation
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=2022)

        # Initialize the Grid Search model with k-fold cross-validation
        grid_search = GridSearchCV(
            model, 
            param_grid,
            cv=kfold, 
            scoring='neg_mean_squared_error',  # Negative MSE as sklearn tries to maximize the score
            verbose=1, 
            n_jobs=-1
        )

        # Perform the grid search
        grid_search.fit(X, y)

        # Set the best model and parameters based on the grid search
        self.model = grid_search.best_estimator_


    def simulate_matches(self, elo_scores, lam_multiplier=1, home_advantage=111):
        """
        Simulate the outcomes of matches using Poisson-distributed random variables. 
        The Poisson lambda parameter is obtained by predicting with the trained model.

        Parameters:
        elo_scores (numpy.ndarray): An array of ELO scores for home and away teams. 
                                    The first column represents home teams, and the second column represents away teams.
        lam_multiplier (float, optional): A multiplier for the Poisson lambda parameter. Defaults to 1.
        home_advantage (int, optional): The ELO score advantage attributed to home teams. Defaults to 111.

        Returns:
        score (numpy.ndarray): The simulated score of the matches, rounded to the nearest integer.

        Note:
        - The function adjusts the home team's ELO score by adding a home advantage before making predictions.
        """

        # Add home advantage to the ELO score of the home team
        elo_scores[:, 0] += home_advantage

        # Predict the average number of goals for each team using the trained model
        lam = self.model.predict(elo_scores)

        # Generate Poisson-distributed random scores based on the predicted lambda
        score = np.random.poisson(lam * lam_multiplier)

        # Return the simulated score
        return score
