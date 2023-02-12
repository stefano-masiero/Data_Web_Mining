import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, median_absolute_error, max_error
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import cross_val_score


def split(X, y, test_size, use_sklearn=False):
    """
    This function is used to split the input data into training and testing sets, with the option to use the built-in train_test_split() 
    function from the scikit-learn library or to use a custom implementation.

    The function takes 4 parameters as input:
        X: a dataframe or matrix containing the independent variables
        y: a dataframe or matrix containing the dependent variable
        test_size: a float value between 0 and 1 representing the proportion of data to be used as the test set
        use_sklearn: a boolean value indicating whether to use the built-in train_test_split() function from scikit-learn or not.
    """

    if not use_sklearn:
        # Sorts X and respectively y by Year_Sold and Mo_Sold
        df = pd.concat([X, y], axis=1)
        df = df.sort_values(by=['Year_Sold', 'Mo_Sold'], ascending=True)
        _X = df.drop(y.name, axis=1)
        _y = df[y.name]

        # Splits the data into train and test sets where the test set is the last `test_size` of the data
        X_train = _X.iloc[:int(len(_X) * (1-test_size))]
        X_test = _X.iloc[int(len(_X) * (1-test_size)):]

        y_train = _y.iloc[:int(len(_y) * (1-test_size))]
        y_test = _y.iloc[int(len(_y) * (1-test_size)):]

        return X_train, X_test, y_train, y_test
    else:
        return train_test_split(X, y, test_size=test_size, shuffle=False)


class EvaluationResult:
    """
        The EvaluationResult class is a custom class that is used to store the evaluation metrics of a machine learning model. It is used to store five different evaluation metrics:
            r2: R-squared score
            explained_variance: explained variance score
            rmse: root mean squared error
            mae: mean absolute error
            max_error: maximum error
    """

    def __init__(self, r2: float, explained_variance: float, rmse: float, mae: float, max_error: float):
        self.r2 = r2
        self.explained_variance = explained_variance
        self.rmse = rmse
        self.mae = mae
        self.max_error = max_error

    def to_dict(self):
        """
             This method returns a dictionary that contains the values of the evaluation metrics.
        """
        return {
            'r2': self.r2,
            'explained_variance': self.explained_variance,
            'rmse': self.mse,
            'mae': self.mae,
            'max_error': self.max_error
        }

    def inline(self):
        """
            This method returns a string that contains the values of the evaluation metrics in a single line.
        """
        return f"r2: {self.r2:.5f}  -  " + \
               f"explained_variance: {self.explained_variance:.5f}  -  " + \
               f"rmse: {self.rmse:.5f}  -  " + \
               f"mae: {self.mae:.5f}  -  " + \
               f"max_error: {self.max_error:.5f}"

    def __str__(self):
        """
            This method returns a string that contains the values of the evaluation metrics in a multi-line format.
        """
        return f"r2:                 {self.r2: .5f}" + \
            f"\nexplained_variance: {self.explained_variance: .5f}" + \
            f"\nrmse:               {self.rmse: .5f}" + \
            f"\nmae:                {self.mae: .5f}" + \
            f"\nmax_error:          {self.max_error: .5f}"

    def __repr__(self):
        return self.__str__()


class Evaluator:
    """
      A class that evaluates a model on a dataset
    """

    def __init__(self, model: object, df: pd.DataFrame, ylabel: str):
        """
          Parameters
          ----------
          `model` : `object`
            The model to evaluate
          `df` : `pd.DataFrame`
            The dataset to evaluate the model on
          `ylabel` : `str`
            The column with the data to predict
        """
        self.model = model
        self.X = df.drop(ylabel, axis=1)
        self.y = df[ylabel]
        self.ylabel = ylabel

        self.yhat = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_dataset(self, test_size: float = 1/3, use_sklearn: bool = False):
        """
          Splits the dataset into a train and test set and fits the model

          Parameters
          ----------
          `test_size` : `float`
            The size of the test set relative to the whole dataset
            default: 0.2
        """
        self.X_train, self.X_test, self.y_train, self.y_test = split(
            self.X, self.y, test_size=test_size, use_sklearn=use_sklearn)

    def fit_model(self, epochs: int = 500):
        """This method fits the model to the training set. If the model is a Keras model, it will also set the number of epochs to run during the training."""

        if (self.X_train is None) or (self.X_test is None) or (self.y_train is None) or (self.y_test is None):
            self.split_dataset()

        # If the model is a keras model
        if type(self.model).__name__ == 'Sequential':
            self.model.fit(self.X_train, self.y_train,
                           epochs=epochs, verbose=0)
        else:
            self.model.fit(self.X_train, self.y_train)

    def get_predictions(self, n_pred=10, best=True):
        """
            This method returns the best or worst predictions of the model on the test set,
            based on the residuals. It returns a dataframe containing the top n_pred rows sorted by residuals,
            with the best predictions first if best=True and worst predictions first if best=False.

          Parameters
          ----------
          `no_pred` : `int`
            The number of outliers to return
            default: 10

          `best` : `bool`
            Retrieve best predictions if True, else worst predictions
            default: True
        """
        y_test = self.y_test
        y_pred = self.model.predict(self.X_test)

        if (type(self.model).__name__ == 'Sequential'):
            y_pred = y_pred.flatten()

        # Creates a new dataframe with all the data, but ordered by the residual
        df = pd.concat([self.X_test, y_test], axis=1)
        df['residual'] = y_test - y_pred

        # if worst is True then ascending should be False, since we want to order by absolute value of residual
        # and we want the largest possible value first
        df = df.sort_values(by='residual', key=abs, ascending=best)

        return df.head(n_pred)

    def evaluate(self, epochs: int = 500):
        """
            This method fits the model to the training set, uses the model to make predictions on the test set,
            and returns an EvaluationResult object which contains the evaluation metrics for the model,
            including R-squared, explained variance, root mean squared error, mean absolute error and maximum error.

          Returns
          -------
          `EvaluationResult` : `object`
            An object containing the r2, adjusted r2 and rmse of the model
        """

        self.fit_model(epochs=epochs)

        y_test = self.y_test
        y_pred = self.model.predict(self.X_test)
        X_test = self.X_test

        return EvaluationResult(
            r2=r2_score(y_test, y_pred),
            explained_variance=explained_variance_score(y_test, y_pred),
            rmse=mean_squared_error(y_test, y_pred, squared=False),
            mae=median_absolute_error(y_test, y_pred),
            max_error=max_error(y_test, y_pred)
        )

    def temporal_cv(self, n_splits=10, verbose=False, epochs: int = 500):
        """
        This method is used to perform temporal cross-validation on a dataset, using the TimeSeriesSplit class from the sklearn.model_selection module to split the data into train and validation sets.
            Takes these parameters:
                n_splits: an integer that specifies the number of splits to perform on the data.
                verbose: a boolean that controls whether or not to print out the evaluation results for each split.
                epochs: an integer that specifies the number of training epochs for the model, in case the model is a keras model.
        """

        X = self.X_train
        y = self.y_train

        # The method initializes several empty lists to store the evaluation metrics for each fold, and creates a TimeSeriesSplit object with the specified number of splits.
        r2_list = []
        ev_list = []
        rmse_list = []
        mae_list = []
        me_list = []

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # The evaluation metrics such as r2_score, explained_variance_score, mean_squared_error,
        # median_absolute_error and max_error are calculated and appended to the corresponding lists.
        # If verbose is set to True, the evaluation results for each fold are printed.
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            if (type(self.model).__name__ == 'Sequential'):
                self.model.fit(X_train, y_train, epochs=epochs, verbose=0)
            else:
                self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_val)

            r2_list.append(r2_score(y_val, y_pred))
            ev_list.append(explained_variance_score(y_val, y_pred))
            rmse_list.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae_list.append(median_absolute_error(y_val, y_pred))
            me_list.append(max_error(y_val, y_pred))

            if verbose:
                res = EvaluationResult(r2=r2_list[-1], explained_variance=ev_list[-1],
                                       rmse=rmse_list[-1], mae=mae_list[-1], max_error=me_list[-1])
                print(
                    f"[Split {fold}/{n_splits}] - train_size: {(X_train.shape[0] / X.shape[0]):.3f}")
                print(res.inline(), end='\n\n')
                # print(f"train_size: {(X_train.shape[0] / X.shape[0]):.3f} | r2: {r2_list[-1]:.5f} | exp_var: {ev_list[-1]:.5f} | rmse: {rmse_list[-1]:.5f}")

        # Finally, the function returns an EvaluationResult object containing the mean of the evaluation metrics for all the splits.
        return EvaluationResult(
            r2=np.mean(r2_list),
            explained_variance=np.mean(ev_list),
            rmse=np.mean(rmse_list),
            mae=np.mean(mae_list),
            max_error=np.mean(me_list)
        )

    def grid_search_cv(self, params: dict, train_size=2/3, scoring=None, verbose=0):
        """
        The method first creates an instance of the GridSearchCV class from sklearn, 
        passing in the model, params, cv, and scoring. 
        It then splits the data into training and test sets using the split method. 
        The GridSearchCV object is then fit to the training data using the fit method. 
        The method returns the best parameters and best score obtained during the grid search.

        Parameters:
            params: a dictionary containing the parameters to be searched over in the grid search
            train_size: the proportion of the data to be used for training (default 2/3)
            scoring: the scoring metric to use for evaluating the model during the grid search (default is None)
            verbose: whether or not to print out progress messages during the grid search (default is 0)
        """
        Gs = GridSearchCV(self.model, params, cv=5, scoring=scoring)
        X_train, X_test, y_train, y_test = split(
            self.X, self.y, test_size=1-train_size)
        Gs.fit(X_train, y_train)

        return Gs.best_params_, Gs.best_score_

    def plot_residuals(self, confidence_interval: float = 0.95, plot_normal_curve: bool = True, bins: int = 150):
        """
        The plot_residuals function is used to visualize the residuals (the difference between the predicted values and the actual values) of a model's predictions.
        The function takes in several optional parameters:

        Parameters:
            confidence_interval: a float between 0 and 1, used to indicate the level of confidence interval to plot on the residuals vs predicted plot. This will plot a shaded region around the 0 residual line, indicating the range of residuals that would be considered acceptable at the specified confidence level.
            plot_normal_curve: a boolean, which when set to True, will plot a normal distribution curve on the residuals distribution plot, to visually compare the distribution of residuals to a normal distribution.
            bins: an integer indicating the number of bins to use when plotting the histogram of residuals.
        """

        y_test = self.y_test
        y_pred = self.model.predict(self.X_test)

        if (type(self.model).__name__ == 'Sequential'):
            y_pred = y_pred.flatten()

        residuals = y_test - y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))

        ax1.scatter(y_pred, residuals, c=abs(residuals),
                    cmap='coolwarm', marker='x', alpha=0.75, s=100)

        if (confidence_interval > 0):
            ci = ((1 - confidence_interval) / 2) * residuals.std()
            ax1.axhline(y=ci, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(y=-ci, color='red', linestyle='--', alpha=0.5)
            ax1.fill_between(np.linspace(
                residuals.min(), residuals.max(), bins), ci, -ci, color='red', alpha=0.1)

        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')

        cm = plt.cm.get_cmap('coolwarm_r')
        n, _, patches = ax2.hist(residuals, bins=bins,
                                 density=True, color='grey', alpha=0.75)
        col = (n - n.min()) / (n.max() - n.min())

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))

        # Draws a curve representing the residuals distribution
        if plot_normal_curve:
            x = np.linspace(residuals.min(), residuals.max(), bins)
            mu, std = norm.fit(residuals)

            pdf = norm.pdf(x, mu, std)
            ax2.plot(x, pdf, linewidth=2,
                     label='Normal Distribution', color='gray')
            # Colors the area under the curve
            ax2.fill_between(x, pdf, color='gray', alpha=0.25)
            ax2.legend()

        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Density')
        ax2.set_title('Residuals Distribution')

        plt.show()

    def plot_grid_search(self, params, scores):
        """
            This method plots the grid search results using matplotlib.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(params, scores, marker='o')
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Score')
        ax.set_title('Grid Search')
        plt.show()
