from numpy import median
from pandas import DataFrame, concat

def get_feature_groups(train_set: DataFrame):
    """
    Take a DataFrame and return two lists:
        - Numerical Features
        - Categorical Features
    """

    # Numerical Features
    num_features = train_set.select_dtypes(
        include=['int64', 'float64']).columns

    # Categorical Features
    cat_features = train_set.select_dtypes(include=['object']).columns

    return list(num_features), list(cat_features)

def add_median_price(df: DataFrame):
    """
    Takes a DataFrame that must have the following columns:
        - Neighborhood
        - Year_Sold
        - Mo_Sold
        - Sale_Price

    Returns a DataFrame without the Neighborhood column, but with
    a new column called Neighborhood_Median_Sale_Price
    where each row contains the median sale price of all houses
    sold in that neighborhood before the current house was sold.
    """

    # Sort the dataframe first by neighborhood, year of sale, and then by month of sale
    df = df.sort_values(by=['Neighborhood', 'Year_Sold', 'Mo_Sold'], ascending=[True, True, True])

    # Create a new column in the dataframe for the median sale price of each neighborhood
    df['Neighborhood_Median_Sale_Price'] = 0

    # Create an empty list to store each neighborhood dataframe
    neighborhoods = []

    # Loop through each unique neighborhood in the dataframe
    for n in df['Neighborhood'].unique():
        # Copy the data for the current neighborhood
        nbh = df[df['Neighborhood'] == n].copy()
        # Loop through each house in the neighborhood
        for i in range(1, len(nbh)):
            # Get the index of the Neighborhood_Median_Sale_Price and Sale_Price columns
            msp_index = nbh.columns.get_loc('Neighborhood_Median_Sale_Price')
            sp_index  = nbh.columns.get_loc('Sale_Price')
            # Calculate the median sale price of all the houses sold before the current house
            nbh.iloc[i, msp_index] = median(nbh.iloc[:i, sp_index])
        # Append the data for the current neighborhood to the neighborhoods list
        neighborhoods.append(nbh)

    # Concatenate all the neighborhood dataframes into a single dataframe
    df = concat(neighborhoods)

    # Return the final dataframe
    return df