import pandas as pd
import numpy as np

def interpolate_and_standardize(data):
    """
    Interpolate missing dates and standardize agronomic characteristics in time-series data.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing time-series agronomic data with columns:
        - "time": Timestamp for each record
        - "treatment": Experimental treatment condition (AVITI, TVITI, or DVITI)
        - Additional columns containing agronomic parameters to be standardize (assumed to be in columns 3-6)

    Returns
    -------
    - new_data : pandas.DataFrame
        DataFrame with interpolated missing dates and standardize agronomic parameters

    Notes
    -----
    - The function assumes a fixed date range from "2024-04-24" to "2024-08-31"
    - Standardization is applied only to columns 3-6 of the DataFrame
    - The function handles three treatment conditions: AVITI, TVITI, and DVITI
    - Missing treatment values are filled with the current treatment condition
    - Standardization parameters are calculated per treatment condition
    """

    ## Time Conversion
    # Convert time values to date objects for consistent processing.
    data["time"] = [ti.date() for ti in pd.to_datetime(data["time"])]

    ## Data Interpolation by Treatment
    # Create the complete date range.
    date_range = pd.date_range(start="2024-04-24", end="2024-08-31", freq="D")
    for cond in ["AVITI", "TVITI", "DVITI"]:
        # Filter data for the current treatment condition.
        data_temp = data[data["treatment"] == cond]
        # Set time as the index for time-based operations.
        data_temp.set_index("time", inplace=True)
        # Reindex to the complete date range to include all dates (highlighting the missing time values)
        data_temp = data_temp.reindex(date_range)
        # Interpolate missing values using time-based interpolation.
        data_temp = data_temp.infer_objects(copy=False).interpolate(method="time", limit_direction='both')
        # Fill the NaN values in the treatment column.
        data_temp["treatment"] = data_temp["treatment"].fillna(cond)
        # Combine data for all treatments.
        if cond == "AVITI":
            new_data = data_temp
        else:
            new_data = pd.concat([new_data, data_temp], axis=0)

    ## Standardization Parameters Calculation
    # Get all the unique treatments and their counts.
    treat = np.unique(new_data.loc[:, "treatment"], return_counts=True)

    # Calculate standardization parameters (mean and standard deviation) for each treatment
    # And create matrixes to standardize our data
    first = True
    for i in range(len(treat[0])):
        # Select agronomic characteristics for the current treatment.
        # The actual columns being selected are the last four (indices 3-6).
        data_filt = new_data.loc[
            new_data["treatment"] == treat[0][i],
            [False, False, False, True, True, True, True]
        ].to_numpy()

        # Calculate mean and standard deviation.
        mean_norm = np.mean(data_filt, axis=0)
        std_norm = np.std(data_filt, axis=0)

        # Create matrices of repeated mean and std values.
        mean_mat_inter = np.array(mean_norm.tolist() * treat[1][i]).reshape((130, 4))
        std_mat_inter = np.array(std_norm.tolist() * treat[1][i]).reshape((130, 4))
        # Combine matrices for all treatments.
        if first:
            mean_mat = mean_mat_inter
            std_mat = std_mat_inter
            first = False
        else:
            mean_mat = np.vstack((mean_mat, mean_mat_inter))
            std_mat = np.vstack((std_mat, std_mat_inter))

    ## Data Standardization
    # Apply standardization to agronomic characteristics.
    # Standardization formula: (value - mean) / std
    new_data.iloc[:, 3:7] = (new_data.iloc[:, 3:7] - mean_mat) / std_mat

    return new_data

def dist_manathan(data_1, data_2):
    """
    Calculate the Manhattan distance between two matrices or arrays.

    The Manhattan distance (also known as L1 distance or taxicab distance) is the sum of the
    absolute differences between corresponding elements of two matrices or arrays.
    This function computes the total Manhattan distance between all elements of the inputs.

    Parameters
    ----------
    data_1 : numpy.ndarray
        First input matrix or array.
    data_2 : numpy.ndarray
        Second input matrix or array.

    Returns
    -------
    float
        The Manhattan distance between `data_1` and `data_2`, rounded to 2 decimal places.

    Notes
    -----
    - The function assumes `data_1` and `data_2` are of the same shape.
    - The result is rounded to 2 decimal places for readability.
    """

    ## Manhattan Distance Calculation
    # Compute the element-wise absolute difference between the two matrices.
    diff = (data_2 - data_1).abs()

    # Sum the absolute differences across all dimensions.
    # The first .sum() sums across columns, the second .sum() sums across rows.
    dist = diff.sum().sum()

    # Round the result to 2 decimal places for readability.
    return round(dist, 2)

from itertools import combinations
from itertools import compress
from copy import deepcopy

def select_variable(data_all, data_train, dist_func, interact=False):
    """
    Select the optimal combination of variables (and their interactions, if specified) by finding a compromise between having variable which :
    - Maximise the distinction between treatments using a distance-based approach.
    - Minimise their eloignement between predicted values (using predicted mask) and ground truth values (using ground truth mask)
    
    To do this, it calculates distances between treatment groups and between training and full datasets, 
    normalizes these distances, and selects the combination with the highest overall score.

    Parameters
    ----------
    data_all : pandas.DataFrame
        Complete dataset containing all images and their predicted characteristics with columns:
        - "treatment": Treatment condition (e.g., "AVITI", "DVITI", "TVITI")
        - Additional columns containing variables (aka characteristics) to evaluate (starting from column 3)
    data_train : pandas.DataFrame
        Dataset containing the images used in training for the segmentation 
        and their caracteristics derived from ground truth mask with the same structure as `data_all`.
    dist_func : function
        Distance function to calculate dissimilarity and assimilarity between datasets.
        Should accept two DataFrames of same shape and return a scalar distance.
    interact : bool, optional
        If True, considers interaction terms (products of variables) in the evaluation.
        Default is False.

    Returns
    -------
    list
        List of variable names (and their interactions, if applicable) that form the optimal combination
        for our criterias.

    Notes
    -----
    - The function evaluates all combinations of variables from column 3 onward.
    - The function removes all the combinations of variables and interaction where: 
        * There are only interaction(s).
        * If there are interaction(s) that doesn't reference the non-interacting variable(s).
    - The function calculates 6 distance metrics for each combination:
        * A_D: Distance between AVITI and DVITI treatments
        * A_T: Distance between AVITI and TVITI treatments
        * D_T: Distance between DVITI and TVITI treatments
        * A_A: Distance between AVITI predicted and ground truth datasets
        * D_D: Distance between DVITI predicted and ground truth datasets
        * T_T: Distance between TVITI predicted and ground truth datasets
    - Distances are normalized using min-max normalization (0-1 range).
    - For between-treatment distances, higher values are better (normalized directly).
    - For between-datasets distances, lower values are better (normalized as 1 - normalized_value).
    - The optimal combination is selected based on the sum of normalized scores.
    """

    ## Interaction Terms Creation (if enabled)
    # Initialisation
    col_name = data_all.iloc[:, 3:].columns
    len_col = len(data_all.iloc[:, 3:].columns)
    
    if interact:
        # Generate all possible combinations of variables.
        all_combinaison_name = [
            i for k in range(2, len_col)
            for i in combinations(col_name, k)
        ]
        # Link each combinations with a string keys.
        # Example : "A:B : ["A", "B"]"
        all_combinaison_name = {":".join(j): j for j in all_combinaison_name}

        # Adding the interaction terms to the dataset by multiplying the variables that interact.
        for col in all_combinaison_name:
            new_col_val_all = []
            new_col_val_train = []
            # Multiply variables to create interaction terms.
            for comp in all_combinaison_name[col]:
                if len(new_col_val_all) == 0:
                    new_col_val_all = data_all[comp].values
                    new_col_val_train = data_train[comp].values
                else:
                    new_col_val_all = new_col_val_all * data_all[comp].values
                    new_col_val_train = new_col_val_train * data_train[comp].values

            # Add interaction terms to the datasets.
            data_all[col] = new_col_val_all
            data_train[col] = new_col_val_train

    ## Variable Combinations Generation
    # Generate all possible combinations of variables (from 0 to all variables).
    all_combinaison_name_dist = [
        list(i) for k in range(len(data_all.iloc[:, 3:].columns) + 1)
        for i in combinations(data_all.iloc[:, 3:].columns, k)
    ]
    # Link each combinations with a string keys.
    # Example : "A + B : ["A", "B"]"
    all_combinaison_name_dist = {" + ".join(j): j for j in all_combinaison_name_dist}

    ## Filtering Interaction (if enabled)
    if interact:
        # Create a copy of the combinations for validation.
        all_combinaison_name_dist_inter = deepcopy(all_combinaison_name_dist)

        # Remove combinations with invalid interactions.
        for combin_name in all_combinaison_name_dist_inter:
            # Identify interaction terms (containing ':') and non-interaction terms.
            mask_interact = [':' in i for i in all_combinaison_name_dist_inter[combin_name]]
            mask_no_interact = [not i for i in mask_interact]

            # If no non-interaction terms, remove the combination.
            if sum(mask_no_interact) == 0:
                all_combinaison_name_dist.pop(combin_name)
                continue

            # Get lists of non-interaction and interaction variables.
            variable_no_interact = list(compress(
                all_combinaison_name_dist_inter[combin_name],
                mask_no_interact
            ))
            variable_interact = list(compress(
                all_combinaison_name_dist_inter[combin_name],
                mask_interact
            ))

            # Generate all possible theoretical interactions from non-interaction variables.
            variable_theoric_interact = [
                ":".join(list(i))
                for k in range(2, len(variable_no_interact) + 1)
                for i in combinations(variable_no_interact, k)
            ]

            # Check if all interaction terms are valid (exist in theoretical interactions).
            var_in_inter = [var_i in variable_theoric_interact for var_i in variable_interact]

            # If no interaction terms, assume valid.
            if len(var_in_inter) == 0:
                var_in_inter = [True]

            # If any interaction term is invalid, remove the combination.
            if (sum(var_in_inter) != len(var_in_inter)):
                all_combinaison_name_dist.pop(combin_name)

    ## Distance Calculation
    # Initialize dictionary to store distance metrics for each combination.
    all_model_values = {
        n: {
            "A_D": 0, "A_T": 0, "D_T": 0,
            "A_A": 0, "D_D": 0, "T_T": 0
        }
        for n in all_combinaison_name_dist.keys()
    }

    # Calculate distances for each variable combination.
    for col in all_combinaison_name_dist:
        # Filter data for each treatment condition.
        data_all_AVITI = data_all.loc[data_all["treatment"] == "AVITI", all_combinaison_name_dist[col]]
        data_all_DVITI = data_all.loc[data_all["treatment"] == "DVITI", all_combinaison_name_dist[col]]
        data_all_TVITI = data_all.loc[data_all["treatment"] == "TVITI", all_combinaison_name_dist[col]]
        data_train_AVITI = data_train.loc[data_train["treatment"] == "AVITI", all_combinaison_name_dist[col]]
        data_train_DVITI = data_train.loc[data_train["treatment"] == "DVITI", all_combinaison_name_dist[col]]
        data_train_TVITI = data_train.loc[data_train["treatment"] == "TVITI", all_combinaison_name_dist[col]]

        # Calculate distances between treatment groups.
        all_model_values[col]["A_D"] = dist_func(data_all_AVITI, data_all_DVITI)
        all_model_values[col]["A_T"] = dist_func(data_all_AVITI, data_all_TVITI)
        all_model_values[col]["D_T"] = dist_func(data_all_DVITI, data_all_TVITI)

        # Calculate distances between predicted and ground truth datasets.
        all_model_values[col]["A_A"] = dist_func(data_all_AVITI, data_train_AVITI)
        all_model_values[col]["D_D"] = dist_func(data_all_DVITI, data_train_DVITI)
        all_model_values[col]["T_T"] = dist_func(data_all_TVITI, data_train_TVITI)

    ## Data Normalization and Selection
    all_model_values = pd.DataFrame(all_model_values)
    # Removes the line with no variables
    all_model_values = all_model_values.T.iloc[1:, :]

    # Normalize all distance metrics using min-max normalization.
    for score in all_model_values.columns:
        score_split = str.split(score, "_")

        # For between-treatment distances, higher values are better (normalized directly).
        if score_split[0] != score_split[1]:
            all_model_values[score] = (
                all_model_values[score] - min(all_model_values[score])
            ) / (
                max(all_model_values[score]) - min(all_model_values[score])
            )
        # For between-datasets distances, lower values are better (normalized as 1 - normalized_value).
        else:
            all_model_values[score] = 1 - (
                (all_model_values[score] - min(all_model_values[score]))
                / (max(all_model_values[score]) - min(all_model_values[score]))
            )

    # Sum the normalized scores for each combination.
    all_model_values = all_model_values.sum(axis=1)

    # Sort combinations by total score in ascending order.
    all_model_values.sort_values(axis=0, ascending=True, inplace=True)

    # Return the best combination (highest score).
    return str.split(
        all_model_values[all_model_values == max(all_model_values)].index.values.tolist()[0],
        " + "
    )

if __name__ == "__main__":
    import argparse
    import importlib
    
    ## Ask the relevant arguments
    parser = argparse.ArgumentParser(description='Select the variables used for prediction')
    # Input the path to the csv file with the agronomic characteritics of all the images.
    parser.add_argument('--agro_chara_all', type=str, required=False, default="Core/Results/Agro_chara_vine.csv",
                        help='Path to the csv file with the agronomic characteritics of all the images where the variable are selected')
    # Input the path to the csv file with the agronomic characteritics of the ground truth images used for training.
    parser.add_argument('--agro_chara_train', type=str, required=False, default="Core/Results/Agro_chara_vine_train.csv",
                        help='Path to the csv file with the agronomic characteritics of the trained images used for selecting variables')
    # Distance used to compare each time series.
    parser.add_argument('--dist_func', type=str, required=False, default="dist_manathan",
                        help='Name of the function which calculate the distance used to compare each time series')
    # Choose the save or not the results for the prediction.
    parser.add_argument('--save', type=bool, required=False, default=False,
                        help='If True, saves the results of the function in the "parameters.txt" file for the prediction')
    # Store the parsed arguments
    args = parser.parse_args()
    
    ## Preparing the database and the distance used
    # Extracting the distance used
    dist_func = {"dist_manathan": dist_manathan}
    dist_func_used = dist_func[args.dist_func]
    # Loading the databases used
    data_para_all = pd.read_csv(args.agro_chara_all)
    data_para_train = pd.read_csv(args.agro_chara_train)
    # Interpolate the values of variables at missing time and normalize them
    data_para_all = interpolate_and_standardize(data=data_para_all)
    data_para_train = interpolate_and_standardize(data=data_para_train)
    
    ## Output the variables that results in : 
    # The different treatment are well separated
    # The ground truth values generated with ground-truth mask are close to predicted ones.
    res = select_variable(data_all=data_para_all, data_train=data_para_train, dist_func=dist_func_used)
    print("Variable(s) selected : ", str(res))