## General function
import os
import PIL.Image
import base64
import requests
from io import BytesIO
from typing import Optional, Union

def load_image(image: Union[str, "PIL.Image.Image"], timeout: Optional[float] = None, mode = ["RGB"]) -> "PIL.Image.Image":
    # Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/image_utils.py
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.
        mode ('list'): (added)
            Image representation space ("RGB", "HSV", "LAB"...)

    Returns:
        `np.array`: A PIL Image converted in a np.array.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            image = PIL.Image.open(BytesIO(requests.get(image, timeout=timeout).content))
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            if image.startswith("data:image/"):
                image = image.split(",")[1]

            # Try to load as base64
            try:
                b64 = base64.decodebytes(image.encode())
                image = PIL.Image.open(BytesIO(b64))
            except Exception as e:
                raise ValueError(
                    f"Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got {image}. Failed with {e}"
                )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise TypeError(
            "Incorrect format used for image. Should be an url linking to an image, a base64 string, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    if len(mode) == 1:
        image_fin = image.convert(mode[0])
        image_fin = np.array(image_fin)
    else :
        for m in range(len(mode)) :
            if m==0:
                image_fin = image.convert(mode[m])
                image_fin = np.array(image_fin)
            else:
                image_temp = image.convert(mode[m])
                image_temp = np.array(image_temp)
                image_fin = np.append(image_fin, image_temp, axis=2)
    return image_fin

import random as rand
from datetime import timedelta
import pandas as pd
import numpy as np

## Function and metrics used for the learning loop
# Function to shape data while learning

def time_seperate_train_test(data, freq_sep_time=15, train_size=0.6):
    """
    Split time-series data into training and testing sets while maintaining temporal structure.

    This function separates data into training and testing sets where the data is separated in time periods of 15 days (by default)
    and assigned to test and train depending on determined train size (with 60% as a default).
    This repartition also ensure that, in both sets, we have the same repartition of treatments.
    It also interpolates missing data points (due to image of bad quality removed by hand).

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing time-series data with a 'time' column and 'treatment' column.
    freq_sep_time : int, optional
        Length of time (in days) used for splitting the whole data in time periods. Default is 15.
    train_size : float, optional
        Proportion of time periods to allocate to the training set. Default is 0.6.

    Returns
    -------
    tuple
        A tuple containing:
        - data_set_train (pd.DataFrame): Training set.
        - data_set_test (pd.DataFrame): Testing set.
        - (index_tr, index_ts) (tuple): Tuple of dictionaries containing time indices for train and test sets.
    """

    ## Initialization
    # Extract treatment labels from the data.
    labels = data.loc[:, "treatment"]
    unique_labels = np.unique(labels)
    # Initialize variables for tracking indices and time ranges.
    first = True
    index_ts = {"AVITI": [], "DVITI": [], "TVITI": []}
    index_tr = {"AVITI": [], "DVITI": [], "TVITI": []}
    min_indx = 0
    max_indx = 0

    ## Processing each treatment
    for i in unique_labels:
        # Filter data for the current treatment.
        data_set_filt = data.loc[labels == i, :]
        # Add an index column to preserve original indices.
        data_set_filt.loc[:, "index_second"] = data_set_filt.index
        # Set the time column as the index for time-based operations.
        data_set_filt.set_index("time", inplace=True, drop=False)

        ## Interpolating missing values (for the missing times)
        # Create a complete date range which is used to reveal the missing dates in the data.
        time_corr = pd.date_range(start=min(data_set_filt.index),
                                  end=max(data_set_filt.index),
                                  normalize=True,
                                  freq="D")
        data_set_filt = data_set_filt.reindex(time_corr)
        # Interpolate missing values using time-based interpolation.
        data_set_filt = data_set_filt.infer_objects(copy=False).interpolate(method="time",
                                                                            limit_direction='both')
        # Fill remaining NaN values in treatment and image columns.
        data_set_filt.loc[:, "treatment"] = data_set_filt.loc[:, "treatment"].fillna(i)
        data_set_filt.loc[:, "image"] = data_set_filt.loc[:, "image"].fillna("No Image")

        ## For the time interpolation to work, the time needed to be set as an index
        ## The next block retreives the date and change back the index into appropriate integers 
        # Extract date from the index and convert index_second to integer.
        data_set_filt.loc[:, "time"] = data_set_filt.index.date
        data_set_filt.index_second = data_set_filt.index_second.astype(int)
        # Update the global index range.
        max_indx += len(data_set_filt.loc[:, "index_second"])
        data_set_filt.loc[:, "index_second"] = [j for j in range(min_indx, max_indx)]
        min_indx = max_indx
        # Set index_second as the index for further processing.
        data_set_filt.set_index("index_second", inplace=True)

        ## First cutting the data into multiple time periods
        # Define the step size for time periods.
        step_time = str(freq_sep_time) + 'D'
        # Create time ranges with the specified frequency.
        time_rang = pd.date_range(start=min(data_set_filt["time"]),
                                  end=max(data_set_filt["time"]),
                                  normalize=True,
                                  freq=step_time)
        # Create time period tuples for splitting.
        test_train_index = [(time_rang[i], time_rang[i+1] - timedelta(days=1))
                            for i in range(len(time_rang)-2)]
        # Shuffle the time periods for random splitting.
        rand.shuffle(test_train_index)

        ## Splitting time periods into train and test sets
        # Allocate time periods to training set based on train_size.
        train_index_temp = [test_train_index[i]
                            for i in range(len(test_train_index))
                            if (i/len(test_train_index)) <= train_size]
        # Allocate remaining time periods to testing set.
        test_index_temp = [test_train_index[i]
                           for i in range(len(test_train_index))
                           if (i/len(test_train_index)) > train_size]
        # Convert time ranges to daily date ranges for training set.
        train_index_nodate = [pd.date_range(start=j[0], end=j[1], freq="D")
                              for j in train_index_temp]
        # Convert time ranges to daily date ranges for testing set.
        test_index_nodate = [pd.date_range(start=j[0], end=j[1], freq="D")
                             for j in test_index_temp]
        # Store training and testing time ranges.
        # This is the one that will be used by the Dataloader to generate the input.
        index_tr[i].append(train_index_nodate)
        index_ts[i].append(test_index_nodate)

        ## Adding target time periods (shifted by freq_sep_time) 
        ## Target time periods are those which holds the characteristics we would like to predict.
        ## So for training and testing, it is essential to make sure they are in both sets.
        # Create shifted time ranges for targets in training set.
        train_index_add = [(train_index_temp[i][0] + timedelta(days=freq_sep_time),
                            train_index_temp[i][1] + timedelta(days=freq_sep_time))
                           for i in range(len(train_index_temp))]
        # Create shifted time ranges for targets in testing set.
        test_index_add = [(test_index_temp[i][0] + timedelta(days=freq_sep_time),
                           test_index_temp[i][1] + timedelta(days=freq_sep_time))
                          for i in range(len(test_index_temp))]
        # Combine original and shifted time ranges.
        train_index_temp.extend(train_index_add)
        test_index_temp.extend(test_index_add)
        # Convert combined time ranges to daily date ranges.
        train_index_nodate = [pd.date_range(start=j[0], end=j[1], freq="D")
                              for j in train_index_temp]
        test_index_nodate = [pd.date_range(start=j[0], end=j[1], freq="D")
                             for j in test_index_temp]
        # Flatten the date ranges to lists of dates.
        train_index = [j[m].date() for j in train_index_nodate for m in range(len(j))]
        test_index = [j[m].date() for j in test_index_nodate for m in range(len(j))]
        # Create boolean masks for train and test dates.
        train_index_bool = [item in train_index for item in data_set_filt["time"]]
        test_index_bool = [item in test_index for item in data_set_filt["time"]]

        ## Building the final datasets
        if first:
            # Initialize datasets for the first treatment.
            data_set_train = data_set_filt.loc[train_index_bool, :]
            data_set_test = data_set_filt.loc[test_index_bool, :]
            label = data_set_train.loc[:, "treatment"]
            first = False
        else:
            # Concatenate datasets for subsequent treatments.
            data_set_train = pd.concat([data_set_train, data_set_filt.loc[train_index_bool, :]])
            data_set_test = pd.concat([data_set_test, data_set_filt.loc[test_index_bool, :]])
            label = pd.concat([label, data_set_train.loc[:, "treatment"]])

    return data_set_train, data_set_test, (index_tr, index_ts)

# Inspired by https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

trans = transforms.Compose([transforms.ToTensor()])
class TimeAgrocamDataset(Dataset):
    """
    Representation of the Agrocam dataset for use with PyTorch DataLoader 
    to train an algorithm to use Agrocam Images (vineyard images) time-series to predict the vineyard futur characteristics.

    This dataset class handles time-series data for vineyard images and their associated agronomic characteristics.
    It loads images and their characteristics (normalized), as well as their futur characteristics used for prediction.
    
    Note : When providing index for this dataset, all must refer to images of the same treatment.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing time-series data with columns for treatments, agronomic characteristics,
        and image paths.
    characteristics : list, optional
        List of characteristics (by naming their columns name) to use for prediction. 
        Default is ['treatment', 'H_vigne', 'P_vigne', 'Hue_vigne', 'Hue_rang'].
    len_predict : int, optional
        Number of the input images (1 per time) and length of the prediction horizon (number of days to predict ahead). 
        Default is 15.
    mode : list, optional
        Color space mode for image loading. 
        Default is ["HSV"].

    Parameters
    ----------
    mode : list
        Color space mode for image loading.
    characteristics : list
        List of characteristics used for learning.
    len_predict : int
        Number of the input images and prediction horizon length.
    label : pd.Series
        Treatment labels associated with each images.
    time : pd.Series
        Time values associated with each images.
    att_notnorm : pd.DataFrame
        Original input and target characteristics (without normalization).
    data : pd.DataFrame
        Normalized input and target characteristics used for training.
    transform : callable
        Image transformation pipeline.
    """

    def __init__(self, data,
                 characteristics=['treatment', 'H_vigne', 'P_vigne', 'Hue_vigne', 'Hue_rang'],
                 len_predict=15, mode=["HSV"]):
        ## Initializing learning parameters
        # Store the image loading mode.
        self.mode = mode
        # Store the characteristics list for prediction.
        self.characteristics = characteristics
        # Store the prediction horizon length.
        self.len_predict = len_predict

        ## Extracting and preparing data
        # Extract treatment labels.
        self.label = data.loc[:, "treatment"]
        # Extract time values.
        self.time = data.loc[:, "time"]
        # Store the original non-normalized data.
        self.att_notnorm = data
        # Select relevant columns and normalize characteristics (excluding treatment and image).
        data = data.loc[:, characteristics + ["image"]]
        data[characteristics[1:len(characteristics)]] = (
            data[characteristics[1:len(characteristics)]] -
            np.mean(data[characteristics[1:len(characteristics)]], axis=0)
        ) / np.std(data[characteristics[1:len(characteristics)]], axis=0)
        # Store the normalized data.
        self.data = data
        # Set the image transformation pipeline.
        self.transform = trans

    def __len__(self):
        """
        Return the total number of samples (here images) in the dataset.

        Returns
        -------
        int
            Number of samples (here images) in the dataset.
        """
        return len(self.label)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at specified indexes.

        Parameters
        ----------
        idx : int or torch.Tensor
            Index or tensor of indices for the sample(s) to retrieve.

        Returns
        -------
        dict
            Dictionary containing (for each given index):
            - 'image': numpy array of loaded image(s)
            - 'target': numpy array of target characteristic(s)
            - 'cond': treatment label 
        """
        ## Convert (if needed) the indexed input
        # Convert tensor index to list if necessary.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Convert single integer index to a list.
        if isinstance(idx, int):
            idx = [idx]

        ## Initialization
        image_list = []
        target_list = []
        for x in idx:
            ## Loading, transforming and adding images located at the specified data index.
            # If no image is available, create a black image (that will be ignored during training).
            if pd.isna(self.data.loc[x, "image"]) or self.data.loc[x, "image"] == "No Image":
                image = np.full((1080, 1920, 3), 0.0)
            # Otherwise, load the image in the specified mode.
            else:
                image = load_image(self.data.loc[x, "image"], mode=self.mode)
            # Apply transformations if specified.
            if self.transform:
                image = self.transform(image)
            # Add the processed image to the list.
            image_list.append(image)

            ## Loading, transforming and adding target characteristics associated with the specified index
            # Get the target characteristics for the prediction horizon.
            # Given a specified index of 1 and a lenght of prediction of 15, it will take the characteristics of the 16th index.
            target = self.data.loc[(x + self.len_predict), self.characteristics].tolist()
            # Convert treatment labels to numerical values.
            class_cond = {"AVITI": 0, "DVITI": 1, "TVITI": 2}
            if self.transform:
                for c_c in class_cond:
                    if target[0] == c_c:
                        target[0] = class_cond[c_c]
                # Convert target to a PyTorch tensor.
                target = torch.tensor(target, dtype=torch.float32)
            # Add the target characteristics (excluding treatment) to the list.
            target_list.append(target[1:])
        # Specify the treatment linked to the index inputed (all index must refer to image of the same treatment).
        treatment = target[0]

        ## Create and return the sample dictionary.
        sample = {
            'image': np.array(image_list),
            'target': np.array(target_list),
            'cond': treatment
        }
        return sample

    def get_label_time(self):
        """
        Get the labels and time values of the dataset.

        Returns
        -------
        tuple
            A tuple containing:
            - label: Treatment labels associated with each images.
            - time: Time values associated with each images.
        """
        return (self.label, self.time)

from torch.utils.data.sampler import Sampler

class TimePerClassSampler(Sampler):
    """
    A custom PyTorch Sampler for time-series data which, given that balances samples index generated by `TimeAgrocamDataset` per class.

    This sampler handles time-series data by organizing them into multiple batches.
    Each batch is composed of 3 (one per treatment) temporal sequences (taken as random) used for each training iteration.
    
    This sampler also balances the number of sequences across classes by oversampling or undersampling classes.
    It also removes the temporal sequences at the end of the temporal series (because we wouldn't have the data to verify the prediction)
    
    Parameters
    ----------
    labels : np.array
        Treatment labels associated with each images (using indexes).
    times : pd.DataFrame
        Time values associated with each images (using indexes).
    time_index : dict
        Dictionary associating indexes to each class, organized by temporal sequences.
    len_predict : int, optional
        Length of the prediction horizon. Default is 15.

    Attributes
    ----------
    index_time_cond : dict
        Dictionary mapping each class to its temporal sequences.
    classes : numpy.ndarray
        Unique class labels.
    time_class_to_indices : dict
        Dictionary mapping each class to its time-series indices.
    samples_per_class : dict
        Number of temporal sequences per class.
    num_batches : int
        Number of batches that can be generated.
    """

    def __init__(self, labels, times, time_index, len_predict=15):
        ## Initializing the sampler
        # Get unique class labels.
        self.classes = np.unique(np.array(labels))
        # Initialize dictionary to store time indices for each condition.
        index_time_cond = {cls: [] for cls in self.classes}
        # Convert date ranges to lists of dates for each condition.
        for c in time_index:
            for j in time_index[c][0]:
                j_date = [j[m].date() for m in range(len(j))]
                index_time_cond[c].append(j_date)
        time_index = index_time_cond

        ## Mapping each time (in each class) to its corresponding index
        # Initialize dictionary to map each class to its time indices.
        self.time_class_to_indices = {cls: [] for cls in self.classes}
        # Populate the time indices for each class.
        for cond in self.time_class_to_indices:
            # Get the time period (times_filt) and its indices (ind_times_filt) for the current condition.
            ind_times_filt = time_index[cond]
            times_filt = times[labels == cond]
            # Map each time point in time period to their corresponding indexes
            # Those indexes are stocked in time_class_to_indices (in the corresponding class and time period)
            for T in range(len(ind_times_filt)):
                self.time_class_to_indices[cond].append([])
                for t in range(len(ind_times_filt[T])):
                    # Find the index for each time point.
                    add_t = times_filt[times_filt == ind_times_filt[T][t]]
                    if len(add_t) == 0:
                        # If no index found, append NaN.
                        self.time_class_to_indices[cond][T].append(np.nan)
                    else:
                        # Otherwise, append the found index.
                        self.time_class_to_indices[cond][T].append(add_t.index[0])

        ## Removing time series with no target time series 
        ## (aka where the quality of the prediction cannot be assessed).
        for cond in self.time_class_to_indices:
            # Find the maximum index for the current condition.
            max_cond = max(max(self.time_class_to_indices[cond]))
            # List of time point that would have no targets.
            list_del = []
            for T in range(len(self.time_class_to_indices[cond])):
                # For each time period, generate the maximal target index and checking if it falls outside of the database.
                # If yes, it will be deleted.
                if (max(self.time_class_to_indices[cond][T]) + len_predict) > max_cond:
                    list_del.append(T)
            # Remove incomplete sequences.
            for d in list_del:
                self.time_class_to_indices[cond].pop(d)

        ## Determining if a balancing method is needed (and which one) or not
        # Calculate the number of samples (sequences) per class.
        self.samples_per_class = {cls: len(self.time_class_to_indices[cls]) for cls in self.time_class_to_indices}
        # Identify the classes with the maximum and minimum number of samples.
        max_samples = max(self.samples_per_class.values())
        min_samples = min(self.samples_per_class.values())
        # Identify classes with max and min samples.
        num_max = [i for i in range(len(self.samples_per_class.values()))
                  if list(self.samples_per_class.values())[i] == max_samples]
        num_min = [i for i in range(len(self.samples_per_class.values()))
                  if list(self.samples_per_class.values())[i] == min_samples]
        # If there are less classes with a minimum number of samples, oversample those classes.
        if len(num_min) < len(num_max):
            balance_method = 'oversample'
        # If there are more classes with a minimum number of samples, undersample the other classes.
        elif len(num_min) > len(num_max):
            balance_method = 'undersample'
        # If all classes have the same number of samples, no balancing is needed.
        elif len(num_min) == len(self.samples_per_class.values()) and len(num_max) == len(self.samples_per_class.values()):
            print("ATTENTION: Well balanced labels")
            balance_method = 'nope'
        # If each class have different number of samples, a random balancing method is chosen.
        elif len(self.samples_per_class.values()) == len(np.unique(self.samples_per_class.values())[0]):
            balance_method = np.random.choice(['oversample', 'undersample'])
        # If the dataset is unbalanced but doesn't fit the above criteria, no balancing is applied.
        else:
            print("ATTENTION: Unbalanced labels")
            balance_method = 'nope'

        ## Balance the samples per class using the chosen method.    
        if balance_method == 'oversample':
            # Oversample under-represented (having the least amount of samples) classes to match the maximum number of samples.
            max_samples = max(self.samples_per_class.values())
            for cls in self.classes:
                while len(self.time_class_to_indices[cls]) < max_samples:
                    # Randomly duplicate a time sequence from the under-represented class.
                    random_index = np.random.randint(0, len(self.time_class_to_indices[cls]))
                    self.time_class_to_indices[cls] = np.vstack((
                        self.time_class_to_indices[cls],
                        self.time_class_to_indices[cls][random_index]
                    ))
        if balance_method == 'undersample':
            # Undersample over-represented (having the most amount of samples) classes to match the minimum number of samples.
            min_samples = min(self.samples_per_class.values())
            for cls in self.classes:
                while len(self.time_class_to_indices[cls]) > min_samples:
                    # Randomly remove a time sequence from the over-represented class.
                    random_index = np.random.randint(0, len(self.time_class_to_indices[cls]))
                    self.time_class_to_indices[cls] = np.delete(
                        self.time_class_to_indices[cls],
                        random_index,
                        0
                    )

        # Recalculate the number of samples per class after balancing.
        self.samples_per_class = {cls: len(self.time_class_to_indices[cls]) for cls in self.time_class_to_indices}
        # Calculate the total number of batches.
        self.num_batches = min(self.samples_per_class.values())

    def __iter__(self):
        """
        Generate batches of time-series indices.

        Yields
        ------
        list
            List of indices for a batch of time-series sequences.
        """
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                # Get a time sequence for each class.
                indices = self.time_class_to_indices[cls][:1]
                batch.extend(indices)
            yield batch

    def __len__(self):
        """
        Return the number of batches.

        Returns
        -------
        int
            Number of batches that can be generated.
        """
        return self.num_batches

from torch.utils.tensorboard import SummaryWriter

# Custom Metrics and Loss

def custom_MSE(yhat, y):
    """
    Compute a custom Mean Squared Error (MSE).

    This function calculates the mean squared error between predicted and target values,
    normalized by twice the batch size. This is equivalent to the standard MSE divided by 2.

    Parameters
    ----------
    yhat : torch.Tensor
        Predicted values from the model.
        Shape: (batch_size, ...) where ... can be any additional dimensions.
    y : torch.Tensor
        Ground truth target values.
        Shape: Must match yhat.

    Returns
    -------
    MSE : torch.Tensor
        Scalar tensor containing the computed MSE score for each characteristics.
        The MSE is calculated as: sum((yhat - y)^2) / (2 * batch_size)
    """
    # Calculate squared differences between predictions and targets
    squared_diff = (yhat - y) ** 2
    # All squared differences are divided by twice the batch size
    MSE = squared_diff / (2 * yhat.size(0))
    return MSE

def custom_acc(yhat, y):
    """
    Compute custom accuracy for classification tasks.

    This function calculates the accuracy as the ratio of correct predictions
    (True Positives + True Negatives) to the total number of predictions.

    Parameters
    ----------
    yhat : torch.Tensor
        Predicted class labels from the model.
        Shape: (batch_size, ...) where ... can be any additional dimensions.
    y : torch.Tensor
        Ground truth class labels.
        Shape: Must match yhat.

    Returns
    -------
    accuracy : torch.Tensor
        Scalar tensor containing the computed accuracy.
        The accuracy is calculated as: (TP + TN) / (TP + FP + TN + FN)
        where:
        - TP: True Positives (correct predictions)
        - TN: True Negatives (correct rejections)
        - FP: False Positives (incorrect predictions)
        - FN: False Negatives (missed predictions)

    Notes
    -----
    This implementation assumes that:
    1. yhat and y are class labels (not probabilities)
    2. The tensors contain integer class indices
    3. The function counts exact matches between yhat and y
    """
    # Count the number of correct predictions (True Positives + True Negatives)
    TP_TN = torch.sum(yhat == y)

    # Count the total number of predictions (TP + FP + TN + FN)
    TP_FP_TN_FN = torch.sum(yhat == y) + torch.sum(yhat != y)

    # Calculate accuracy as the ratio of correct predictions to total predictions
    # This handles both correct and incorrect predictions across all classes
    accuracy = TP_TN / TP_FP_TN_FN

    return accuracy

import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

def time_train(data_total,model, epochs, save=True):
    """
    Train a time-series model for agronomic characteristics prediction and time series classification.

    This function handles the complete training loop for a CNN-LSTM model, including:
    - Data loading with temporal separation
    - Forward/backward passes
    - General loss calculation with weighted components (0.1% prediction, 99.9% classification) (not used)
    - Metrics logging using TensorBoard
    - Model checkpoint saving

    Parameters
    ----------
    model : torch.nn.Module
        The CNN-LSTM model to train.
    epochs : int
        Number of training epochs.
    save : bool, optional
        If True, saves model checkpoints every 5 epochs. Default is True.

    Returns
    -------
    None
        The function trains the model and logs metrics to TensorBoard.
        Model checkpoints are saved to disk if save=True.

    Notes
    -----
    - Uses Huber Loss for parameter prediction and CrossEntropyLoss for classification
    - Implements custom accuracy metrics for both prediction and classification
    - Only backpropagates prediction loss (as noted in comments)
    - Saves checkpoints every 5 epochs when save=True
    """

    ## Initializing training setup
    # Create a TensorBoard writer for logging metrics.
    writer = SummaryWriter(f"{'/logs'}/{model.name}")
    # Initialize the Adam optimizer with learning rate 1.
    optim = torch.optim.Adam(model.parameters(), lr=1)
    # Set the device to MPS (Apple Metal) if available, otherwise use CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    # Print the model name to confirm which model is being trained.
    print(f"Running {model.name}")
    # Define loss functions and accuracy metrics.
    loss_predict = nn.HuberLoss()  # For parameter prediction
    loss_classif = nn.CrossEntropyLoss()  # For time series classification
    acc_predict = custom_MSE  # Custom MSE for prediction accuracy
    acc_class = custom_acc  # Custom accuracy for classification

    ## Training loop
    for epoch in tqdm(range(epochs)):
        ## Initialize different variables for training performance tracking
        # Initialize cumulative loss and accuracy counters.
        cumloss_pred, cumloss_classif, cumloss_total, count = 0, 0, 0, 0
        cumacc_classif = 0
        # Define labels for tracking prediction accuracy of each characteristic.
        label = ['H_vigne', 'P_vigne', 'Hue_vigne', 'Hue_rang']
        cumacc_train = {i: {"score_pred": 0,
                            "label": label[i]} for i in range(len(label))}

        ## Data preparation
        # Split data into training and testing sets with temporal separation.
        X_train, X_test, index_tr_ts = time_seperate_train_test(data_total)
        # Create dataset and dataloader for training and testing.
        data_train = TimeAgrocamDataset(X_train, mode=["HSV"])
        data_test = TimeAgrocamDataset(X_test, mode=["HSV"])
        # Get labels and times for training and testing.
        label_train, time_train = data_train.get_label_time()
        label_test, time_test = data_test.get_label_time()
        # Create data loaders with custom samplers.
        train_loader = DataLoader(dataset=data_train,
                                  batch_sampler=TimePerClassSampler(label_train, time_train, time_index=index_tr_ts[0]))
        test_loader = DataLoader(dataset=data_test,
                                 batch_sampler=TimePerClassSampler(label_test, time_test, time_index=index_tr_ts[1]))

        ## Training phase
        # Set model to training mode.
        model.train()
        for z in train_loader:
            # Extract data from the batch.
            x, y_predict, y_class = z['image'], z["target"], z["cond"]

            ## Training structure
            # Zero the gradients to avoid accumulation.
            optim.zero_grad()
            # Move data to the device (MPS/CPU).
            x, y_predict, y_class = x.to(device), y_predict.to(device), y_class.to(device)
            # Forward pass: compute predictions.
            yhat_predict, yhat_classif = model(x)
            # Compute losses for prediction and classification.
            l_pred = loss_predict(yhat_predict, y_predict)
            l_classif = loss_classif(yhat_classif, y_class)
            # Combine losses with weighting (0.1% for prediction, 99.9% for classification).
            # (Loss values given by classification loss are far lower than prediction  loss)
            l = 0.001 * l_pred + 0.999 * l_classif
            # Backward pass: compute gradients and update weights.
            # Note: Only backpropagate prediction loss (as per original implementation)
            l_pred.backward()
            optim.step()

            ## Metrics calculation
            # Loop over each sample in the batch.
            for batch in range(yhat_predict.shape[0]):
                pred = yhat_predict[batch]
                target = y_predict[batch]
                # Calculate and accumulate prediction accuracy for each characteristic.
                res_acc_predict = acc_predict(pred, target).mean(axis=0)
                for o in cumacc_train:
                    cumacc_train[o]["score_pred"] += res_acc_predict[o]
                # Calculate and accumulate classification accuracy.
                cumacc_classif += acc_class(np.argmax(yhat_classif[batch].detach().numpy()),
                                            y_class[batch])
            # Update cumulative losses and sample count.
            cumloss_pred += l_pred * len(x)
            cumloss_classif += l_classif * len(x)
            cumloss_total += l * len(x)
            count += len(x)
        ## Logging training metrics
        # Log prediction accuracy for each characteristic.
        val = 0
        for k in cumacc_train:
            writer.add_scalar('train/' + cumacc_train[k]["label"] + '/MSE',
                              cumacc_train[k]["score_pred"] / count,
                              epoch)
            val += 1
        # Log classification accuracy and losses.
        writer.add_scalar('train/acc', cumacc_classif / count, epoch)
        writer.add_scalar('train/pred_loss', cumloss_pred / count, epoch)
        writer.add_scalar('train/classif_loss', cumloss_classif / count, epoch)
        writer.add_scalar('train/cumloss_total', cumloss_total / count, epoch)

        ## Validation phase
        # Set model to evaluation mode.
        model.eval()
        if epoch % 1 == 0:
            # Reset counters for validation.
            cumloss_pred, cumloss_classif, cumloss_total, count = 0, 0, 0, 0
            cumacc_classif = 0
            label = ['H_vigne', 'P_vigne', 'Hue_vigne', 'Hue_rang']
            cumacc_train = {i: {"score_pred": 0,
                                "label": label[i]} for i in range(len(label))}
            with torch.no_grad():
                for z in test_loader:
                    # Extract and prepare validation data.
                    x, y_predict, y_class = z['image'], z["target"], z["cond"]
                    x, y_predict, y_class = x.to(device), y_predict.to(device), y_class.to(device)
                    # Forward pass for validation.
                    yhat_predict, yhat_classif = model(x)
                    # Calculate validation metrics.
                    for batch in range(yhat_predict.shape[0]):
                        pred = yhat_predict[batch]
                        target = y_predict[batch]
                        res_acc_predict = acc_predict(pred, target).mean(axis=0)

                        for o in cumacc_train:
                            cumacc_train[o]["score_pred"] += res_acc_predict[o]

                        cumacc_classif += acc_class(np.argmax(yhat_classif[batch].detach().numpy()),
                                                    y_class[batch])
                    count += len(x)
                # Log validation metrics.
                val = 0
                for k in cumacc_train:
                    writer.add_scalar('test/' + cumacc_train[k]["label"] + '/MSE',
                                      cumacc_train[k]["score_pred"] / count,
                                      epoch)
                    val += 1
                writer.add_scalar('test/acc', cumacc_classif / count, epoch)

        ## Saving model checkpoints
        if save and (epoch % 5 == 0):
            # Save model checkpoint every 5 epochs.
            torch.save(model.state_dict(),
                       "checkpoint/" + model.name + '_checkpoint.pth')

def data_filter(data_total, treatment, time_start):
    """
    Filter and prepare time-series agronomic data for a specific treatment and time period 
    in order to make a prediction of futur agronomic characteristics.

    This function filters the input dataset for a specific treatment condition and a time period, 
    handles missing data through interpolation, and return the indices matching the input time period 
    and what would be the time predicted.

    Parameters
    ----------
    data_total : pandas.DataFrame
        Input DataFrame containing time-series agronomic data with columns:
        - "treatment": Experimental treatment condition
        - "time": Timestamp for each record
        - "image": Path to image (with no Image if there is none associated with the timestamp)
        - Other agronomic parameter columns
    time_start : str, optional
        Start date for the time series used for the prediction in 'YYYY-MM-DD' format.
    treatment : str, optional
        Experimental condition of the vineyard where the prediction is done.

    Returns
    -------
    input_index
        List of indices corresponding to the input time period (15 days starting from time_start).
    time_verif
        DatetimeIndex representing the verification period (15 days after the last day of the input period)
    """
    
    ## Data Preparation
    # Filter data for the specified condition.
    data_represent = data_total[data_total["treatment"] == treatment]
    # Initialize index counters.
    max_indx = 0
    min_indx = 0

    ## Interpolating missing characteristics (due to missing time)
    # Store original indices and set time as index.
    data_represent.loc[:, "index_second"] = data_represent.index
    data_represent["time"] = pd.to_datetime(data_represent["time"])
    data_represent.set_index("time", inplace=True, drop=False)
    # Create a complete date range which is used to reveal the missing dates in the data.
    time_corr = pd.date_range(start=min(data_represent.index),
                              end=max(data_represent.index),
                              normalize=True,
                              freq="D")
    data_set_filt = data_represent.reindex(time_corr)
    # Interpolate missing values and fill NaN in treatment and image columns.
    data_set_filt = data_set_filt.interpolate(limit_direction='both')
    data_set_filt.loc[:, "treatment"] = data_set_filt.loc[:, "treatment"].fillna(treatment)
    data_set_filt.loc[:, "image"] = data_set_filt.loc[:, "image"].fillna("No Image")
    # Update time and index columns.
    data_set_filt.loc[:, "time"] = data_set_filt.index.date
    data_set_filt.index_second = data_set_filt.index_second.astype(int)
    # Update indices for the filtered dataset.
    max_indx += len(data_set_filt.loc[:, "index_second"])
    data_set_filt.loc[:, "index_second"] = [j for j in range(min_indx, max_indx)]
    min_indx = max_indx
    data_set_filt.set_index("index_second", inplace=True)
    # Update the working dataset.
    data_represent = data_set_filt

    ## Generation of the input characteristics in the given time range.
    # Generate input time ranges.
    time_input = pd.date_range(start=time_start, periods=15, normalize=True)
    time_verif = time_input + timedelta(days=15)
    data_represent["time"] = [d.date() for d in data_represent["time"]]
    time_input = [d.date() for d in time_input]
    # Extract indices for input times.
    input_index = [list(data_represent.loc[data_represent["time"].values == i, :].index.values)[0]
                   for i in time_input
                   if i in data_represent["time"].values]
    return input_index, time_verif

if __name__ == "__main__":
    import argparse
    import importlib
    
    ## Ask the relevant arguments
    parser = argparse.ArgumentParser(description='Train or Use a prediction model for vineyard agronomic characteristics from a set of images.')
    parser.add_argument('--lstm_model', type=str, required=False, default="model.cnn_lstm.CNN_LSTM",
                        help='Class (in the designated package) of the LSTM used as a prediction model')
    parser.add_argument('--weight_url_cnn', type=str, required=False, default="No_weight",
                        help='Import weight of the cnn model. If given "No_weight", pretrained weights for MobileNetV3 are used')
    parser.add_argument('--weight_url_lstm', type=str, required=False, default="Prediction/checkpoint/MobileNet3_LSTM_checkpoint_final_hsv_notbi_norm.pth",
                        help='Import weight of the lstm model.')
    parser.add_argument('--train_or_predict', type=str, required=False, 
                        help='Choose to train an algorithm or use it to predict agronomic characteristics', default="predict")
    parser.add_argument('--chara_chosen', type=str, required=False, 
                        help='Input which characteristics will be predicted', default="[H_vigne,P_vigne,Hue_vigne,Hue_rang]")
    # Argments for training
    parser.add_argument('--epochs', type=int, required=False, help='Number of epochs for training', 
                        default=10)
    # Argments for prediction
    parser.add_argument('--time_start', type=str, required=False, help='Start date for prediction', 
                        default="2024-04-20")
    parser.add_argument('--treatment', type=str, required=False, help='Treatment of vine where the images were taken', 
                        default="AVITI")
    # Store the parsed arguments
    args = parser.parse_args()
    
    from model.mobilenet_LRASPP import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large
    
    ## Prepare the images and model
    # Loading the format to test
    format_strings = args.string_for_list_format
    format_list = str.split(format_strings, ",")
    format_list[0] = format_list[0][1:]
    format_list[-1] = format_list[-1][0:-1]
    # Generate the database for all the images and characteristics
    data = pd.read_csv("Core/Results/Agro_chara_vine.csv")
    # Generate the model and its pretrained weight
    if args.lstm_model != "no_CNN_LSTM":
        # If the model chosen is an hybrid between CNN and LSTM, the cnn model is loaded
        model_cnn = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1, classif = True)
        model_cnn.classifier.low_classifier = torch.nn.Conv2d(40, 5, kernel_size=(1, 1), stride=(1, 1))
        model_cnn.classifier.high_classifier = torch.nn.Conv2d(128, 5, kernel_size=(1, 1), stride=(1, 1))
        if args.weight_url_cnn != "No_weight" :
            model_cnn.load_state_dict(torch.load(args.weight_url_cnn))
            print("Weight loaded !!")
        # NEED TO HAD THE CHANGE TO THE FIRST LAYER WITH THE CHANNEL NUMBER CHANGING WITH CLASS
        model_cnn.name = "Mobilenetv3"
    
    # Loading the lstm model
    package = str.split(args.lstm_model, ".")[str.count(args.lstm_model, ".")]
    path_package = str.replace(args.lstm_model, package, "")
    model_lstm_pck = importlib.import_module(path_package[:len(path_package)-1])
    model_lstm_mod = getattr(model_lstm_pck, package)
    
    if "no_CNN_LSTM" not in args.lstm_model:
        model_lstm = model_lstm_mod(CNN = model_cnn, num_layers = 10, bidir=False)
        weigth = torch.load(args.weight_url_lstm, weights_only=True)
    else:
        model_lstm = model_lstm_mod(num_layers = 10, bidir=False)
        # For this specific weight, the cnn was implemented in the model (which it didn't used)
        # So, I'm sure that it saved the cnn weight in addition to the lstm weight
        # The next line of code aims at rectifying the mistake
        if "MobileNet3_LSTM_test_nocnn_checkpoint.pth" in args.weight_url_lstm:
            weigth = torch.load(args.weight_url_lstm, weights_only=True)
            list_key_del = []
            for i in weigth.keys():
                if "cnn.backbone" in i or "cnn.classifier" in i:
                    list_key_del.append(i)
            for j in list_key_del:
                del weigth[j]
    model_lstm.load_state_dict(weigth)
    
    if args.train_or_predict == "train":
        ## Train the model
        epochs = args.epochs
        time_train(model_lstm, epochs, data, save=True, mode=["HSV"])
    else:
        trans = transforms.Compose([transforms.ToTensor()])
        mean_norm, std_norm = np.mean(data[format_list], axis=0) , np.std(data[format_list], axis=0)
        # Generating the index of the data with the start date and the following date
        input_index, time_verif = data_filter(data, args.treatment, args.time_start)
        # Generating the images of each image
        data_represent_set = TimeAgrocamDataset(data)
        data_input = data_represent_set[input_index]
        input_image = data_input["image"]
        input_image = np.expand_dims(input_image, 0)  # Add batch dimension
        ## Model Predictions
        # Set all models to evaluation mode.
        model_lstm.eval()
        # Generate predictions from all the model.
        res = model_lstm(input_image)
        res_pred_1 = res[0].detach().numpy()
        result_pred_1 = pd.DataFrame(res_pred_1.squeeze(0),
                                     index=time_verif,
                                     columns=format_list)
        # Denormalize all model predictions.
        result_pred_1 = (result_pred_1 * std_norm) + mean_norm
        print(result_pred_1)

