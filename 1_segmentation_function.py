# General function used to load images and create dataset with those images

# Create environment 
# python -m .venv_seg
# py -m pip install -r requirements.txt
# 

import os
import PIL.Image
import base64
import requests
from io import BytesIO
from typing import Optional, Union

def load_image(image: Union[str, "PIL.Image.Image"], timeout: Optional[float] = None, format = ["RGB"]) -> "PIL.Image.Image":
    # Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/image_utils.py
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.
        format ('list'): (added)
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
    if len(format) == 1:
        image_fin = image.convert(format[0])
        image_fin = np.array(image_fin)
    else :
        for m in range(len(format)) :
            if m==0:
                image_fin = image.convert(format[m])
                image_fin = np.array(image_fin)
            else:
                image_temp = image.convert(format[m])
                image_temp = np.array(image_temp)
                image_fin = np.append(image_fin, image_temp, axis=2)
    return image_fin

import numpy as np
from pandas import DataFrame, to_datetime
# import os
# import PIL.Image

def data_loading(img_path: str = "Images/segmentation_images/images", target_path: str = "Results/all_images_mask") -> "DataFrame":
    """
    Load and store the Agrocam image and corresponding mask into a structured dataset for training or evaluation.

    This function extracts image paths and corresponding masks and associated metadata, and organizes
    the data into a structured DataFrame. It also calculates the distance between the top of the
    trunk and the bottom of the canopy for each image used to calculate the vegetation porosity.

    Parameters
    ----------
    img_path : str, optional
        Path to the directory containing input images. Default is "image".
    target_path : str, optional
        Path to the directory containing ground truth masks. Default is "masque_final".

    Returns
    -------
    pd.DataFrame
        DataFrame containing image paths, mask paths, conditions (aka treatments), day when the image is taken and calculated distances.
    """
    # List all files in the image directory.
    all_image = os.listdir(img_path + "/")
    img_data = []

    # Processing each image
    for i in all_image:
        # Skip non-image files.
        if not ((".jpg" in i) or (".png" in i)):
            continue
        
        # Extract camera ID from the filename.
        id_cam = i.split("_")[0]
        # Get the treatment for the camera ID.
        treatment_trad = {"79bt3wkh" : "TVITI", "7s3a5abm" : "AVITI", "4j7g2wk9" : "DVITI"}
        cond = treatment_trad[id_cam]
        # Extract date and time from the filename.
        times = i.split("_")

        # Remove the file extension for mask path construction.
        i_remove = i.replace(".jpg", "")
        # Retreiving path
        img = img_path + '/' + i

        # Constructing data entries
        mask = target_path + "/" + i_remove + "__mask.png"
        l_add = [times[1], times[2], cond, img, mask]
        img_data.append(l_add)

    # Create DataFrame with columns for training data.
    img_data = DataFrame(
        img_data,
        columns=["day", "time", "treatment", "image", 'mask']
        )
    
    # Combine day and time into a single datetime column.
    img_data["day_time"] = img_data["day"] + " " + img_data["time"]
    img_data['day_time'] = to_datetime(img_data['day_time'], format="%Y-%m-%d %H%M%S")
    # Drop the separate day and time columns.
    img_data = img_data.drop(columns=["day", "time"])

    return img_data

## Metrics and Function used for learning
# Metrics used

def IoU(pred, target, dim_z=0):
    """
    Calculate the Intersection over Union (IoU) score of binary masks for semantic segmentation.

    IoU measures the overlap between the predicted segmentation and the ground truth.
    It is defined as the ratio of the intersection to the union of the predicted and target masks.

    Parameters
    ----------
    pred : np.array
        Predicted binary mask or an image where the mask has been applied.
    target : np.array
        Ground truth binary mask or an image where the mask has been applied.
    dim_z : int, optional
        Number of parameters representing each pixel. 
        For example, a pixel represented in a RGB format have 3 parameters.
        
        If 0, a pixel is represented with a scalar. If greater than 0, it is represented by a vector of length `dim_z`. 
        Default is 0.

    Returns
    -------
    score : float
        IoU score, rounded to 4 decimal places. 
        Returns `np.nan` if the mask predicted and ground truth doesn't have any positive pixel.
    """

    ## Creating masks for intersection and union
    # If dim_z is 0, create masks by comparing with scalar 0.
    if dim_z == 0:
        pred_mask = pred != 0
        target_mask = target != 0
    # Otherwise, create masks by comparing with a zero vector of length dim_z.
    else:
        pred_mask = pred != [0] * dim_z
        target_mask = target != [0] * dim_z

    ## Calculating intersection and union
    # Intersection: Pixels where both prediction and target are positive (one).
    intersec = np.logical_and(pred_mask, target_mask)
    # Union: Pixels where either prediction or target is positive (one).
    union = np.logical_or(pred_mask, target_mask)

    ## Handling edge case where union is zero
    # If the union is zero, return NaN to avoid division by zero.
    if union.sum() == 0:
        return np.nan
    # Otherwise, calculate IoU as intersection / union.
    else:
        score = intersec.sum() / union.sum()
    # Round the IoU score to 4 decimal places for readability.
    return round(score, 4)

def specificity(pred, target, dim_z=0):
    """
    Calculate the specificity score of binary masks for semantic segmentation.

    Specificity measures the proportion of negatives that are correctly identified by the prediction.
    It is defined as TN / (TN + FP), where TN is true negatives and FP is false positives.
    
    Its application to semantic segmentation consist of superposing predicted and ground truth pixel and comparing their classes.

    Parameters
    ----------
    pred : np.array
        Predicted binary mask or an image where the mask has been applied.
    target : np.array
        Ground truth binary mask or an image where the mask has been applied.
    dim_z : int, optional
        Number of parameters representing each pixel. 
        For example, a pixel represented in a RGB format have 3 parameters.
        
        If 0, a pixel is represented with a scalar. If greater than 0, it is represented by a vector of length `dim_z`. 
        Default is 0.

    Returns
    -------
    score : float
        Specificity score, rounded to 4 decimal places. 
        Returns `np.nan` if the mask predicted and ground truth doesn't have any negative pixel.
    """

    ## Creating masks for true negatives and false positives (if an image is given).
    # If dim_z is 0, create masks by comparing with scalar 0.
    if dim_z == 0:
        pred_mask = pred == 0
        target_mask = target == 0
    # Otherwise, create masks by comparing with a zero vector of length dim_z.
    else:
        pred_mask = pred == [0] * dim_z
        target_mask = target == [0] * dim_z

    ## Calculating true negatives (VN) and false positives (FP)
    # VN: Pixels where both prediction and target are negative (zero).
    VN = np.logical_and(pred_mask, target_mask)
    # FP: Pixels where prediction is positive (one) but target is negative (zero).
    FP = np.logical_and(~pred_mask, ~target_mask)

    ## Calculating the specificity score
    # If there are no true negatives or false positives, return NaN to avoid division by zero.
    if (VN.sum() + FP.sum()) == 0:
        return np.nan
    # Otherwise, calculate specificity as TN / (TN + FP).
    else:
        score = VN.sum() / (VN.sum() + FP.sum())
    # Round the specificity score to 4 decimal places for readability.
    return round(score, 4)

def sensitivity(pred, target, dim_z=0):
    """
    Calculate the sensitivity score of binary masks for semantic segmentation.
        
    Sensitivity measures the proportion of positives that are correctly identified by the prediction.
    It is defined as TP / (TP + FN), where TP is true positives and FN is false negatives.
    
    Its application to semantic segmentation consist of superposing predicted and ground truth pixel and comparing their classes.

    Parameters
    ----------
    pred : np.array
        Predicted binary mask or an image where the mask has been applied.
    target : np.array
        Ground truth binary mask or an image where the mask has been applied.
    dim_z : int, optional
        Number of parameters representing each pixel. 
        For example, a pixel represented in a RGB format have 3 parameters.
        
        If 0, a pixel is represented with a scalar. If greater than 0, it is represented by a vector of length `dim_z`. 
        Default is 0.

    Returns
    -------
    score : float
        Sensitivity score, rounded to 4 decimal places. 
        Returns `np.nan` if the mask predicted and ground truth doesn't have any positive pixel.
    """

    ## Creating masks for true positives and false negatives (if an image is given)
    # If dim_z is 0, create masks by comparing with scalar 0.
    if dim_z == 0:
        pred_mask = pred != 0
        target_mask = target != 0
    # Otherwise, create masks by comparing with a zero vector of length dim_z.
    else:
        pred_mask = pred != [0] * dim_z
        target_mask = target != [0] * dim_z

    ## Calculating true positives (VP) and false negatives (FN)
    # VP: Pixels where both prediction and target are positive (one).
    VP = np.logical_and(pred_mask, target_mask)
    # FN: Pixels where prediction is negative (zero) but target is positive (one).
    FN = np.logical_and(~pred_mask, target_mask)

    ## Calculating the sensitivity score
    # If there are no true positives or false negatives, return NaN to avoid division by zero.
    if (VP.sum() + FN.sum()) == 0:
        return np.nan
    # Otherwise, calculate sensitivity as TP / (TP + FN).
    else:
        score = VP.sum() / (VP.sum() + FN.sum())
    # Round the sensitivity score to 4 decimal places for readability.
    return round(score, 4)

# Function for learning

from sklearn.model_selection import StratifiedShuffleSplit

def seperate_train_test(data_set):
    """Separate the Agrocam images between train and test sets.

    Stratified shuffling is used to preserve the distribution of treatment groups in both train and test sets.

    Parameters
    ----------
    data_set : pd.DataFrame
        DataFrame with the URLs for the original images and the target images with all their
        characteristics (e.g., treatments, date).

    Returns
    -------
    train : list of int
        indiceses referring to the images used for training (66% of the data).
    test : list of int
        indiceses referring to the images used for testing (33% of the data).
    """
    
    # StratifiedShuffleSplit is a cross validation method which split the data into train and test (66% of the data in train and 33% in testing).
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33)
    # It also ensure that the class distribution (treatments stored in "label") is preserved in both train and test sets.
    label = data_set.loc[:, "treatment"]

    # Generate the split indices for the dataset.
    # `sss.split` returns a generator; `next` retrieves the first (and only) split.
    # Returns the tuple : (train_indices, test_indices).
    indices = next(sss.split(data_set, label))

    # To help visualize the indices used for training are stored in train and the testing ones are in test
    train = indices[0]
    test = indices[1]

    return train, test

# Inspired by https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

trans = transforms.Compose([transforms.ToTensor()])
class AgrocamDataset(Dataset):
    """
    Representation of the Agrocam dataset for use with PyTorch DataLoader to train an algorithm to segment an Agrocam Image into several zone of interest.

    This class loads and preprocesses the Agrocam database (images and their respective target maks) 
    for it to give directly, at each indices, both images at a given format .

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least the URLs for the original images, target images,
        and the treatment group for each image.
    format : list of str, optional
        List of color spaces for image representation (e.g., ["RGB"], ["HSV"], ["LAB"]).
        Default is ["RGB"].

    Returns
    -------
    sample : dict
        Dictionary with two keys:
        - 'image' (torch.Tensor): Original image of vineyards in the specified color space.
        - 'target' (torch.Tensor): Target image with benchmark clustering for each class.
    """

    def __init__(self, data, format=["RGB"]):
        ## Initializing the function
        # Reset the DataFrame indices to ensure consistent indicesing.
        self.data = data.reset_index()
        # Extract the 'treatment' column as labels for the dataset.
        self.label = self.data.loc[:, "treatment"]
        # Assign the image transformation pipeline (e.g., normalization, resizing).
        self.transform = trans
        # Store the color space format(s) for image loading.
        self.format = format

    def __len__(self):
        ## Return the total number of samples in the dataset, based on the number of labels.
        return len(self.label)

    def __getitem__(self, idx):
        ## Convert tensor indices to list if necessary (for compatibility with DataFrame indicesing).
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ## Loading the different images
        # Load the original image in the specified color space 
        image = load_image(self.data.loc[idx, "image"], format=self.format)
        # Load the image representing all the mask in grayscale ("L" format).
        # In this image, each pixels is attributed with a number (0-4) representing a class
        target = load_image(self.data.loc[idx, "mask"], format="L")

        ## Changing the format the maks image so that it can be used by the algorithm
        # This new format assign one binary layer (e.g one binary mask) for each class (4 class + background)
        new_target = np.zeros((5, 1080, 1920))
        for i in range(0, 5):
            extract_target = (target == i) * 1
            new_target[i] = extract_target
        # Rearrange the target array to (Length, Width, Layers) format for compatibility with PyTorch.
        new_target = np.moveaxis(new_target, 0, 2)
        
        ## Apply transformations (e.g., normalization) to both image and target.
        if self.transform:
            image = self.transform(image)
            new_target = self.transform(new_target)

        ## Return the sample as a dictionary with 'image' and 'target' tensors.
        sample = {"image": image, "target": new_target}
        return sample

# Adapted through a creation by Mistral 
# Request : Hello. Can you write a custom sampler for Pytorch which correspond to the following description :
# “““
# At every iteration, this will return m samples per class, assuming that the batch size is a multiple of m. 
# For example, if your dataloader's batch size is 100, and m = 5, then 20 classes with 5 samples each will be 
# returned. Note that if batch_size is not specified, then most batches will have m samples per class, 
# but it's not guaranteed for every batch.
# “““

from torch.utils.data.sampler import Sampler

class MSamplesPerClassSampler(Sampler):
    """
    Custom sampler to ensure the dataloader, for each iteration, is returning a batch 
    containing one image for each treatment (e.g. 3 image per batch)
    
    This sampler balances treatment representation in each batch by either :
    oversampling under-represented treatment or undersampling over-represented treatment.

    Parameters
    ----------
    labels : array-like
        Array of class labels for the dataset.

    Attributes
    ----------
    classes : ndarray
        Unique class labels in the dataset.
    num_classes : int
        Number of unique classes.
    class_to_indices : dict
        Mapping from each class to the indices of its samples in the dataset.
    samples_per_class : dict
        Number of samples available for each class.
    balance_method : str
        Method used for balancing ('oversample', 'undersample', or 'nope').
    num_batches : int
        Number of batches that can be generated from the dataset.
    """

    def __init__(self, labels):
        ## Mapping each treatment (named class) with every indices (named samples) of the dataset with an image of this treatment.
        self.classes = np.unique(labels)
        self.num_classes = len(self.classes)
        self.class_to_indices = {cls: np.where(labels == cls)[0] for cls in self.classes}
        
        ## Initializing variables to use for class balance in the whole dataset
        # Calculate the number of samples per class.
        self.samples_per_class = {cls: len(indices) for cls, indices in self.class_to_indices.items()}
        # Identify the classes with the maximum and minimum number of samples.
        max_samples = max(self.samples_per_class.values())
        min_samples = min(self.samples_per_class.values())
        num_max = [i for i in range(len(self.samples_per_class.values()))
                   if list(self.samples_per_class.values())[i] == max_samples]
        num_min = [i for i in range(len(self.samples_per_class.values()))
                   if list(self.samples_per_class.values())[i] == min_samples]
        
        ## Determining if a balancing method is needed (and which one) or not
        # If there are less classes with a minimum number of samples, oversample those classes.
        if len(num_min) < len(num_max):
            self.balance_method = 'oversample'
        # If there are more classes with a minimum number of samples, undersample the other classes.
        elif len(num_min) > len(num_max):
            self.balance_method = 'undersample'
        # If all classes have the same number of samples, no balancing is needed.
        elif len(num_min) == len(self.samples_per_class.values()) and len(num_max) == len(self.samples_per_class.values()):
            print("ATTENTION: Well balanced labels")
            self.balance_method = 'nope'
        # If each class have different number of samples, a random balancing method is chosen.
        elif len(self.samples_per_class.values()) == len(np.unique(self.samples_per_class.values())[0]):
            balance_method = np.random.choice(['oversample', 'undersample'])
        # If the dataset is unbalanced but doesn't fit the above criteria, no balancing is applied.
        else:
            print("ATTENTION: Unbalanced labels")
            self.balance_method = 'nope'
        
        ## Balance the samples per class using the chosen method.        
        if self.balance_method == 'oversample':
            # Oversample under-represented (having the least amount of samples) classes to match the maximum number of samples.
            max_samples = max(self.samples_per_class.values())
            for cls in self.classes:
                while len(self.class_to_indices[cls]) < max_samples:
                    # Randomly duplicate an indices from the under-represented class.
                    random_indices = np.random.choice(self.class_to_indices[cls])
                    self.class_to_indices[cls] = np.append(self.class_to_indices[cls], random_indices)
        if self.balance_method == 'undersample':
            # Undersample over-represented (having the most amount of samples) classes to match the minimum number of samples.
            min_samples = min(self.samples_per_class.values())
            for cls in self.classes:
                while len(self.class_to_indices[cls]) > min_samples:
                    # Randomly remove an indices from the over-represented class.
                    random_indices = np.random.choice(self.class_to_indices[cls])
                    self.class_to_indices[cls] = np.delete(self.class_to_indices[cls],
                                                          np.where(self.class_to_indices[cls] == random_indices)[0][0])
        
        ## Recalculate the number of samples per class after balancing and calculate the number of batches
        self.samples_per_class = {cls: len(indices) for cls, indices in self.class_to_indices.items()}
        self.num_batches = min(self.samples_per_class.values()) // 1

    def __iter__(self):
        ## Generating batches
        # (This is the function used by Dataloader to generate the indices of each batch (one per classes aka treatment))
        # Shuffle the indices for each class to ensure randomness.
        for cls in self.class_to_indices:
            np.random.shuffle(self.class_to_indices[cls])
        # Generate batches of 3 indices (1 of each class).
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                indices = self.class_to_indices[cls][:1]
                batch.extend(indices)
            yield batch

    def __len__(self):
        ## Return the number of batches.
        return self.num_batches

# Learning function of Vincent Guigue adapted for our purpose (https://github.com/vguigue/tuto_deep/blob/main/notebooks/2_1-CNN.ipynb)
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

def train(model, epochs, data, save=True, format=["RGB"]):
    """
    Train a PyTorch model for semantic segmentation on the Agrocam dataset.
    This function handles the training loop, validation, metrics monitoring, and model checkpointing.
    It supports focal loss for segmentation and tracks its loss, and the IoU, sensitivity, and specificity for each class.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    epochs : int
        Number of training epochs.
    data : pd.DataFrame
        DataFrame containing image paths, target paths, and treatment conditions.
    save : bool, optional
        If True, saves the model checkpoint after training. Default is True.
    format : list of str, optional
        List of color spaces for image representation (e.g., ["RGB"]). Default is ["RGB"].

    Returns
    -------
    Metrics :
        The metrics are saved in a logs (located at the root) folder that can be accessed with TensorBoard.
    Model :
        If save is True, the function saves the model checkpoint as a pth file (located in the folder "checkpoint" where the script is).
    """

    ## Initializing training setup
    # Create a TensorBoard writer for logging metrics.
    writer = SummaryWriter(f"{'/logs'}/{model.name}")
    # Initialize the Adam optimizer with a learning rate of 1e-3.
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Set the device to MPS (Apple Metal) if available, otherwise use CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    # Define the loss function (sigmoid focal loss).
    loss = sigmoid_focal_loss
    print(f"Running {model.name}")

    for epoch in tqdm(range(epochs)):
        ## Initialize different variables for training
        # Initialize cumulative loss and sample count for the epoch.
        cumloss, count = 0, 0
        # Initialize dictionaries to accumulate accuracy, sensitivity, and specificity for each class.
        label = ["bck", "feuille", "interrang", "fil", "trunc"]
        cumacc_train = {i: {"score_acc": 0,
                            "score_sens": 0,
                            "score_speci": 0,
                            "label": label[i]} for i in range(5)}
        idx = 0

        ## Data preparation
        # Split the dataset into training and testing sets.
        indices_train, indices_test = seperate_train_test(data)
        X_train, X_test = data.loc[indices_train, :], data.loc[indices_test, :]
        # Convert the train and test sets to AgrocamDataset
        data_train = AgrocamDataset(X_train, format=format)
        data_test = AgrocamDataset(X_test, format=format)
        # Initialize a dataloader using MSamplesPerClassSampler sampler.
        train_loader = DataLoader(dataset=data_train,
                                  batch_sampler=MSamplesPerClassSampler(data_train.label))
        test_loader = DataLoader(dataset=data_test,
                                 batch_sampler=MSamplesPerClassSampler(data_test.label))

        ## Training phase
        # Set the model to training format.
        model.train()
        for z in train_loader:  # Loop over batches
            idx += 1
            x, y = z['image'], z['target']  # Extract images and targets from the batch

            ## Training structure
            # Zero the gradients to avoid accumulation.
            optim.zero_grad()
            # Move data to the device (MPS/CPU).
            x, y = x.to(device), y.to(device)
            # Forward pass: compute predictions.
            yhat = model(x)
            # Extract predictions for DeepLabv3 and MobileNetV3 models.
            if model.name == "Deeplab3" or model.name == "Mobilenetv3":
                yhat = yhat["out"]
            # Compute the loss.
            l = loss(yhat, y, reduction="mean")
            # Backward pass: compute gradients.
            l.backward()
            # Update model parameters.
            optim.step()

            ## Metrics calculation
            # Loop over each sample in the batch.
            for batch in range(yhat.shape[0]):
                # Get predicted class for each pixel as an unified images.
                all_target = np.argmax(yhat[batch].detach().numpy(), axis=0)
                # Looping through all classes
                for o in cumacc_train:
                    # Extract the predicted and target binary mask of the selected class
                    pred = (all_target == o) * 1.0
                    target = np.array(y[batch][o].tolist())
                    # Compute and add the IoU, sensitivity, and specificity of the class
                    cumacc_train[o]["score_acc"] += IoU(pred, target)
                    cumacc_train[o]["score_sens"] += sensitivity(pred, target)
                    cumacc_train[o]["score_speci"] += specificity(pred, target)
            # Update the total sample count and cumulative loss.
            count += len(x)
            cumloss += l * len(x)

        ## Logging training metrics
        val = 0
        mean_acc_cum = 0
        for k in cumacc_train:
            # Log sensitivity, specificity, and IoU for each class.
            writer.add_scalar('train/' + cumacc_train[k]["label"] + '/sensibility',
                              cumacc_train[k]["score_sens"] / count,
                              epoch)
            writer.add_scalar('train/' + cumacc_train[k]["label"] + '/specificity',
                              cumacc_train[k]["score_speci"] / count,
                              epoch)
            writer.add_scalar('train/' + cumacc_train[k]["label"] + '/iou',
                              cumacc_train[k]["score_acc"] / count,
                              epoch)
            mean_acc_cum += cumacc_train[k]["score_acc"] / count
            val += 1
        # Log the average loss and total IoU for the epoch.
        writer.add_scalar('train/loss', cumloss / count, epoch)
        writer.add_scalar('train/iou_tot', mean_acc_cum / val, epoch)

        ## Validation phase
        if epoch % 1 == 0:
            ## Initialize different variables for testing
            # Initialize cumulative loss and sample count for the epoch.
            cumloss, count = 0, 0
            # Initialize dictionaries to accumulate accuracy, sensitivity, and specificity for each class.
            label = ['bck', "feuille", "interrang", "fil", "trunc"]
            cumacc_test = {i: {"score_acc": 0,
                               "score_sens": 0,
                               "score_speci": 0,
                               "label": label[i]} for i in range(5)}
            idx = 0

            # Set the model to evaluation format.
            model.eval()
            with torch.no_grad():  # Disable gradient computation for validation.
                for w in test_loader:
                    ## Compute the mask prediction of the model
                    idx += 1
                    x, y = w['image'], w['target']
                    x, y = x.to(device), y.to(device)
                    yhat = model(x)
                    # Extract predictions for DeepLabv3 and MobileNetV3 models.
                    if model.name == "deeplab3_new_" or model.name == "Mobilenetv3":
                        yhat = yhat["out"]

                    ## Metrics calculation
                    # Loop over each sample in the batch.
                    for batch in range(yhat.shape[0]):
                        # Get predicted class for each pixel as an unified images.
                        all_target = np.argmax(yhat[batch].detach().numpy(), axis=0)
                        # Looping through all classes
                        for o in cumacc_test:
                            # Extract the predicted and target binary mask of the selected class
                            pred = (all_target == o) * 1.0
                            target = np.array(y[batch][o].tolist())
                            # Compute and add the IoU, sensitivity, and specificity of the class
                            cumacc_test[o]["score_acc"] += IoU(pred, target)
                            cumacc_test[o]["score_sens"] += sensitivity(pred, target)
                            cumacc_test[o]["score_speci"] += specificity(pred, target)
                    # Update the total sample count.
                    count += len(x)

                ## Logging validation metrics
                val = 0
                mean_acc_cum = 0
                for k in cumacc_test:
                    # Log sensitivity, specificity, and IoU for each class.
                    writer.add_scalar('test/' + cumacc_test[k]["label"] + '/sensibility',
                                      cumacc_test[k]["score_sens"] / count,
                                      epoch)
                    writer.add_scalar('test/' + cumacc_test[k]["label"] + '/specificity',
                                      cumacc_test[k]["score_speci"] / count,
                                      epoch)
                    writer.add_scalar('test/' + cumacc_test[k]["label"] + '/iou',
                                      cumacc_test[k]["score_acc"] / count,
                                      epoch)
                    mean_acc_cum += cumacc_test[k]["score_acc"] / count
                    val += 1
                # Log the total IoU for validation.
                writer.add_scalar('test/iou_tot/', mean_acc_cum / val, epoch)

    ## Saving the model as a checkpoint
    if save:
        ## Adding a checkpoint folder to save the model weights
        # Define the path to the Segmentation folder
        checkpoint_folder = os.path.join("Segmentation", "checkpoint")
        # Check if the checkpoint folder exists, if not, create it
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        torch.save(model.state_dict(),"Segmentation/checkpoint/" + model.name + '_checkpoint.pth')

if __name__ == "__main__":
    import argparse
    ## Ask the relevant arguments
    parser = argparse.ArgumentParser(description='Train or Use a segmentation model on a set of images.')
    # Arguments for the generation of the training database
    parser.add_argument('--folder_url_train_img', type=str, required=False, default="Core/Images/segmentation_images/images",
                        help='URL of the folder containing images for segmentation or training.')
    parser.add_argument('--folder_url_train_mask', type=str, required=False, default="Core/Images/segmentation_images/masks",
                        help='URL of the folder containing related mask. When training they correspond to ground truth mask and when segmenting, they correspond to predicted mask.')
    parser.add_argument('--train_or_segment', type=str, required=False, default="segment",
                        help='Choose to train an algorithm or use it to segment images.')
    parser.add_argument('--weight_url', type=str, required=False, default="No_weight",
                        help='Import weight of the model. If not used, pretrained weights for MobileNetV3 are used.')
    parser.add_argument('--format_used', type=str, required=False, default="HSV",
                        help='Image format used for training the model and for generating the different mask.')
    # Argments for training
    parser.add_argument('--epochs', type=int, required=False, 
                        help='Number of epochs for training', 
                        default=10)
    parser.add_argument('--saving', type=bool, required=False, 
                        help='Do we want to save the model weights ?', 
                        default=True)
    # Store the parsed arguments
    args = parser.parse_args()
    
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
    
    ## Prepare the images and model
    # Generate the database for all the images
    img_url = args.folder_url_train_img
    if args.train_or_segment == "train":
        ## Training Database
        mask_url = args.folder_url_train_mask
        data = data_loading(img_url, mask_url)
    else :
        ## Segmentation Database
        mask_url = "Core/Results/all_images_mask"
        data = data_loading(img_url, mask_url)
    
    ## Generate the model and its pretrained weight
    model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
    model.classifier.low_classifier = torch.nn.Conv2d(40, 5, kernel_size=(1, 1), stride=(1, 1))
    model.classifier.high_classifier = torch.nn.Conv2d(128, 5, kernel_size=(1, 1), stride=(1, 1))
    # Extracting the format used in a list
    format_used = args.format_used
    if "-" in format_used:
        format_used_list = str.split(format_used, "-")
    else:
        format_used_list = [format_used]
    # Changing the first convolution layer of the model so that it fits the number of channels
    Number_of_channels = len("".join(format_used_list))
    if Number_of_channels != 3:
        from functools import partial
        from torch.nn import BatchNorm2d, Hardswish
        from torchvision.ops.misc import Conv2dNormActivation
        model.backbone["0"] = Conv2dNormActivation(Number_of_channels,
                                            16,
                                            kernel_size=3,
                                            stride=2,
                                            norm_layer=partial(BatchNorm2d, eps=0.001, momentum=0.01),
                                            activation_layer=Hardswish)
    if args.weight_url != "No_weight" :
        model.load_state_dict(torch.load(args.weight_url))
        print("Weight loaded !!")
    model.name = "Mobilenetv3"
    
    if args.train_or_segment == "train":
        ## Train the model
        epochs = args.epochs
        saving = args.saving
        train(model, epochs, data, save=saving, format=format_used_list)
    else:
        model.eval()
        trans = transforms.Compose([transforms.ToTensor()])
        ## Apply the segmentation on a set of images
        with torch.no_grad():
            for i in tqdm(range(data.shape[0])):
                # Load the image
                img = load_image(data.loc[i, "image"], format = format_used_list)
                img = trans(img)
                img = torch.unsqueeze(img, 0)
                # Generate its predicted mask
                img_mask = model(img)['out'].numpy()
                img_mask = np.argmax(img_mask[0], axis=0)
                # Save the mask
                image = PIL.Image.fromarray(img_mask.astype('uint8'), mode="L")
                num_slash = data.loc[i, 'image'].count('/')
                image_file_name = "Results/all_images_mask/" + str.removesuffix(str.split(data.loc[i, 'image'], '/')[num_slash], ".jpg") + '__mask.png'
                image.save(image_file_name)