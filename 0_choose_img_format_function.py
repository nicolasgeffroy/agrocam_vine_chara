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

from pandas import DataFrame, to_datetime
# import numpy as np

def data_loading(img_path: str = "Core/Images/image_train/images", target_path: str = "Core/Results/Image_mask") -> "DataFrame":
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

import numpy as np
from sklearn.cluster import KMeans

def k_means_seg(img, c_num=4):
    """
    Perform K-means clustering segmentation on an image.

    This function applies K-means clustering to an input image, segmenting it into
    a specified number of clusters. The function returns a label map where each pixel
    is assigned to one of the clusters.

    Parameters
    ----------
    img : numpy.ndarray
        Input image as a numpy array which shape is (height, width, channels)
    c_num : int, optional
        Number of clusters for K-means segmentation. Default is 4.

    Returns
    -------
    numpy.ndarray
        A 2D array with shape (height, width) where each value represents 
        the cluster number assigned to the corresponding pixel.

    Notes
    -----
    - The function uses scikit-learn's KMeans implementation.
    - The random_state parameter is set to 0 for reproducibility.
    - n_init is set to 'auto' to automatically determine the number of initializations.
    - The returned label map has the same spatial dimensions as the input image.
    """

    ## Image Preparation
    # Get the height and width of the image.
    x, y = img.shape[0], img.shape[1]
    # Determine the number of channels.
    if len(img.shape) == 2:
        # Grayscale image (1 channel).
        z = 1
    else:
        # Color image (multiple channels).
        z = img.shape[2]
    # Reshape the image to a 2D array of pixels (each row is a pixel, columns are channels).
    # This format is required by scikit-learn's KMeans implementation.
    img_base = np.reshape(img, (-1, z))

    ## K-means Clustering
    # Initialize KMeans with specified number of clusters.
    clust = KMeans(n_clusters=c_num, random_state=0, n_init='auto')
    # Fit the KMeans model to the image data.
    res = clust.fit(img_base)
    # Get the cluster labels for each pixel.
    lab = res.labels_
    # Reshape the labels back to the original image dimensions.
    lab = lab.reshape(x, y)

    ## Return Results
    # Return the cluster label map.
    return lab

from tqdm import tqdm
from pandas import DataFrame
# import numpy as np
import copy

def quality_of_cluster_per_format(data, formats_used):
    """
    Evaluate clustering quality across different image formats by comparing K-means clusters with ground truth masks.

    This function processes a dataset of images in various color formats, performs K-means clustering,
    and compares the resulting clusters with ground truth masks.
    For the comparaison it first calculates the IoU score between each possible duo of cluster and classes and extract the best duo cluster and class.
    Then it calculates the mean IoU for each classes (and a mean IoU for each format) and a cluster uniqueness score for each format.


    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing image paths and mask paths with columns:
        - "image": Path to input images
        - "mask": Path to combined mask images (with values 1-4 representing different classes)
    formats_used : list or str
        Color formats to evaluate. Can be:
        - A string like "RGB" or "RGB-LAB" (hyphen-separated for multiple formats)
        - A list of strings like ["RGB"]
        - A list of hyphen-separated strings like ["RGB-LAB", "HSV"]

    Returns
    -------
        data_all_mask_format: Dictionary with detailed results for each format and each classes
            {format_name: {"IoU": {class: mean_IoU}, "clust": mean_clust_score}}
        data_mean_mask_format: Dictionary with mean results for each format
            {format_name: {"IoU": mean_IoU_across_classes, "clust": mean_clust_score}}

    Notes
    -----
    - Uses K-means with 4 clusters to match the 4 target classes (leaf, inter-row, sheath, trunk)
    - Cluster uniqueness score (clust_score) measures how many unique clusters are used:
        clust_score = (number of unique clusters used) / (total number of clusters)
    - Higher IoU values indicate better segmentation quality for this format
    - Higher clust_score values indicate less redundancy in cluster usage
    """

    ## Data Preparation
    # Create a deep copy of the input data to avoid modifying the original.
    data_new = copy.deepcopy(data)
    # Initialize dictionaries to store results.
    data_all_mask_format = {}  # Detailed results for each format
    data_mean_mask_format = {} # Mean results for each format

    ## Processing Each Format
    for format_config in formats_used:
        # Format Normalization
        # Normalize format_config to a list of formats.
        if isinstance(format_config, str):
            if "-" in format_config:
                current_formats = format_config.split("-")
            else:
                current_formats = [format_config]
        else:
            current_formats = format_config  # Assume it's already a list

        ## Results Storage Initialization
        # Prepare column names for IoU scores.
        col_iou = ['feuille', 'interrow', 'sheath', 'trunc']
        # Initialize dictionaries to store IoU scores for each class.
        data_inter_iou = {i: [] for i in col_iou}
        # Initialize list to store cluster uniqueness scores.
        data_inter_clust_score = []

        print("Calculating the ", format_config, " format.")

        ## Processing Each Image
        for i in tqdm(range(data_new.shape[0])):
            # Image and Mask Loading
            # Load the input image in the current format.
            img = load_image(data_new.loc[i, "image"], format=current_formats)
            # Load the target mask as a single-channel label image.
            target_mask_labels = load_image(data_new.loc[i, "all"], format=["L"])

            ## K-means Clustering
            # Set the number of clusters for K-means segmentation.
            c_num = 4
            # Perform K-means clustering on the image.
            k_means_labels = k_means_seg(img, c_num=c_num)

            ## IoU Calculation for Each Target Class
            # Initialize list to store assigned clusters for each target class.
            list_clust_per_image = []
            # Iterate through each target segment (values 1-4 in the mask).
            for idx in range(1, 5):
                # Create binary mask for the current target class.
                target_segment_binary = (target_mask_labels == idx).astype(np.uint8)
                
                # Compare with each K-means cluster.
                # Initialize list to store IoU scores for the current target class.
                current_iou_scores = []
                for k_cluster_idx in range(c_num):
                    # Create binary mask for the current K-means cluster.
                    k_cluster_binary = (k_means_labels == k_cluster_idx).astype(np.uint8)

                    # Calculate IoU between the target segment and K-means cluster.
                    current_iou_scores.append(IoU(k_cluster_binary, target_segment_binary, dim_z=1))

                ## Best Cluster Selection
                # Find the cluster with maximum IoU for the current target segment.
                if current_iou_scores:  # Ensure there are scores to prevent errors
                    clust_max = np.argmax(current_iou_scores)
                    iou_max = current_iou_scores[clust_max]
                else:
                    clust_max = -1  # Indicate no cluster found
                    iou_max = 0.0   # Default IoU
                
                ## Storing the results
                # Store the maximum IoU for the current target class.
                data_inter_iou[col_iou[idx-1]].append(iou_max)
                # Store the best cluster index for the current target class.
                list_clust_per_image.append(clust_max)

            ## Cluster uniqueness Calculation
            # Calculate how many unique clusters are used for this image.
            count_redundancy_clust = np.unique(list_clust_per_image)
            # Calculate cluster uniqueness score: (unique clusters used) / (total clusters)
            data_inter_clust_score.append(len(count_redundancy_clust) / c_num)

        ## Format Results Aggregation
        # Create a name for the current format combination.
        name_format = "-".join(current_formats)
        # Store results for the current format.
        data_all_mask_format[name_format] = {"IoU": {key: np.mean(val) for key, val in data_inter_iou.items()},
                                             "clust": np.mean(data_inter_clust_score)}
        # Store mean results for the current format.
        data_mean_mask_format[name_format] = {"IoU": np.mean(list(data_all_mask_format[name_format]["IoU"].values())),
                                              "clust": np.mean(data_inter_clust_score)}

    return data_all_mask_format, data_mean_mask_format


def choose_format(data):
    """
    Select the optimal image format based on cluster uniqueness and IoU metrics.

    This function evaluates different image formats based on their IoU (Intersection over Union)
    and cluster uniqueness scores, normalizes these scores, and selects the format with the
    highest combined score.

    Parameters
    ----------
    data : dict
        Dictionary containing evaluation metrics for each image format.
        Expected structure:
            {"format_name_1": {"IoU": float, "clust": float},
            "format_name_2": {"IoU": float, "clust": float},
            ...}
        where:
         - "IoU" is the mean Intersection over Union score across all classes
         - "clust" is the cluster uniqueness score (higher is better)

    Returns
    -------
    str
        The name of the image format with the highest combined score of IoU and cluster uniqueness metrics.

    Notes
    -----
    - Both IoU and cluster uniqueness scores are normalized to a 0-1 range using min-max normalization
    - The combined score is the sum of normalized IoU and cluster uniqueness scores
    - Higher IoU values indicate better segmentation quality
    - Higher cluster uniqueness scores indicate less redundancy in cluster usage
    """

    ## Metrics Extraction
    # Initialize lists to store metrics for each format.
    iou_values = []    # To store IoU scores
    clust_values = []  # To store cluster uniqueness scores
    format_values = [] # To store format names
    # Extract metrics from the input dictionary.
    for format_name in data:
        iou_values.append(data[format_name]["IoU"])
        clust_values.append(data[format_name]["clust"])
        format_values.append(format_name)

    ## Metrics Normalization
    # Normalize IoU values to a 0-1 range using min-max normalization.
    iou_values = [(j - min(iou_values)) / (max(iou_values) - min(iou_values)) for j in iou_values]
    # Normalize cluster uniqueness scores to a 0-1 range using min-max normalization.
    clust_values = [(j - min(clust_values)) / (max(clust_values) - min(clust_values)) for j in clust_values]

    ## Combined Score Calculation
    # Calculate a unified score by summing normalized IoU and cluster uniqueness scores.
    # This gives equal weight to both segmentation quality and cluster efficiency.
    unified_score = [iou_values[i] + clust_values[i] for i in range(len(iou_values))]

    ## Optimal Format Selection
    # Find the format with the maximum combined score.
    format_max_unified_score = format_values[np.argmax(unified_score)]

    return format_max_unified_score


if __name__ == "__main__":
    import argparse
    
    ## Ask the relevant arguments
    parser = argparse.ArgumentParser(description='Select the variables used for prediction')
    # Input the path to the csv file with the agronomic characteritics of ground-truth images.
    parser.add_argument('--folder_url_train_img', type=str, required=False, default="Core/Images/image_train/images",
                        help='URL of the folder containing all the images that will be used for segmentation training.')
    parser.add_argument('--folder_url_train_mask', type=str, required=False, default="Core/Images/image_train/masks",
                        help='URL of the folder containing all the mask that will be used for segmentation training.')
    # Input the list of format to test
    parser.add_argument('--string_for_list_format', type=str, required=False, default="[RGB,LAB,RGBA,HSV,RGB-LAB,RGB-HSV,LAB-HSV,RGB-LAB-HSV]",
                        help='String which represent the different format to test. The formats must be inputed between brakets [] and seperated with comas , ')

    # Store the parsed arguments
    args = parser.parse_args()
    
    ## Preparing the database and the format tested
    # Loading the database with the images used
    data = data_loading(img_path=args.folder_url_train_img,
                        target_path=args.folder_url_train_mask)
    # Loading the format to test
    format_strings = args.string_for_list_format
    format_list = str.split(format_strings, ",")
    format_list[0] = format_list[0][1:]
    format_list[-1] = format_list[-1][0:-1]
    
    ## Calculating the best format for segmentation
    _ ,all_mean_score = quality_of_cluster_per_format(data, formats_used=format_list)
    best_format = choose_format(all_mean_score)
    print("The best format is the ", best_format, " format")