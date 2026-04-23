import os
import PIL.Image
import base64
import requests
from io import BytesIO
from typing import Optional, Union
import numpy as np
import pandas as pd

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

def data_loading(img_path: str , target_path: str ) -> "pd.DataFrame":
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
    img_data = pd.DataFrame(
        img_data,
        columns=["day", "time", "treatment", "image", 'mask']
        )
    
    # Combine day and time into a single datetime column.
    img_data["day_time"] = img_data["day"] + " " + img_data["time"]
    img_data['day_time'] = pd.to_datetime(img_data['day_time'], format="%Y-%m-%d %H%M%S")
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
