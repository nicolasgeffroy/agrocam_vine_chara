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

from pandas import DataFrame, to_datetime
import numpy as np

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
        all = target_path + "/" + i_remove + "__all.png"
        l_add = [times[1], times[2], cond, img, all]
        img_data.append(l_add)

    # Create DataFrame with columns for training data.
    img_data = DataFrame(
        img_data,
        columns=["day", "time", "treatment", "image", 'all']
        )
    
    # Combine day and time into a single datetime column.
    img_data["day_time"] = img_data["day"] + " " + img_data["time"]
    img_data['day_time'] = to_datetime(img_data['day_time'], format="%Y-%m-%d %H%M%S")
    # Drop the separate day and time columns.
    img_data = img_data.drop(columns=["day", "time"])

    return img_data

## Different function used to extract different agronomic characteristics from the vineyard

# import numpy as np

def height_para(img: np.ndarray) -> float:
    """
    Calculate the normalized height of the canopy vineyard in the image.

    This function determines the vertical span of non-zero pixels in the canopy mask,
    representing the height of the canopy.
    The result is then normalized by dividing by the total image height (1080 pixels).

    Parameters
    ----------
    img : numpy.ndarray
        Input canopy mask or image as either a 3D or a 2D numpy array.

    Returns
    -------
    float
        Normalized height of the canopy (height / 1080).
        Returns `numpy.nan` if the image contains only zero pixels.
    """
    # Summing pixel values across width (and channels)
    if len(img.shape) == 3:
        # Sum pixel values along the width (axis=1) and then across channels (axis=1) if the input is a 3D array.
        sum_RGB = np.sum(img, axis=1)
        sum_RGB = sum_RGB.sum(axis=1)
    elif len(img.shape) == 2:
        # If the input is a 2D array, sum pixel values along the width (axis=1)
        sum_RGB = np.sum(img, axis=1)
    else:
        print("Incorrect dimension")

    # Handling edge case: all-zero image
    # If the sum of all pixel values is zero, return NaN (no canopy detected).
    if sum_RGB.sum() == 0:
        return np.nan

    # Finding the top of the canopy
    # Iterate from the bottom to the top of the image to find the highest non-zero row.
    for i in range(len(sum_RGB)):
        if sum_RGB[i] != 0:
            high = i  # Row index of the top of the canopy
            break
    
    # Finding the bottom of the canopy
    # Iterate from the top to the bottom of the image to find the lowest non-zero row.
    for i in reversed(range(len(sum_RGB))):
        if sum_RGB[i] != 0:
            low = i  # Row index of the bottom of the canopy
            break

    # Calculating and returning normalized height
    # Calculate the height as the difference between the top and bottom width 
    # normalized by the total image width (1080 pixels).
    return round(-(high - low) / 1080, 3)

# import numpy as np

def correcting_porosity_para(all_truth_mask_path: str) -> dict:
    """
    Calculate correction values for porosity measurements based on ground truth masks.

    This function processes ground truth masks to determine correction values for porosity
    calculations. It identifies the top of the trunk and the bottom of the canopy for each
    image and calculates the distance between them. These distances are then averaged per
    treatment to provide correction values used in porosity calculations.

    Parameters
    ----------
    all_truth_mask_path : str
        Path to the directory containing ground truth masks.

    Returns
    -------
    dict
        Dictionary with correction values for each treatment (TVITI, AVITI, DVITI).
        Returns NaN for treatments where the trunk top cannot be found.
    """
    mask_name_list = os.listdir(all_truth_mask_path + "/")
    correcting_values_per_treatment = {"TVITI" : [], "AVITI" : [], "DVITI" : []}
    for mask_name in mask_name_list:
        mask_url = all_truth_mask_path + "/" + mask_name
        treatment_trad = {"79bt3wkh" : "TVITI", "7s3a5abm" : "AVITI", "4j7g2wk9" : "DVITI"}
        mask_treatment = treatment_trad[mask_name.split("_")[0]]
        if os.path.exists(all_truth_mask_path):
            # Load the image with all the mask
            all_mask_img = load_image(mask_url, mode="L")
            # Extract the sheath and trunc (with the respective index 3 and 4)
            sheath_img = (all_mask_img == 3) * 1
            trunc_img = (all_mask_img == 4) * 1
            
            # Sum the values of pixels along the width to locate the width where the mask is.
            sum_sheath = np.sum(sheath_img, axis=1).reshape(1080)
            sum_trunc = np.sum(trunc_img, axis=1).reshape(1080)
            # Initialize variables to track the locations.
            sup_iter = len(sum_sheath) - 1
            one_iter_sup, one_iter_trunc = False, False
            for k in range(len(sum_sheath)):
                # Find the top of the trunk (first non-zero row encontered when coming from the top).
                if (sum_trunc[k] != 0) & (not one_iter_trunc):
                    trunc_loc = k
                    one_iter_trunc = True
                # Find the bottom of the sheath (first non-zero row encontered when coming from the bottom).
                if (sum_sheath[sup_iter] != 0) & (not one_iter_sup):
                    sheath_loc = sup_iter
                    one_iter_sup = True
                sup_iter = sup_iter - 1
                # Exit the loop once both locations are found.
                if one_iter_sup & one_iter_trunc:
                    break
            # If the trunk top is not found, set distance to NaN.
            if trunc_loc == 0:
                correcting_values_per_treatment[mask_treatment] = np.nan
            else:
                correcting_values_per_treatment[mask_treatment] = trunc_loc - sheath_loc
        else:
            print("No path found")
    return {"TVITI" : np.mean(correcting_values_per_treatment["TVITI"]), 
            "AVITI" : np.mean(correcting_values_per_treatment["AVITI"]), 
            "DVITI" : np.mean(correcting_values_per_treatment["DVITI"])}

def porosity_para(img_zone: np.ndarray, img_enti: np.ndarray, type_entity: str = "sheath", corr: int = 50) -> float:
    """
    Calculate the porosity of a plant canopy zone relative to the entire plant area.

    This function computes the porosity by substracting the area (number of pixel) of the canopy 
    by the area of the upper part of the image (determining using the trunc or the sheath). 
    It is then normalized by the area of the upper part of the image.

    Parameters
    ----------
    img_zone : numpy.ndarray
        3D or 2D array representing the trunc and sheath.
    img_enti : numpy.ndarray
        3D or 2D array representing the canopy.
    type_entity : str
        Informing the type of mask the `img_zone` represent.
        Either the sheath mask ("sheath") or the trunk mask ("trunk").
    corr : int, optional
        Correction factor to adjust the lower boundary of the zone. Default is 50.

    Returns
    -------
    float
        Porosity value (ratio of empty space to total zone area).
        Returns `numpy.nan` if either input image is all zeros.
    """
    
    # Summing pixel values across width (and channels)
    if len(img_zone.shape) == 3:
        # Sum pixel values along the width (axis=1) and then across channels (axis=1) if the input is a 3D array.
        sum_RGB = np.sum(img_zone, axis=1)
        sum_RGB = sum_RGB.sum(axis=1)
    elif len(img_zone.shape) == 2:
        # If the input is a 2D array, sum pixel values along the width (axis=1)
        sum_RGB = np.sum(img_zone, axis=1)
    else:
        print("Incorrect dimension")

    # Handling edge case: all-zero image
    # If either image is all zeros, return NaN (no data to process).
    if img_zone.sum() == 0 or img_enti.sum() == 0:
        return np.nan

    # Determining the mask of the upper part of the image using either the trunk or sheath mask.
    # Find the bottom of the upper part of the image using the sheath mask or the trunk mask.
    if type_entity == "sheath":
        # When using the sheath mask, we use the first non-zero row (for the bottom of the sheath).
        for i in reversed(range(len(sum_RGB))):
            if sum_RGB[i] != 0:
                low_zone = i
                break
        # Adjust the lower boundary of the zone using the correction factor.
        low_zone = low_zone + corr
    elif type_entity == "trunk":
        # When using the trunk mask, we use the first non-zero row (for the top of the trunk).
        for i in range(len(sum_RGB)):
            if sum_RGB[i] != 0:
                low_zone = i
                break
    else:
        print("Wrong entity name")
    # Creating a mask for the upper zone by taking its bottom (determined earlier) and all the pixel to the top of the image.
    # Initialize a zero array for the zone mask.
    img_zone_plus = np.zeros((1080, 1920, 3))
    # Fill the zone mask up to the adjusted lower boundary.
    for i in range(1080):
        for j in range(1920):
            # Set pixels in the upper zone to 1.
            if i < low_zone:
                img_zone_plus[i, j, :] = 1
            # Binarize the entire plant image (set non-zero pixels to 1).
            if img_enti[i, j, :].sum() != 0:
                img_enti[i, j, :] = 1

    # Calculating porosity
    # Subtract the binarized canopy mask from the upper zone mask to find empty spaces.
    img_z_e = img_zone_plus - img_enti
    # Calculate porosity as the ratio of empty space to total upper zone area.
    return round(img_z_e.sum() / img_zone_plus.sum(), 3)

# import numpy as np

def hue_para(ori_img: np.ndarray, img: np.ndarray) -> float:
    """
    Calculate the average hue channel intensity in order to caracterize the leaf color in the canopy 
    (if the canopy mask is used) and the state of the interrow (if the interrow mask is used).

    This function filters out black pixels (where R=G=B=0) and computes the mean intensity
    of the hue channel for the remaining pixels.

    Parameters
    ----------
    ori_img : numpy.ndarray
        Original image used to filter out black pixels.
    img : numpy.ndarray
        Input HSV image as a 3D numpy array (height × width × channels).

    Returns
    -------
    float
        Mean intensity of the hue channel for non-black pixels, rounded to 2 decimal places.
        Returns `numpy.nan` if the image contains only black pixels.
    """

    # Filtering out black pixels
    img = ori_img * img
    # Create a boolean mask where True indicates non-black pixels (R, G, and B all non-zero).
    filter = np.sum(img != [0, 0, 0], axis=2) == 3
    # Apply the filter.
    img = img[filter]
    # If no non-black pixels are found, return NaN.
    if filter.sum() == 0:
        return np.nan

    # Calculating mean hue intensity
    # Extract the hue channel (first channel in HSV format in PIL formating).
    img_hue = [img[i][0] for i in range(img.shape[0])]
    # Calculate the mean hue intensity across all non-black pixels.
    img_hue = np.mean(img_hue)
    # Round the result to 2 decimals.
    return round(img_hue, 2)

if __name__ == "__main__":
    import argparse
    ## Ask the relevant arguments
    parser = argparse.ArgumentParser(description='Train or Use a segmentation model on a set of images.')
    
    parser.add_argument('--folder_url_all_img', type=str, required=False, default="Core/Images/all_image",
                        help='URL of the folder containing all the images we want to extract their caracteristics.')
    parser.add_argument('--folder_url_all_mask', type=str, required=False, default="Core/Results/Image_mask",
                        help='URL of the folder containing all the mask used for caracteristics extraction.')
    
    parser.add_argument('--folder_url_truth_mask', type=str, required=False, default="Core/Images/image_train/masks",
                        help='URL of the folder containing the ground truth mask. Used to correct the porosity measure.')
    
    parser.add_argument('--path_saving', type=str, required=False, 
                        help='Path to which the database will be saved', 
                        default="Core/Results/Agro_chara_vine.csv")
    
    parser.add_argument('--name_of_mask_used', type=str, required=False, 
                        help='Name of the entity used to determine the upper part of the image', 
                        default="sheath")
    
    args = parser.parse_args()
    
    img_path = args.folder_url_all_img
    target_path = args.folder_url_all_mask
    all_truth_mask_path = args.folder_url_truth_mask
    save_path = args.path_saving
    type_entity = args.name_of_mask_used
    
    import copy
    from pandas import DataFrame, to_datetime
    from tqdm import tqdm
    
    ## Loading and preparing data
    data = data_loading(img_path, target_path)
    # Initialize a DataFrame to store extracted parameters.
    data['day_time'] = to_datetime(data['day_time'], format="%Y-%m-%d %H:%M:%S")
    para_agro = DataFrame({
        "image": data["image"],
        "time": data["day_time"].dt.date,
        "treatment": data["treatment"]
    })
    # Initialize columns for agronomic parameters.
    para_agro["H_vigne"], para_agro["P_vigne"], para_agro["Hue_vigne"], para_agro["Hue_rang"] = 0, 0, 0, 0
    
    corr_per_treatment = correcting_porosity_para(all_truth_mask_path)
    print(corr_per_treatment)

    ## Extracting parameters for each image
    for i in tqdm(range(data.shape[0])):
        
        all_target = load_image(data.loc[i, "all"], mode=["L"])
        treatment_trad = {"79bt3wkh" : "TVITI", "7s3a5abm" : "AVITI", "4j7g2wk9" : "DVITI"}
        
        mask_treatment = treatment_trad[data.loc[i, "all"].split("/")[-1].split("_")[0]]
        
        label = ["bck", "feuille", "inter", "sheath", "trunk"]
        all_classes = {i : 0 for i in label if i != "bck"}
        for o in range(1,len(label)):
            # Extract the predicted and target binary mask of the selected class
            mask = (all_target == o) * 1.0
            mask = np.expand_dims(mask, 2)
            all_classes[label[o]] = mask
        
        # Load original image in HSV color space used for hue calculation.
        ori_img = load_image(data.loc[i, "image"], mode=["HSV"])
        feuille_2 = copy.deepcopy(all_classes["feuille"])  # Create a copy for hue analysis
        
        ## Calculating agronomic parameters
        # Calculate vine height (normalized).
        para_agro.loc[i, "H_vigne"] = height_para(all_classes["feuille"])
        # Calculate vine porosity using the sheath mask or the trunk mask.
        para_agro.loc[i, "P_vigne"] = porosity_para(all_classes[type_entity], all_classes["feuille"], corr=corr_per_treatment[mask_treatment])
        # Calculate inter-row mean hue.
        para_agro.loc[i, "Hue_rang"] = hue_para(ori_img, all_classes["inter"])
        # Calculate vine mean hue.
        para_agro.loc[i, "Hue_vigne"] = hue_para(ori_img, feuille_2)

        ## Periodically saving results
        # Save results to CSV every 50 iterations and at the end of processing.
        if ((i % 10 == 0) or (i == data.shape[0] - 1)):
            para_agro.to_csv(save_path, index=False)