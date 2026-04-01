# <p style="text-align:center;"> 🌳 Agronomic Characteristics Extraction and Prediction from Agrocam Vineyard Images 🌳 </p>

## 📘 Context

This project is the result of my end-of-year internship which consist in **extracting agronomic characteristics of vineyard images** taken with an open-source and low-cost camera system (named Agrocam) and **predicting them for a 15 day period**.

The extraction and prediction have been done on **images taken by an [Agrocam](https://agrocam.agrotic.org)**. We focused on camera installed on three type of vineyard :
- **[TVITI](https://agrocam.agrotic.org/data/79bt3wkh/)** => Plot managed sustainably by the winegrower and without ground cover
- **[AVITI](https://agrocam.agrotic.org/data/7s3a5abm/)** => Plot with an interrow vegetal cover composed of green fertilizer (sow by the winegrower)
- **[DVITI](https://agrocam.agrotic.org/data/4j7g2wk9/)** => Plot with an interrow vegetal cover composed of spontaneous vegetation (with no human intervention)

The **agronomic characteristics** extracted from the images and forecasted for 15 days are : 
- Canopy height
- Canopy porosity
- Leaf hue
- Inter-row hue.

## 📜 Table of Contents

1. [Pipeline Overview](##Pipeline%20Overview)
2. [Repository Organization](##Repository%20Organization)
3. [Installation and usage](##Installation%20and%20usage)
    1. [Initializing](###Initializing)
    2. [How to use it](###How%20to%20use%20it)
        1. [Format](###Format)
        1. [Segmentation](###Segmentation)
        2. [Extraction](###Extraction)
        3. [Selection](###Selection)
        4. [Prediction](###Prediction)
4. [Pretrained model weight](##Pretrained%20model%20weight)  
5. [Model Evaluation](##Model%20Evaluation)  
6. [How to contribute ?](##How%20to%20contribute%20?)  
7. [License](##License)  
8. [Citation](##Citation)

## ⚙️ Pipeline Overview

This pipeline aims at **extracting and predicting agronomic parameters from vineyard images in real condition**. For that, it has been seperated into 4 main steps :
- **Segmentation** of vineyard images into semantic zones (leaves, trunk, inter-row, irrigation sheath)
- **Extraction** of agronomics characteristics using those semantic zones
- **Selection** of those characteristics so that it's coherent with scientific knowledge
- **Prediction** of characteristics using temporal series of those vineyard images

## 🔎 Repository Organization

This pipeline is composed of 1 common block (*Core*) 4 *almost* self-sufficient block (*Segmentation*, *Extraction*, *Selection* and *Prediction*) :

| Folder | Content |
|:-:|:-:|
| [Core](https://github.com/nicolasgeffroy/agrocam_agro_chara/Core) | Contains an **example of an image database and intermediate results** which can be directly used by pipeline blocks. It is also where **output of each blocks as stored**. |
| [Format](https://github.com/nicolasgeffroy/agrocam_agro_chara/Format) | Contains all the functions which goal is to **determine the best image format to use for its [Segmentation](https://github.com/nicolasgeffroy/agrocam_agro_chara/Segmentation)** |
| [Segmentation](https://github.com/nicolasgeffroy/agrocam_agro_chara/Segmentation) | Contains all the function to summarize the **images into a dataset**, use them to **train a model (MobileNetv3 or DeepLabv3) for segmentation** and use this trained model. It also contains the function used to **determine the image format used for learning**. |
| [Extraction](https://github.com/nicolasgeffroy/agrocam_agro_chara/Extraction) | Contains all the functions which **uses the mask generated** by the [Segmentation](https://github.com/nicolasgeffroy/agrocam_agro_chara/Segmentation) (highlighting different ZOI) to **extract different agronomic characteristics** of images. |
| [Selection](https://github.com/nicolasgeffroy/agrocam_agro_chara/Selection) | Contains all the function which **uses the [extracted](https://github.com/nicolasgeffroy/agrocam_agro_chara/Extraction) agronomic characteristics** of each images to select the characteristics which **best represent agronomic reality**. |
| [Prediction](https://github.com/nicolasgeffroy/agrocam_agro_chara/Prediction) | Contains all the function which **trains a model (LSTM or CNN-LSTM hybrid) to predict**, using temporal series of vineyards images, **vineyard's futur characteristics** as well as function using the trained model. |

<details><summary><b> Diagram </b></summary>

```graphql
agrocam_agro_chara/
├── Core/
│   ├── Images/
│   |   ├── all_image
|   |   └── image_train
│   ├── Results/
|   |   ├── Image_mask/
|   |   ├── Agro_chara_vine.csv
|   |   ├── Agro_chara_vine_train.csv
|   |   ├── Image_chara_all.csv
|   |   └── Image_chara_train.csv
│   └── README.md
├── Format/
│   ├── choose_img_format_function.py
│   ├── requirements.txt
│   └── README.md
├── Segmentation/
│   ├── format_choice/
│   │   ├── Image_chara_all.csv
│   │   └── k_means_seg.ipynb
│   ├── Liste_Agrocam.csv
│   ├── segmentation_function.py
│   ├── requirements.txt
│   └── README.md
├── Extraction/
│   ├── extraction_function.py
│   ├── requirements.txt
│   └── README.md
├── Selection/
│   ├── segmentation_function.py
│   ├── requirements.txt
│   └── README.md
├── Prediction/
│   ├── model/
│   |   ├── cnn_lstm.py
│   |   └── mobilenet_LRASPP.py
│   ├── prediction_function.py
│   ├── requirements.txt
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md
```

</details>

## 💻 Installation and usage

### 1️⃣ Initializing

<details><summary><b> 1. Clone the repository </b></summary>

To get this repository on your computer (to then use it), you can use those lines of codes with either the URL of this repository...

```bash
git clone https://github.com/nicolasgeffroy/agrocam_agro_chara
cd agrocam_agro_chara
```

or the SHH key (after [connecting to GitHub with SSH](https://decodementor.medium.com/connect-git-to-github-using-ssh-68ab338f4523)). 

```bash
git clone git@github.com:nicolasgeffroy/agrocam_agro_chara.git
cd agrocam_agro_chara
```

</details>

<details><summary><b> 2. Create a virtual environment (and use it) </b></summary>

Creating a virtual environment ensures that there is no dependency problem with other packages (downloaded for other project) and helps with reproductibility (keeps the package in a state where all the repository worked).

```bash
bash python -m venv .venv 
.venv/bin/activate # On Windows: venv\Scripts\activate
```

Alternatively you can use [uv](https://github.com/astral-sh/uv) :

```bash
uv venv --python 3.11.9
```

</details>

<details><summary><b> 3. Download the required package for this repository </b></summary>

After creating it, we download all the required package for this repository (in the "requirements.txt") in this environment.

```bash
python -m pip install -r requirements.txt
```

Alternatively with [uv](https://github.com/astral-sh/uv), you can add all the requiered package with :

```bash
uv add -r requirements.txt
```

**All the package installed:**

| Package  | Version | Keywords                                         | For information                                                       |
| :-:      | :-:     | :-:                                              | :-:                                                                   |
| pillow | 12.0 | Image processing                                     | [link](https://pillow.readthedocs.io/en/stable)                       |
| requests | 2.32.5 | Make QGIS like request on the internet             | [link](https://requests.readthedocs.io/en/latest)                     |
| numpy | 2.3.4 | Adding Array format and function to exploit them      | [link](https://numpy.org)                                             |
| pandas | 2.3.3 | Adding DataFrame format and function to exploit them | [link](https://pandas.pydata.org/docs/getting_started/overview.html)  |
| tqdm | 4.67.1 | Adds a progress bar to loops                           | [link](https://tqdm.github.io)                                        |
| torch | 2.9 | Base for Deep Learning application                    | [link](https://docs.pytorch.org/docs/stable/index.html)               |
| torchvision | 0.24.0 | Image Deep Learning framework                  | [link](https://docs.pytorch.org/vision/stable/index.html)             |
| scikit-learn | 1.7.2 | Adds various model and Preprocessing tools     | [link](https://scikit-learn.org/stable)                               |
| tensorboard | 2.20 | Adds a vizualisation kit for machine learning   | [link](https://www.tensorflow.org/tensorboard?hl=fr)                  |
| matplotlib | 3.10.7 | Adds various plot to vizualise any data          | [link](https://matplotlib.org/cheatsheets)                            |

</details>

PS : All those packages have been used under the version **3.11.9** of Python.

**Summary**

```bash
# 1. Clone the repository
git clone https://github.com/nicolasgeffroy/agrocam_agro_chara
# git clone git@github.com:nicolasgeffroy/agrocam_agro_chara.git
cd agrocam_agro_chara
# 2. Create a virtual environment (and use it)
bash python -m venv .venv 
.venv/bin/activate # On Windows: venv\Scripts\activate
# 3. Download the required package for this repository
python -m pip install -r requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv) :

```bash
# 1. Clone the repository
git clone https://github.com/nicolasgeffroy/agrocam_agro_chara
# git clone git@github.com:nicolasgeffroy/agrocam_agro_chara.git
cd agrocam_agro_chara
# 2. Create a virtual environment (and use it)
uv venv --python 3.11.9
# 3. Download the required package for this repository
uv add -r requirements.txt
```

### 2️⃣ How to use it

**Each blocks can be called using this following example of line of code :**

```bash
python <name_folder>_function.py --<argument> ...
```

---

Just below, is displayed, for each block :
- **Purpose** => What do they do when called ?
- *Input =* Which default input do they use ?
- *Output =* Which output do they provide ?

| Arguments | description | Input |
| :-: |:-:| :-:|
|  --\<argument> | Description of the argument | Which input does it take |

<details> <summary><b> Examples of lines of code to use. </b></summary> 

```bash
python <name_folder>/<name_folder>_function.py --<argument> ...
```
==> Description of what does this line of code

</details>

---

### Format

- **Purpose** = Using the k-means algorithm, the block approximate the efficacity of the image segmentation for each image format and proposes a format to use for the segmentation block.
- *Input =* 
    - The database representing each image and their ground-truth mask that will be used for the training of the segmentation model  (*Core/Results/Image_chara_train.csv*).
- *Output =* 
    - A string displaying the best format to use to segment images.

| Arguments | description | Input | Default |
| :-: | :-: | :-: | :-: |
|  --\<folder_url_train_img> | URL to the folder containing the images used for segmentation training  | string | "Core/Images/image_train/images" |
|  --\<folder_url_train_mask> | URL to the folder containing the mask used for segmentation training  | string | "Core/Images/image_train/masks" |
|  --\<string_for_list_format> | String representing the list of format to test | string | "[RGB,LAB,RGBA,HSV,RGB-LAB,RGB-HSV,LAB-HSV,RGB-LAB-HSV]" |

<details> <summary><b> Examples </b></summary>

```bash
python Format/choose_img_format_function.py 
```
==> Determine the best image format to use for the image segmentation model between RGB, LAB, HSV and some of their combinaison (RGB-LAB,RGB-HSV,LAB-HSV,RGB-LAB-HSV) using images (*Core/Images/image_train/images*) and their mask (*Core/Images/image_train/masks*) used for segmentation.

```bash
python Segmentation/segmentation_function.py 
    --string_for_list_format "[RGB,LAB,RGBA,HSV]"
```
==> Determine the best image format to use for the image segmentation model between RGB, LAB, HSV using images (*Core/Images/image_train/images*) and their mask (*Core/Images/image_train/masks*) used for segmentation.

</details>

### Segmentation

- **Purpose** = Trains a MobileNetv3 model to segment an image into 4 zone of interest : leaf, inter-row, trunc and sheath (**train**) or use a trained model to segment a set of images (**segment**).
- *Input =* 
    - (**train**) Image and ground truth mask chosen for training (*Core/image_train*) 
    - (**segment**) A set of vineyard images (*Core/all_image*) 
- *Output =* 
    - A database representing all the images inputed (*Core/Results/Image_chara_train.csv* for **train** or *Core/Results/Image_chara_all.csv* for **segment**)
    - (**train**) Weight of the trained model (*Segmentation/checkpoint*)
    - (**segment**) A mask per images (with all the classes) inputed (*Core/Results/Image_mask*)

| Arguments | description | Input | Default |
| :-: | :-: | :-: | :-: |
|  --\<folder_url_train_img> | URL of the folder containing input images | string | "Core/Images/image_train/image" |
|  --\<train_or_segment> | Choose to train an algorithm or use a trained one to segment images | "train" or "segment" | "segment" |
|  --\<folder_url_train_mask> | **training** = Ground-truth mask // **segment** = URL of the folder containing mask for calcultating the distance between trunc and sheath | string | "Core/Images/ image_train/masque_final" |
|  --\<weight_url> | Import weight of the model. If "No_weight" used, [pretrained weights for MobileNetV3](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#torchvision.models.segmentation.lraspp_mobilenet_v3_large) are used | string | "No_weight" |
|  --\<format_used> | Which image format is used (**train**) to train the model or (**segment**) to generate all the associated mask | string | "HSV"
|  --\<saving> | **train** Decides if the trained weights are saved | bool | True |
|  --\<epochs> | **train** Number of epochs for training | int | 10 |

<details> <summary><b> Examples </b></summary>

```bash
python Segmentation/segmentation_function.py 
    --weight_url Segmentation/checkpoint/mobileNetv3_checkpoint_focal_HSV.pth 
    --folder_url_train_img Core/Images/all_image
```
==> **Generate for each images** in *Core/Images/all_image* the **segmentation masks** (in *Core/Results/Image_mask*) using a **MobileNetv3 model trained** with the specified weight (*mobileNetv3_checkpoint_focal_HSV.pth*). It also generate a database (*Core/Results/Image_chara_all.csv*) with all the images used and information about them.

```bash
python Segmentation/segmentation_function.py 
    --train_or_segment train 
    --epochs 100
    --saving False
    --format_used "HSV-LAB"
```
==> **Train a pretrained MobileNetv3 model** for 100 epocks using training images (*Core/Images/image_train/image*), represented using the combinaison of "HSV" and "LAB" format, and their ground truth mask (*Core/Images/image_train/masque_final*) without saving its weights. It also generate a database (*Core/Results/Image_chara_train.csv*) with all the images used and information about them.

</details>

### Extraction
- **Purpose** = Extracts the different agronomic characteristics from each vineyard images using the different mask generated by the [segmentation](#segmentation) block.
- *Input =* The database representing each images inputed during [segmentation](#segmentation) (*Core/Results/Image_chara_all.csv*).
- *Output =* A database with, for each image, all the vineyard's agronomic characteristics (*Core/Results/Agro_chara_vine.csv*)

| Arguments | description | Input | Default |
| :-: | :-: | :-: | :-: |
|  --\<folder_url_all_img> | URL to the folder that containing all the images we want to extract agronomic characteristics  | string | "Core/Images/all_image" |
|  --\<folder_url_all_mask> | URL to the folder that containing all the mask generated during segmentation  | string | "Core/Results/Image_mask" |
|  --\<folder_url_truth_mask> | URL to the folder containing the mask used for segmentation training. Used for correcting porosity characteristics when using sheath. | string | "Core/Images/image_train/masks" |
|  --\<name_of_mask_used> | Name of the entity used to determine the upper part of the image (used for the porosity characteristics) | "trunc" or "sheath" | "sheath" |
|  --\<path_saving> | Path to which the database with all the agronomic characteristics will be saved  | string | "Core/Results/Agro_chara_vine.csv" |

<details> <summary><b> Examples </b></summary>

```bash
python Extraction\extraction_function.py
```
==> Generate a database (saved in *Core/Results/Agro_chara_vine.csv*) with for each image in *Core/Images/all_image* we have their agronomic parameters using their associated mask in *Core/Results/Image_mask*. To determine the porosity, it uses the sheath mask to determine the upper zone of the image and correct it using the masks in *Core/Images/image_train/masks*.

```bash
python Extraction\extraction_function.py 
    --folder_url_all_img "Core/Results/Image_chara_train.csv"
    --name_of_mask_used "trunc"
    --path_saving "Core/Results/Agro_chara_vine_train.csv"
```
==> Generate a database (saved in *Core/Results/Agro_chara_vine_train.csv*) where for each image in *Core/Images/image_train/images* we have their agronomic parameters using their associated mask in *Core/Results/Image_mask*. To determine the porosity, it uses the trunc mask to determine the upper zone of the image and no correcting is needed.
</details>

### Selection 

- **Purpose** = Select the different agronomic characteristics that will be used for [prediction](#prediction).
- *Input =* Two databases with one representing each images and their characteristics determined during [extraction](#extraction) (*Core/Results/Image_chara_all.csv*) and the other others the images used in training with their characteristics determined with their ground truth mask (*Core/Results/Image_chara_train.csv*) .
- *Output =* A message with the characteristics selected and, if save is set to True, those characteristics are written in the file *parameters.txt* located in the [prediction](#prediction) block.

| Arguments | description | Input | Default |
| :-: | :-: | :-: | :-: |
|  --\<agro_chara_all> | URL to the csv file with the agronomic characteritics of all the images where the variable are selected | string | "Core/Results/Agro_chara_vine.csv" |
|  --\<agro_chara_train> | URL to the csv file with the agronomic characteritics of the trained images used for selecting variables | string | "Core/Results/Agro_chara_vine_train.csv" |
|  --\<dist_func> | Name of the function which calculate the distance used to compare each time series | string | "dist_manathan" |
|  --\<save> | If True, saves the results of the function in the "parameters.txt" file for the prediction | bool | False |

<details> <summary><b> Examples </b></summary>

```bash
python Selection\selection_function.py
```
==> Prints the variables in *Core/Results/Agro_chara_vine.csv* that must be used for prediction in order to make sure all the treatment can be distinguished and that the values of each treatment in *Core/Results/Agro_chara_vine_train.csv* are close to *Core/Results/Agro_chara_vine.csv*.

</details>

### Prediction 

- **Purpose** => Trains a LSTM or CNN-LSTM model to predict, using 15 last vineyard images, the next 15 days of vineyard agronomic characteristics : canopy porosity and height as well as leaf and interrow hue (**train**) or use a trained model to predict vineyard agronomic characteristics (**predict**).
- *Input =* Database with images of which we have extracted the vineyards characteristics during [extraction](#extraction) (*Core/Results/Agro_chara_vine.csv*) 
- *Output =* 
    - (**train**) Weight of the trained model (*Prediction/checkpoint*)
    - (**predict**) Prints the characteristics of the vineyard for the next 15 days.

| Arguments | description | Input | Default |
| :-: | :-: | :-: | :-: |
|  --\<lstm_model> | Class (in the designated package) of the LSTM model used as a prediction model | string | "model.cnn_lstm.CNN_LSTM" |
|  --\<weight_url_cnn> | Import weight of the cnn model used when a CNN-LSTM model is used. If given "No_weight", [pretrained weights for MobileNetV3](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#torchvision.models.segmentation.lraspp_mobilenet_v3_large) are used. | string | "No_weight" |
|  --\<weight_url_lstm> | Import weight of the lstm model. | string | "Prediction/checkpoint/MobileNet3_LSTM _checkpoint_final_hsv_notbi_norm.pth" |
|  --\<train_or_predict> | Choose to train an algorithm or use it to predict agronomic characteristics | "train" or "predict" | "predict" |
|  --\<time_start> | **predict** Start date of the 15 image used for prediction | string | "2024-04-20" |
|  --\<treatment> | Treatment of vine where the images were taken | string | "AVITI" |
|  --\<epochs> | **train** Number of epochs for training | int | 10 |

<details> <summary><b> Examples </b></summary>

```bash
python Prediction/prediction_function.py 
    --lstm_model model.cnn_lstm.CNN_LSTM 
    --weight_url_cnn Segmentation/checkpoint/mobileNetv3_checkpoint_focal_HSV.pth
    --weight_url_lstm Prediction/checkpoint/MobileNet3_LSTM_checkpoint_final_hsv_notbi_norm.pth
```
==> Using the 15 images described in *Core/Results/Agro_chara_vine.csv* that are AVITI vineyards and with the first taken the 2024-04-20, it prints the next vineyard characteristics (lasting from 2024-05-06 to 2024-05-20) predicted by a CNN-LSTM model with the cnn weight ("mobileNetv3_checkpoint_focal_HSV.pth") and the lstm weight ("MobileNet3_LSTM_checkpoint_final_hsv_notbi_norm.pth") given.

```bash
python Prediction/prediction_function.py 
    --lstm_model model.cnn_lstm.no_CNN_LSTM 
    --weight_url_lstm Prediction/checkpoint/MobileNet3_LSTM_test_nocnn_checkpoint.pth
```
==> Using the 15 images described in *Core/Results/Agro_chara_vine.csv* that are the AVITI vineyards and with the first taken the 2024-04-20, it prints the next vineyard characteristics (lasting from 2024-05-06 to 2024-05-20) predicted by a LSTM model with the lstm weight ("MobileNet3_LSTM_test_nocnn_checkpoint.pth") given.

```bash
python Prediction/prediction_function.py 
    --lstm_model model.cnn_lstm.CNN_LSTM 
    --train_or_predict train
    --epochs 100
```
==> Train a CNN-LSTM model (with a CNN pretrained to COCO) to predict the next 15 vineyard characteristics with 15 prior vineyard images retrieved in *Core/Results/Agro_chara_vine.csv* for 100 epochs.

</details>

## 🎁 Pretrained model weight

You can contact me at nico.geffroy.pro@gmail.com for the checkpoint file (.pth) with weight of the model trained for segmentation and the one trained for prediction. 

## ⭐ Model Evaluation

<center>

| **Segmentation** | MobileNetv3 | Deeplabv3 | \|\|\|\| | MobileNetv3 (details) | Leaf | Interrow | Sheath | Trunc
| :-: |:-:| :-:| -------- | :-: |:-:| :-:| :-: |:-:|
|  IoU             | 0.72        |   0.53    | \|\|\|\| | ==> | 0.87 | 0.92 | 0.42 | 0.58
| Sensibility      | 0.82        |   0.63    | \|\|\|\| | ==> | 0.95 | 0.96 | 0.60 | 0.75
| Specificity      | 0.98        |   0.97    | \|\|\|\| | ==> |  0.97 | 0.98 | 1.00 | 1.00

| **Prediction** | ES-LSTM | PE-LSTM | (ES-EP)-LSTM | LSTM |
| :-: |:-:| :-:| :-: |:-:|
|  MSE           | 0.27    | 0.07    | 0.06         | 0.07 |

</center>

<p style="text-align:center;">
(ES = MobileNetV3 entraîné sur la segmentation, EP = MobileNetV3 entraîné sur la
prédiction, PE = MobileNetV3 seulement pré-entraîné avec COCO)
</p>

## 💪 How to contribute ?

You can find what's can/have to be done for this repository (you can also check out the [**Issues**](https://github.com/nicolasgeffroy/agrocam_agro_chara/issues) tab) : 

| Task        | Details           |
| :-: |:-:|
| Adding a nextflow file      |  |
| Configure a DockerFile      | Create a container with all the package and the python version and store it in Docker |
| Intermediate README files      | Adds all the README in the different blocks to detail what does each block (and their functions)  |
| Change where the images are stored and used      | For now, all the images used for this repository are stored in the same repository which can lead to difficulties when pushing or cloning   |

Make sure the stick as much as possible to the style in which the repository has been written.

Feel also free to signal bugs in the [**Issues**](https://github.com/nicolasgeffroy/agrocam_agro_chara/issues) tab. You can also highlight something you don't understand (sorry in advance for syntax errors 😅) or where I made a mistake or other issues linked to the code or its documentation in the [**Discussion**](https://github.com/nicolasgeffroy/agrocam_agro_chara/discussions) tab.

## 📄 License

Released under the **MIT License**.  
You are free to use, modify, and distribute this project with attribution.

## 🌱 Citation

If you use this repository in your research or publication, please cite:

```latex
@misc{NicoGeff2025,
  author = {Nicolas, Geffroy},
  title = {Agronomic Characteristics Extraction and Prediction from Agrocam Vineyard Images},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nicolasgeffroy/agrocam_agro_chara}}
}
```