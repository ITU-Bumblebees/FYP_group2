# First year project 2 - imaging

This repository contains some starting data and code for the first year project imaging. All other information is provided via LearnIT. 


# Data


The ISIC 2017 dataset is available via https://challenge.isic-archive.com/landing/2017. There are more than 2000 images available in this dataset. 
In this repository, only 150 images from this dataset are added, as an example. The following is available per image:

*	`ISIC_[ID].png` the image of the lesion
*	`ISIC_[ID]\_segmentation.png` the mask of the lesion, showing which pixels belong to the lesion or not
* The label of the image, i.e. whether it belongs to the Melanoma class (0 = no, 1 = yes), and/or the Keratosis class (0 = no, 1 = yes). 

You can also use additional/external data for the later tasks in this project.


## Code

In this project you will work with Python. Some starting code is already provided:

* `groupXY_functions.py` with functions to extract features etc.
* `groupXY.py` with the main script, which loads the images, calls the functions, and reproduces the results.  
