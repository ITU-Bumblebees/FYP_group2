
# Task 0

**Description:**
The original data is available at https://challenge.isic-archive.com/landing/2017/. We first only use a subset of this data, available on Github. Go through the data (images and meta-data) that you have available to understand what’s available to you, and write a brief description. What types of diagnoses are there, and how many images are there of each? What kind of meta-data is available? Is there some missing data? Are there images of low quality?  
As there are quite a few images, you are allowed to resize the images (for example, to be 300 pixels in width).

- [x] Look at pictures
- [x] What types of diagnoses are there, and how many images are there of each?
- [ ] What kind of meta-data is available?
- [x] Is there some missing data?
- [ ] Are there images of low quality? 


### Data
In our data we have the area and perimeter of the different lesions and their name. and for the features we have melanoma an seborrheic keratosis also according to the image id's.
The 'sickness' data are flags.
There are 30 melanoma cases and 42 keratosis cases total, out of the 150 images in the data.
We checked all the images and there are no missing segmentations. There are no missing values in either datasets (features and sickness)

### Looking at pictures
Some of the pictures don't match up and don't make sense when you look at the pictures.
Most of the time it's because the lesion in the image is almost not noticeable

Melanoma and seborrheic keratosis
