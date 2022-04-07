import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Config
from PIL import Image 
import os, Config
from skimage import morphology
from statsmodels.robust import mad


class Picture:
    def __init__(self, img, img_bw):
        self.img = img
        self.img_bw = self.cut_image(img_bw)

    #ASYMMETRY
    def cut_image(self, picture):
        width, height = picture.size
        image = np.array(picture)
        
        if width %2 != 0:
            image = np.delete(image, -1, 1)

        if height %2 != 0:
            image = np.delete(image, -1, 0)

        image = Image.fromarray(image)

        return image

    def _asymmetry(self, rot_img):
        ''' get the asymmetry between the left and right part of a given binary mask '''

        width, height = rot_img.size #mask should be quadratic and therefore have equal dimension
        size = width * height

        #check for uneven number of pixels (should not happen but just as a precaution)
        if width %2 != 0:
            raise TypeError("Uneven number of pixel and cannot be symmetric")
        
        #cut in half and fold
        left = rot_img.crop((0, 0, (width/2), height)) #left part of picture (left, top, right, bottom)
        right = rot_img.crop(((width/2), 0, width, height)) #right part of picture
        right = right.transpose(Image.FLIP_LEFT_RIGHT) #flip right part to compare

        #get the binary difference between left an right
        symmetry = np.where(np.array(left) != np.array(right), 1, 0)

        return np.sum(symmetry) / (size/2) #the percentage of asymmetry 
    
    def get_asymmetry(self):
        ''' get the asymmetry for a given mask by folding it in half from multiple angles'''
        return round(np.mean([self._asymmetry(self.img_bw), self._asymmetry(self.img_bw.rotate(30, expand= True)),self._asymmetry(self.img_bw.rotate(60, expand= True)),self._asymmetry(self.img_bw.rotate(90, expand= True))]),2)

    #BORDER
    def measure_area_perimeter(self): #Stolen from Veronika's github
    # Measure area: the sum of all white pixels in the mask image
        mask = np.where(np.array(self.img_bw)==255, 1, 0)
        area = np.sum(mask)

        # Measure perimeter: first find which pixels belong to the perimeter.
        struct_el = morphology.disk(1)
        mask_eroded = morphology.binary_erosion(mask, struct_el)
        image_perimeter = mask - mask_eroded

        # Now we have the perimeter image, the sum of all white pixels in it
        perimeter = np.sum(image_perimeter)

        return area, perimeter

    def get_compactness(self):
        ''' Computes and returns the compactness of a figure '''
        area, perimeter = self.measure_area_perimeter()
        return round(4*np.pi*area / perimeter ** 2, 4)


    #COLOR
    def get_color_variability(self):
        '''
            Assigns a color variability score
        '''
        if self._check_variability() < 20: 
            return 0 
        elif self._check_variability() < 50: 
            return 1
        else: 
            return 2

    def _check_variability(self):
        '''
            Returns a mean of the median absolute deviation of each color (rgb)
        '''
        self.img[self.img_bw==0] = 0
        
        #we then calculate the mad of each dimension 
        r, g, b = self.img[:,:,0], self.img[:,:,1], self.img[:,:,2]
        mad_r= mad(r[np.where(r != 0)])
        mad_g= mad(g[np.where(g != 0)])
        mad_b= mad(b[np.where(b != 0)])
        mad_result= [mad_r,mad_g,mad_b]

        #calculating the mean
        return np.mean(mad_result)