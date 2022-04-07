

import pickle, picture
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img = plt.imread('./../data/example_image/ISIC_0012099.jpg')
    img_bw = Image.open('./../data/example_segmentation/ISIC_0012099_segmentation.png')

    classify(img, img_bw)

def get_row(pict):  
    assym = pict.get_asymmetry()
    comp = pict.get_compactness()
    color = pict.get_color_variability()

    return pd.DataFrame([[assym, comp, color]], 
                        columns=['Asymmetry', 'Compactness', 'Color'])

def classify(img: Image, img_bw: np.ndarray):
    """ Uses a given classifier to predict melanoma """
    pict = picture.Picture(img, img_bw)
    features = get_row(pict)
    with open('treeclassifier.pickle', 'rb') as infile:
        tree = pickle.load(infile)

    k = tree.predict_proba(features)
    print(k)

if __name__ == '__main__':
    main()

