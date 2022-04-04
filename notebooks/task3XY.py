

import Config, os
from picture import Picture
import task3XY_functions.py as utils
from PIL import Image
import matplotlib.pyplot as plt


groundtruth = Config.example_ground_truth_path

def main():
    image_names = [ID for ID in groundtruth['image_id']]

    features_df = pd.DataFrame(columns=['ISIC', 'Assymetry', 'Compactness', 'Color', 'Melanoma'])
    for pic in image_names[0:50]:
        img_bw = Image.open(Config.mask_path + os.sep + pic + '_segmentation.png') # open mask image
        img_bw = utils.cut_image(img_bw)

        img = plt.imread(Config.images_path + os.sep + pic + '.jpg') # open image
        pict = Picture(img = img, img_bw = img_bw)
        
        tempdf = utils.get_row(pict, pic)
        features_df.append(tempdf)

    utils.train_evaluate_classifiers(features_df)

