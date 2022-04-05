

import Config, os
from picture import Picture
import task3XY_functions as utils
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

groundtruth = pd.read_csv(Config.example_ground_truth_path)

def main():
    image_names = [ID for ID in groundtruth['image_id']]

    features_df = pd.DataFrame(columns=['ISIC', 'Assymetry', 'Compactness', 'Color', 'Melanoma'])
    for pic in image_names:
        img_bw = Image.open(Config.mask_path + os.sep + pic + '_segmentation.png') # open mask image

        img = plt.imread(Config.images_path + os.sep + pic + '.jpg') # open image
        pict = Picture(img = img, img_bw = img_bw)
        
        tempdf = utils.get_row(pict, pic)
        features_df = pd.concat([features_df, tempdf], ignore_index=True, axis=0)
        print(features_df, flush=True)

    utils.train_evaluate_classifiers(features_df)

if __name__ == '__main__':
    main()

