

import Config, os
from picture import Picture
import task3XY_functions as utils
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import pickle

groundtruth = pd.read_csv(Config.example_ground_truth_path)

def main():
    image_names = [ID for ID in groundtruth['image_id']]

    features_df = pd.DataFrame(columns=['ISIC', 'Asymmetry', 'Compactness', 'Color', 'Melanoma'])
    for pic in image_names:
        img_bw = Image.open(Config.mask_path + os.sep + pic + '_segmentation.png') # open mask image

        img = plt.imread(Config.images_path + os.sep + pic + '.jpg') # open image
        pict = Picture(img = img, img_bw = img_bw)
        
        tempdf = utils.get_row(pict, pic)
        features_df = pd.concat([features_df, tempdf], ignore_index=True, axis=0)
        
        if not features_df.shape[0] % 10:
            print(f'Still going' + '.'* int(features_df.shape[0] / 10), end='\r')


    features_df.to_csv('file_features', index=False)

    treeclassifier = utils.train_evaluate_classifiers(features_df)
    with open('treeclassifier.pickle', 'wb') as outfile:
        pickle.dump(treeclassifier, outfile)


if __name__ == '__main__':
    main()

