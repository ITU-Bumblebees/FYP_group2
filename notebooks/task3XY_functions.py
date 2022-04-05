import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Config
from PIL import Image 
import os, Config
from skimage import morphology
from statsmodels.robust import mad

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score


groundtruth = pd.read_csv(Config.example_ground_truth_path)

def cut_image(picture):
    width, height = picture.size
    image = np.array(picture)
    
    if width %2 != 0:
        image = np.delete(image, -1, 1)

    if height %2 != 0:
        image = np.delete(image, -1, 0)

    image = Image.fromarray(image)

    return image


def get_row(pict, pic):  
    assym = pict.get_assymetry()
    comp = pict.get_compactness()
    color = pict.get_color_variability()

    melanoma = groundtruth[groundtruth['image_id'] == pic]['melanoma'].iloc[0]
    return pd.DataFrame([[pic, assym, comp, color, melanoma]], 
                        columns=['ISIC', 'Assymetry', 'Compactness', 'Color', 'Melanoma'])


def train_evaluate_classifiers(df):
    testies = df

    X = testies[["Assymetry", "Compactness","Color"]]
    y = testies["Melanoma"]

    #split data set into a train, test and valification set
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size= 0.3, stratify=y, random_state=0)

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size= 0.5, stratify=y_dev)

    #train a classifier
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1trained = knn1.fit(X_train, y_train)

    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3trained = knn3.fit(X_train, y_train)

    loglen = int(np.log(len(df["Melanoma"])))
    knnlog = KNeighborsClassifier(n_neighbors=loglen)
    knnlogtrained = knnlog.fit(X_train, y_train)

    tree = DecisionTreeClassifier()
    treetrained = tree.fit(X_train, y_train)
    
    gauss = GaussianNB()
    gausstrained = gauss.fit(X_test, y_test)

    #evaluate the classifiers
    y_val_knn1 = knn1trained.predict(X_val)
    y_val_knn3 = knn3trained.predict(X_val)
    y_val_knnlog = knnlogtrained.predict(X_val)
    y_val_tree = treetrained.predict(X_val)
    y_val_gauss = gausstrained.predict(X_val)

    print("Accuracy (using numpy)")
    print(f"knn 1: {np.sum(y_val_knn1 == y_val)/ np.size(y_val)*100}")
    print(f"knn 3: {np.sum(y_val_knn3 == y_val)/ np.size(y_val)*100}")
    print(f"knn log: {np.sum(y_val_knnlog == y_val)/ np.size(y_val)*100}")
    print(f"tree: {np.sum(y_val_tree == y_val)/ np.size(y_val)*100}")
    print(f"gauss: {np.sum(y_val_gauss == y_val)/ np.size(y_val)*100}")

    print("Accuracy score")
    print(f"knn 1: {accuracy_score(y_val,y_val_knn1)}")
    print(f"knn 3: {accuracy_score(y_val,y_val_knn3)}")
    print(f"knn log: {accuracy_score(y_val,y_val_knnlog)}")
    print(f"tree: {accuracy_score(y_val,y_val_tree)}")
    print(f"gauss: {accuracy_score(y_val,y_val_gauss)}")

    print("roc auc score")
    print(f"knn1: {roc_auc_score(y_val, y_val_knn1)}")
    print(f"knn3: {roc_auc_score(y_val, y_val_knn3)}")
    print(f"knnlog: {roc_auc_score(y_val, y_val_knnlog)}")
    print(f"tree: {roc_auc_score(y_val, y_val_tree)}")
    print(f"gauss: {roc_auc_score(y_val, y_val_gauss)}")

    
