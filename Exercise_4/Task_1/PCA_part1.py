# First part of Task-1
# reading the pca_dataset.txt and using Library funtion
# to implement PCA using method for Singluar Value decomposition.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

def pca_svd_from_file():
    pca_text_file = pd.read_csv('pca_dataset.txt', delimiter=' ')
    pca_scaled_values = preprocessing.scale(pca_text_file.T)

    pca_object = PCA()
    #pca_values = pca_scaled_values.to_numpy()

    return pca_scaled_values

#plt.plot(pca_values)

#print(pca_results.fit(pca_values))
# pca_values = pca_scaled_values.to_numpy()


