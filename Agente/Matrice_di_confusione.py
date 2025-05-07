print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

try:
    import cPickle as Pickle
except:
    import pickle


import pyAudioAnalysis.audioFeatureExtraction as AuF
from Praat_feature.prosodic_feature import PraatFeatures
import os
import xlsxwriter
import librosa
import numpy
import matplotlib.pyplot as plt
import scipy
import matplotlib
import pitch
from python_speech_features import delta




def get_train_test(df, y_col, x_cols, ratio):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    #mask = np.random.rand(len(df)) &lt; ratio
    #mask = np.random.rand(len(df)); #la funzione np.random.rand restituisce un array di un certo numero di elementi
                                     #pari al valore intero passatogli come paramentro. Gli elementi dell'array sono
                                     #valori compresi tra 0 e 1
    #df_train = df[mask]
    #df_test = df[~mask]

    df_train, df_test = train_test_split(df, test_size=0.5)



    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values

    return df_train, df_test, X_train, Y_train, X_test, Y_test




dataset = pd.read_excel("Dataset_temp.xlsx")
#dataset = pd.read_excel("Dataset_DEFINITIVO.xlsx")
#dataset = pd.read_excel("a.xlsx")
dataset.head()


y_col_glass = 'Emozione'
x_cols_glass = list(dataset.columns.values)
x_cols_glass.remove(y_col_glass)#tolgo la colonna emozione

train_test_ratio = 0.7
#class_names = ['Medie delle MFCC per ogni audio', 'Medie del ZCR per ogni audio', 'Medie SPECTRAL_ROLLOF per ogni audio', 'Valori di SPECTRAL ENTROPY per ogni audio', 'Medie Energy per ogni audio','Pitch_MAX', 'Pitch_MEAN', 'Intensity_MIN', 'Intensity_MEAN', 'Intensity_MAX', 'Pitch_diff_max_mean', 'Medie derivata prima']
class_names = np.array(['niente','Medie delle MFCC per ogni audio', 'Medie del ZCR per ogni audio', 'Medie SPECTRAL_ROLLOF per ogni audio', 'Valori di SPECTRAL ENTROPY per ogni audio', 'Medie Energy per ogni audio','Pitch_MAX', 'Pitch_MEAN', 'Intensity_MIN', 'Intensity_MEAN', 'Intensity_MAX', 'Pitch_diff_max_mean', 'Medie derivata prima'])
#class_names = np.array(['Emozione'])
print("class_names mia: ", class_names)
ds_train, ds_test, X_train, Y_train, X_test, Y_test = get_train_test(dataset, y_col_glass, x_cols_glass,train_test_ratio)

with open('modello3.pickle', 'rb') as mp:
    regressor = pickle.load(mp)
#regressor2 = RandomForestClassifier(n_estimators=1000, random_state=0)
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)#calcolo le predizioni sul test_set



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()