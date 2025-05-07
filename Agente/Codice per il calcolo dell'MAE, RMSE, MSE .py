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
#dataset = pd.read_excel("data.xlsx")
#dataset = pd.read_excel("a.xlsx")
dataset.head()


y_col_glass = 'Emozione'
x_cols_glass = list(dataset.columns.values)
x_cols_glass.remove(y_col_glass)#tolgo la colonna emozione

train_test_ratio = 0.7
ds_train, ds_test, X_train, Y_train, X_test, Y_test = get_train_test(dataset, y_col_glass, x_cols_glass,train_test_ratio)


clf = RandomForestClassifier(n_estimators=10,criterion='gini',max_depth=1000,min_samples_leaf=1,max_features='log2')
clf.fit(X_train, Y_train)


test_score = clf.score(X_test, Y_test)
y_pred = clf.predict(X_test)

with open('modello-pickle5_nuovo.pickle', 'wb') as f:
    pickle.dump(clf,f, pickle.HIGHEST_PROTOCOL)


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

print("accuratezza: ", test_score)


#modello_50.pickle = 0.61564 accuratezza del test set
#modello_51.pickle = 0.58843 accuratezza del test set (ha riconosciuto gioia, 50% test_set, da spesso gioia)
#modello_52.pickle = 0.41777 accuratezza del test set (CONTIENE TUTTI GLI AUDIO EMOVO + NOSTRI, ha riconosciuto paura, resituisce quasi sempre paura)

#modello3.pickle = 0.66 accuretazza del test set (solo su emovo con 0.5 test_set)
#modello5.pickle = 0.54 accuratezza (For (10,gini,1000,1,log2) - train, test score: 	 0.99757 	-	 0.50847 Vettore risultati:  ['rabbia.wav', 'no', 'no', 'no', 'tristezza.wav', 'disgusto.wav', 'gioia.wav']

#modello6.pickle = For (10,entropy,10,1,log2) - train, test score: 	 0.99513 	-	 0.57 Vettore risultati:  ['no', 'no', 'paura.wav', 'no', 'tristezza.wav', 'disgusto.wav', 'gioia.wav']
#modello7.pickle = For (10,entropy,10,2,auto) - train, test score: 	 0.95134 	-	 0.53 Vettore risultati:  ['rabbia.wav', 'no', 'no', 'no', 'tristezza.wav', 'disgusto.wav', 'gioia.wav']
#modello8.pickle = For (100,entropy,100,4,log2) - train, test score: 	 0.95238 	-	 0.58 Vettore risultati:  ['rabbia.wav', 'no', 'paura.wav', 'no', 'tristezza.wav', 'no', 'gioia.wav']
#modello9.pickle = For (1000,gini,100,2,log2) - train, test score: 	 0.99660 	-	 0.68 Vettore risultati:  ['rabbia.wav', 'no', 'no', 'no', 'tristezza.wav', 'no', 'gioia.wav']

#modello-pickle3_nuovo.pickle = 0.68 accuratezza (n_stimatori = 1000, solo su emovo con 0.5 test_set)
#modello-pickle5_nuovo.pickle = 0.55 accuratezza(n_estimators=10,criterion='gini',max_depth=1000,min_samples_leaf=1,max_features='log2')


'''
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

'''
#Qui inizio a prende tutte le features per l'audio su cui voglio la predizione
feature = PraatFeatures()
for filename in os.listdir():
    if (filename == "paura-tristezza.wav"):
        print(filename)
        y, sr = librosa.load(filename)
        ZCR = librosa.feature.zero_crossing_rate(y=y)
        MFCC = librosa.feature.mfcc(y, sr)
        ROLLOF = librosa.feature.spectral_rolloff(y, sr)
        SPECTRAL_ENTROPY = AuF.stSpectralEntropy(y)#Qui prendo il valore di Spectral_entropy per l'audio
        data = feature.get_all_sound_feature(filename)
        print("data: ", data)

        x = data.keys()
        print(x)
        y = -1
        for i in x:
            y = y + 1

            if (y == 0):
                c = data.get(i).keys()
                print("c:" , c)
                a = -1
                for j in c:
                    a = a + 1
                    if (a == 1):
                        print("data[0].[1]: ", data.get(i).get(j))
                        Energy = data.get(i).get(j)#qui prendo l'energia dell'audio

            #Ora prendo i valori del pitch massimo e del pitch medio per ogni audio
            if (y == 1):
                print("")
                c = data.get(i).keys()
                print("c:", c)
                a = -1
                for j in c:
                    a = a + 1
                    if (a == 0):
                        print("data[1].[0]: ", data.get(i).get(j))
                        Pitch_max = data.get(i).get(j)#qui memorizzo il pitch massimo dell'audio

                    if(a == 1):
                        print("data[1][1]: ", data.get(i).get(j))
                        Pitch_mean = data.get(i).get(j)#qui memorizzo il pitch medio dell'audio

                    if (a == 5):
                        print("data[1][5]: ", data.get(i).get(j))
                        Pitch_diff_max_mean = data.get(i).get(j)#qui memorizzo il Pitch_diff_max_mean dell'audio



            if (y == 2):
                print("")
                c = data.get(i).keys()
                print("c:", c)

                a = -1
                for j in c:
                    a = a + 1
                    if (a == 0):
                        print("data[2].[0]: ", data.get(i).get(j))
                        Intensity_min = data.get(i).get(j) #qui memorizzo l'intensity min dell'audio

                    if (a == 1):
                        print("data[2][1]: ", data.get(i).get(j))
                        Intensity_mean = data.get(i).get(j)#qui memorizzo l'intensity mean dell'audio

                    if (a == 2):
                        print("data[2][2]: ", data.get(i).get(j))
                        Intensity_max = data.get(i).get(j)#qui memorizzo l'intensity max dell'audio

        derivata = delta(MFCC, 2)
        derivata_media = derivata.mean()#qui memorizzo la derivata media dell'audio
        print("")
        print("Derivata media: ", derivata_media)

#Qui prendo la media dell'MFCC dell'audio
        somma_MFCC = 0
        contatore_MFCC = 0
        for i in range(0, MFCC.shape[0]):
            for j in range(0, MFCC.shape[1]):
                somma_MFCC = somma_MFCC + MFCC[i][j]
                contatore_MFCC = contatore_MFCC + 1

        media_MFCC = somma_MFCC / contatore_MFCC
        print("")
        print("MFCC: ", MFCC)
        print("media_MFCC: ", media_MFCC)

#Ora prendo la media della ZCR dell'audio
        somma_ZCR = 0
        contatore_ZCR = 0
        for i in range(0, ZCR.shape[0]):
            for j in range(0, ZCR.shape[1]):
                somma_ZCR = somma_ZCR + ZCR[i][j]
                contatore_ZCR = contatore_ZCR + 1

        media_ZCR = somma_ZCR / contatore_ZCR
        print("")
        print("ZCR: ", ZCR)
        print("media_ZCR: ", media_ZCR)

#Ora prendo la media della Spectral Rollof per l'audio
        somma_Rollof = 0
        contatore_Rollof = 0
        for i in range(0, ROLLOF.shape[0]):
            for j in range(0, ROLLOF.shape[1]):
                somma_Rollof = somma_Rollof + ROLLOF[i][j]
                contatore_Rollof = contatore_Rollof + 1

        media_Rollof = somma_Rollof / contatore_Rollof
        print("")
        print("ROLLOF: ", ROLLOF)
        print("media_Rollof: ", media_Rollof)


        print("")
        print("Valore di spectral_entropy: ", SPECTRAL_ENTROPY)
'''

'''
#A questo punto ho tutte le feature dell'audio e quindi posso andare ad effettuare la predizione

#(n_estimators=n_est,criterion=crit, max_depth=max_d, min_samples_leaf=min_samples, max_features=max_feat)
#regressor = RandomForestClassifier(n_estimators=1000)
regressor = RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=100,min_samples_leaf=2,max_features="log2")
regressor.fit(X_train, Y_train)#addestro il classificatore

y_pred = regressor.predict(X_test)
print("Accuratezza con il test set:",metrics.accuracy_score(Y_test, y_pred))
'''


#modello_50.pickle = 0.61564 accuratezza del test set
#modello_51.pickle = 0.58843 accuratezza del test set (ha riconosciuto gioia, 50% test_set, da spesso gioia)
#modello_52.pickle = 0.41777 accuratezza del test set (CONTIENE TUTTI GLI AUDIO EMOVO + NOSTRI, ha riconosciuto paura, resituisce quasi sempre paura)
#modello3.pickle = 0.66 accuretazza del test set (solo su emovo con 0.5 test_set)

#modello5.pickle = 0.54 accuratezza (For (10,gini,1000,1,log2) - train, test score: 	 0.99757 	-	 0.50847 Vettore risultati:  ['rabbia.wav', 'no', 'no', 'no', 'tristezza.wav', 'disgusto.wav', 'gioia.wav']
#modello6.pickle = For (10,entropy,10,1,log2) - train, test score: 	 0.99513 	-	 0.57 Vettore risultati:  ['no', 'no', 'paura.wav', 'no', 'tristezza.wav', 'disgusto.wav', 'gioia.wav']
#modello7.pickle = For (10,entropy,10,2,auto) - train, test score: 	 0.95134 	-	 0.53 Vettore risultati:  ['rabbia.wav', 'no', 'no', 'no', 'tristezza.wav', 'disgusto.wav', 'gioia.wav']
#modello8.pickle = For (100,entropy,100,4,log2) - train, test score: 	 0.95238 	-	 0.58 Vettore risultati:  ['rabbia.wav', 'no', 'paura.wav', 'no', 'tristezza.wav', 'no', 'gioia.wav']
#modello9.pickle = For (1000,gini,100,2,log2) - train, test score: 	 0.99660 	-	 0.68 Vettore risultati:  ['rabbia.wav', 'no', 'no', 'no', 'tristezza.wav', 'no', 'gioia.wav']



#Queste due righe di sotto servono per la serializzazione del modello
'''
with open('modello9.pickle', 'wb') as f:
    pickle.dump(regressor,f, pickle.HIGHEST_PROTOCOL)
'''


''' da togliere
#Qui inizio a prende tutte le features per l'audio su cui voglio la predizione
feature = PraatFeatures()
for filename in os.listdir():
    if (filename == "paura-tristezza.wav"):
        print(filename)
        y, sr = librosa.load(filename)
        ZCR = librosa.feature.zero_crossing_rate(y=y)
        MFCC = librosa.feature.mfcc(y, sr)
        ROLLOF = librosa.feature.spectral_rolloff(y, sr)
        SPECTRAL_ENTROPY = AuF.stSpectralEntropy(y)#Qui prendo il valore di Spectral_entropy per l'audio
        data = feature.get_all_sound_feature(filename)
        print("data: ", data)

        x = data.keys()
        print(x)
        y = -1
        for i in x:
            y = y + 1

            if (y == 0):
                c = data.get(i).keys()
                print("c:" , c)
                a = -1
                for j in c:
                    a = a + 1
                    if (a == 1):
                        print("data[0].[1]: ", data.get(i).get(j))
                        Energy = data.get(i).get(j)#qui prendo l'energia dell'audio

            #Ora prendo i valori del pitch massimo e del pitch medio per ogni audio
            if (y == 1):
                print("")
                c = data.get(i).keys()
                print("c:", c)
                a = -1
                for j in c:
                    a = a + 1
                    if (a == 0):
                        print("data[1].[0]: ", data.get(i).get(j))
                        Pitch_max = data.get(i).get(j)#qui memorizzo il pitch massimo dell'audio

                    if(a == 1):
                        print("data[1][1]: ", data.get(i).get(j))
                        Pitch_mean = data.get(i).get(j)#qui memorizzo il pitch medio dell'audio

                    if (a == 5):
                        print("data[1][5]: ", data.get(i).get(j))
                        Pitch_diff_max_mean = data.get(i).get(j)#qui memorizzo il Pitch_diff_max_mean dell'audio



            if (y == 2):
                print("")
                c = data.get(i).keys()
                print("c:", c)

                a = -1
                for j in c:
                    a = a + 1
                    if (a == 0):
                        print("data[2].[0]: ", data.get(i).get(j))
                        Intensity_min = data.get(i).get(j) #qui memorizzo l'intensity min dell'audio

                    if (a == 1):
                        print("data[2][1]: ", data.get(i).get(j))
                        Intensity_mean = data.get(i).get(j)#qui memorizzo l'intensity mean dell'audio

                    if (a == 2):
                        print("data[2][2]: ", data.get(i).get(j))
                        Intensity_max = data.get(i).get(j)#qui memorizzo l'intensity max dell'audio

        derivata = delta(MFCC, 2)
        derivata_media = derivata.mean()#qui memorizzo la derivata media dell'audio
        print("")
        print("Derivata media: ", derivata_media)

#Qui prendo la media dell'MFCC dell'audio
        somma_MFCC = 0
        contatore_MFCC = 0
        for i in range(0, MFCC.shape[0]):
            for j in range(0, MFCC.shape[1]):
                somma_MFCC = somma_MFCC + MFCC[i][j]
                contatore_MFCC = contatore_MFCC + 1

        media_MFCC = somma_MFCC / contatore_MFCC
        print("")
        print("MFCC: ", MFCC)
        print("media_MFCC: ", media_MFCC)

#Ora prendo la media della ZCR dell'audio
        somma_ZCR = 0
        contatore_ZCR = 0
        for i in range(0, ZCR.shape[0]):
            for j in range(0, ZCR.shape[1]):
                somma_ZCR = somma_ZCR + ZCR[i][j]
                contatore_ZCR = contatore_ZCR + 1

        media_ZCR = somma_ZCR / contatore_ZCR
        print("")
        print("ZCR: ", ZCR)
        print("media_ZCR: ", media_ZCR)

#Ora prendo la media della Spectral Rollof per l'audio
        somma_Rollof = 0
        contatore_Rollof = 0
        for i in range(0, ROLLOF.shape[0]):
            for j in range(0, ROLLOF.shape[1]):
                somma_Rollof = somma_Rollof + ROLLOF[i][j]
                contatore_Rollof = contatore_Rollof + 1

        media_Rollof = somma_Rollof / contatore_Rollof
        print("")
        print("ROLLOF: ", ROLLOF)
        print("media_Rollof: ", media_Rollof)


        print("")
        print("Valore di spectral_entropy: ", SPECTRAL_ENTROPY)
'''










'''


print("Miglioramenti random forest: ")
#Provo a migliorare il random forest e vedo i risultati


GDB_params = {
    'n_estimators': [10, 100, 500, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 100, 1000, None],
    'min_samples_leaf':[1, 2, 4],
    'max_features':["sqrt", "log2", None,"auto"]

}

GDB_params2 = {

    'nome_file': ["rabbia.wav", "sorpresa.wav", "paura.wav", "mio_neutrale.wav", "tristezza_mia.wav", "disgusto.wav","gioia.wav"],

}

'''

'''
([[media_MFCC, media_ZCR, media_Rollof, SPECTRAL_ENTROPY, Energ
Pitch_max, Pitch_mean, Intensity_min, Intensity_mean, Intensity_max
Pitch_diff_max_mean, derivata_media]])
'''


'''
media_MFCC1 = 0
media_ZCR1 = 0
media_Rollof1 = 0
SPECTRAL_ENTROPY1 = 0
Energy1 = 0
Pitch_max1 = 0
Pitch_diff_max_mean1 = 0
Pitch_mean1 = 0
Intensity_min1 = 0
Intensity_max1 = 0
Intensity_mean1 = 0
derivata_media1 = 0

media_MFCC2 = 0
media_ZCR2 = 0
media_Rollof2 = 0
SPECTRAL_ENTROPY2 = 0
Energy2 = 0
Pitch_max2 = 0
Pitch_diff_max_mean2 = 0
Pitch_mean2 = 0
Intensity_min2 = 0
Intensity_max2 = 0
Intensity_mean2 = 0
derivata_media2 = 0


media_MFCC3 = 0
media_ZCR3 = 0
media_Rollof3 = 0
SPECTRAL_ENTROPY3 = 0
Energy3 = 0
Pitch_max3 = 0
Pitch_diff_max_mean3 = 0
Pitch_mean3 = 0
Intensity_min3 = 0
Intensity_max3 = 0
Intensity_mean3 = 0
derivata_media3 = 0


media_MFCC4 = 0
media_ZCR4 = 0
media_Rollof4 = 0
SPECTRAL_ENTROPY4 = 0
Energy4 = 0
Pitch_max4 = 0
Pitch_diff_max_mean4 = 0
Pitch_mean4 = 0
Intensity_min4 = 0
Intensity_max4 = 0
Intensity_mean4 = 0
derivata_media4 = 0


media_MFCC5 = 0
media_ZCR5 = 0
media_Rollof5 = 0
SPECTRAL_ENTROPY5 = 0
Energy5 = 0
Pitch_max5 = 0
Pitch_diff_max_mean5 = 0
Pitch_mean5 = 0
Intensity_min5 = 0
Intensity_max5 = 0
Intensity_mean5 = 0
derivata_media5 = 0



media_MFCC6 = 0
media_ZCR6 = 0
media_Rollof6 = 0
SPECTRAL_ENTROPY6 = 0
Energy6 = 0
Pitch_max6 = 0
Pitch_diff_max_mean6 = 0
Pitch_mean6 = 0
Intensity_min6 = 0
Intensity_max6 = 0
Intensity_mean6 = 0
derivata_media6 = 0



media_MFCC7 = 0
media_ZCR7 = 0
media_Rollof7 = 0
SPECTRAL_ENTROPY7 = 0
Energy7 = 0
Pitch_max7 = 0
Pitch_diff_max_mean7 = 0
Pitch_mean7 = 0
Intensity_min7 = 0
Intensity_max7 = 0
Intensity_mean7 = 0
derivata_media7 = 0



feature = PraatFeatures()
for nome_file in GDB_params2['nome_file']:

    for filename in os.listdir():
        if (filename == nome_file):
            y, sr = librosa.load(filename)
            ZCR = librosa.feature.zero_crossing_rate(y=y)
            MFCC = librosa.feature.mfcc(y, sr)
            ROLLOF = librosa.feature.spectral_rolloff(y, sr)
            SPECTRAL_ENTROPY = AuF.stSpectralEntropy(y)  # Qui prendo il valore di Spectral_entropy per l'audio
            data = feature.get_all_sound_feature(filename)
            x = data.keys()
            y = -1
            for i in x:
                y = y + 1

                if (y == 0):
                    c = data.get(i).keys()
                    a = -1
                    for j in c:
                        a = a + 1
                        if (a == 1):
                            if (filename == "rabbia.wav"):
                                Energy1 = data.get(i).get(j)  # qui prendo l'energia dell'audio
                            if (filename == "sorpresa.wav"):
                                Energy2 = data.get(i).get(j)
                            if (filename == "paura.wav"):
                                Energy3 = data.get(i).get(j)
                            if (filename == "mio-neutrale.wav"):
                                Energy4 = data.get(i).get(j)
                            if (filename == "tristezza.wav"):
                                Energy5 = data.get(i).get(j)
                            if (filename == "disgusto.wav"):
                                Energy6 = data.get(i).get(j)
                            if (filename == "gioia.wav"):
                                Energy7 = data.get(i).get(j)



             # Ora prendo i valori del pitch massimo e del pitch medio per ogni audio
                if (y == 1):
                    c = data.get(i).keys()
                    a = -1
                    for j in c:
                        a = a + 1
                        if (a == 0):
                            if(filename == "rabbia.wav"):
                                Pitch_max1 = data.get(i).get(j)  # qui memorizzo il pitch massimo dell'audio
                            if (filename == "sorpresa.wav"):
                                Pitch_max2 = data.get(i).get(j)
                            if (filename == "paura.wav"):
                                Pitch_max3 = data.get(i).get(j)
                            if (filename == "mio-neutrale.wav"):
                                Pitch_max4 = data.get(i).get(j)
                            if (filename == "tristezza.wav"):
                                Pitch_max5 = data.get(i).get(j)
                            if (filename == "disgusto.wav"):
                                Pitch_max6 = data.get(i).get(j)
                            if (filename == "gioia.wav"):
                                Pitch_max7 = data.get(i).get(j)


                        if (a == 1):

                            if(filename == "rabbia.wav"):
                                Pitch_mean1 = data.get(i).get(j)  # qui memorizzo il pitch medio dell'audio
                            if (filename == "sorpresa.wav"):
                                Pitch_mean2 = data.get(i).get(j)  # qui memorizzo il pitch medio dell'audio
                            if (filename == "paura.wav"):
                                Pitch_mean3 = data.get(i).get(j)  # qui memorizzo il pitch medio dell'audio
                            if (filename == "mio-neutrale.wav"):
                                Pitch_mean4 = data.get(i).get(j)  # qui memorizzo il pitch medio dell'audio
                            if (filename == "tristezza.wav"):
                                Pitch_mean5 = data.get(i).get(j)  # qui memorizzo il pitch medio dell'audio
                            if (filename == "disgusto.wav"):
                                Pitch_mean6 = data.get(i).get(j)  # qui memorizzo il pitch medio dell'audio
                            if (filename == "gioia.wav"):
                                Pitch_mean7 = data.get(i).get(j)  # qui memorizzo il pitch medio dell'audio


                        if (a == 5):
                            if(filename == "rabbia.wav"):
                                Pitch_diff_max_mean1 = data.get(i).get(j)  # qui memorizzo il Pitch_diff_max_mean dell'audio
                            if (filename == "sorpresa.wav"):
                                Pitch_diff_max_mean2 = data.get(i).get(j)
                            if (filename == "paura.wav"):
                                Pitch_diff_max_mean3 = data.get(i).get(j)
                            if (filename == "mio-neutrale.wav"):
                                Pitch_diff_max_mean4 = data.get(i).get(j)
                            if (filename == "tristezza.wav"):
                                Pitch_diff_max_mean5 = data.get(i).get(j)
                            if (filename == "disgusto.wav"):
                                Pitch_diff_max_mean6 = data.get(i).get(j)
                            if (filename == "gioia.wav"):
                                Pitch_diff_max_mean7 = data.get(i).get(j)

                if (y == 2):
                    c = data.get(i).keys()


                    a = -1
                    for j in c:
                        a = a + 1
                        if (a == 0):
                            if(filename == "rabbia.wav"):
                                Intensity_min1 = data.get(i).get(j)  # qui memorizzo l'intensity min dell'audio
                            if (filename == "sorpresa.wav"):
                                Intensity_min2 = data.get(i).get(j)
                            if (filename == "paura.wav"):
                                Intensity_min3 = data.get(i).get(j)
                            if (filename == "mio-neutrale.wav"):
                                Intensity_min4 = data.get(i).get(j)
                            if (filename == "tristezza.wav"):
                                Intensity_min5 = data.get(i).get(j)
                            if (filename == "disgusto.wav"):
                                Intensity_min6 = data.get(i).get(j)
                            if (filename == "gioia.wav"):
                                Intensity_min7 = data.get(i).get(j)



                        if (a == 1):
                            if(filename == "rabbia.wav"):
                                Intensity_mean1 = data.get(i).get(j)  # qui memorizzo l'intensity mean dell'audio
                            if (filename == "sorpresa.wav"):
                                Intensity_mean2 = data.get(i).get(j)
                            if (filename == "paura.wav"):
                                Intensity_mean3 = data.get(i).get(j)
                            if (filename == "mio-neutrale.wav"):
                                Intensity_mean4 = data.get(i).get(j)
                            if (filename == "tristezza.wav"):
                                Intensity_mean5 = data.get(i).get(j)
                            if (filename == "disgusto.wav"):
                                Intensity_mean6 = data.get(i).get(j)
                            if (filename == "gioia.wav"):
                                Intensity_mean7 = data.get(i).get(j)


                        if (a == 2):
                            if(filename == "rabbia.wav"):
                                Intensity_max1 = data.get(i).get(j)  # qui memorizzo l'intensity max dell'audio
                            if (filename == "sorpresa.wav"):
                                Intensity_max2 = data.get(i).get(j)
                            if (filename == "paura.wav"):
                                Intensity_max3 = data.get(i).get(j)
                            if (filename == "mio-neutrale.wav"):
                                Intensity_max4 = data.get(i).get(j)
                            if (filename == "tristezza.wav"):
                                Intensity_max5 = data.get(i).get(j)
                            if (filename == "disgusto.wav"):
                                Intensity_max6 = data.get(i).get(j)
                            if (filename == "gioia.wav"):
                                Intensity_max7 = data.get(i).get(j)



            if(filename == "rabbia.wav"):

                derivata1 = delta(MFCC, 2)
                derivata_media1 = derivata1.mean()  # qui memorizzo la derivata media dell'audio

                somma_MFCC = 0
                contatore_MFCC = 0
                for i in range(0, MFCC.shape[0]):
                    for j in range(0, MFCC.shape[1]):
                        somma_MFCC = somma_MFCC + MFCC[i][j]
                        contatore_MFCC = contatore_MFCC + 1

                media_MFCC1 = somma_MFCC / contatore_MFCC

                somma_ZCR = 0
                contatore_ZCR = 0
                for i in range(0, ZCR.shape[0]):
                    for j in range(0, ZCR.shape[1]):
                        somma_ZCR = somma_ZCR + ZCR[i][j]
                        contatore_ZCR = contatore_ZCR + 1

                media_ZCR1 = somma_ZCR / contatore_ZCR


            # Ora prendo la media della Spectral Rollof per l'audio
                somma_Rollof = 0
                contatore_Rollof = 0
                for i in range(0, ROLLOF.shape[0]):
                    for j in range(0, ROLLOF.shape[1]):
                        somma_Rollof = somma_Rollof + ROLLOF[i][j]
                        contatore_Rollof = contatore_Rollof + 1

                media_Rollof1 = somma_Rollof / contatore_Rollof





            if (filename == "sorpresa.wav"):

                derivata2 = delta(MFCC, 2)
                derivata_media2 = derivata1.mean()  # qui memorizzo la derivata media dell'audio

                somma_MFCC = 0
                contatore_MFCC = 0
                for i in range(0, MFCC.shape[0]):
                    for j in range(0, MFCC.shape[1]):
                        somma_MFCC = somma_MFCC + MFCC[i][j]
                        contatore_MFCC = contatore_MFCC + 1

                media_MFCC2 = somma_MFCC / contatore_MFCC

                somma_ZCR = 0
                contatore_ZCR = 0
                for i in range(0, ZCR.shape[0]):
                    for j in range(0, ZCR.shape[1]):
                        somma_ZCR = somma_ZCR + ZCR[i][j]
                        contatore_ZCR = contatore_ZCR + 1

                media_ZCR2 = somma_ZCR / contatore_ZCR

                # Ora prendo la media della Spectral Rollof per l'audio
                somma_Rollof = 0
                contatore_Rollof = 0
                for i in range(0, ROLLOF.shape[0]):
                    for j in range(0, ROLLOF.shape[1]):
                        somma_Rollof = somma_Rollof + ROLLOF[i][j]
                        contatore_Rollof = contatore_Rollof + 1

                media_Rollof2 = somma_Rollof / contatore_Rollof



            if (filename == "paura.wav"):

                derivata3 = delta(MFCC, 2)
                derivata_media3 = derivata1.mean()  # qui memorizzo la derivata media dell'audio

                somma_MFCC = 0
                contatore_MFCC = 0
                for i in range(0, MFCC.shape[0]):
                    for j in range(0, MFCC.shape[1]):
                        somma_MFCC = somma_MFCC + MFCC[i][j]
                        contatore_MFCC = contatore_MFCC + 1

                media_MFCC3 = somma_MFCC / contatore_MFCC

                somma_ZCR = 0
                contatore_ZCR = 0
                for i in range(0, ZCR.shape[0]):
                    for j in range(0, ZCR.shape[1]):
                        somma_ZCR = somma_ZCR + ZCR[i][j]
                        contatore_ZCR = contatore_ZCR + 1

                media_ZCR3 = somma_ZCR / contatore_ZCR

                # Ora prendo la media della Spectral Rollof per l'audio
                somma_Rollof = 0
                contatore_Rollof = 0
                for i in range(0, ROLLOF.shape[0]):
                    for j in range(0, ROLLOF.shape[1]):
                        somma_Rollof = somma_Rollof + ROLLOF[i][j]
                        contatore_Rollof = contatore_Rollof + 1

                media_Rollof3 = somma_Rollof / contatore_Rollof




            if (filename == "mio-neutrale.wav"):

                derivata4 = delta(MFCC, 2)
                derivata_media4 = derivata1.mean()  # qui memorizzo la derivata media dell'audio

                somma_MFCC = 0
                contatore_MFCC = 0
                for i in range(0, MFCC.shape[0]):
                    for j in range(0, MFCC.shape[1]):
                        somma_MFCC = somma_MFCC + MFCC[i][j]
                        contatore_MFCC = contatore_MFCC + 1

                media_MFCC4 = somma_MFCC / contatore_MFCC

                somma_ZCR = 0
                contatore_ZCR = 0
                for i in range(0, ZCR.shape[0]):
                    for j in range(0, ZCR.shape[1]):
                        somma_ZCR = somma_ZCR + ZCR[i][j]
                        contatore_ZCR = contatore_ZCR + 1

                media_ZCR4 = somma_ZCR / contatore_ZCR

                # Ora prendo la media della Spectral Rollof per l'audio
                somma_Rollof = 0
                contatore_Rollof = 0
                for i in range(0, ROLLOF.shape[0]):
                    for j in range(0, ROLLOF.shape[1]):
                        somma_Rollof = somma_Rollof + ROLLOF[i][j]
                        contatore_Rollof = contatore_Rollof + 1

                media_Rollof4 = somma_Rollof / contatore_Rollof


            if (filename == "tristezza.wav"):

                derivata5 = delta(MFCC, 2)
                derivata_media5 = derivata1.mean()  # qui memorizzo la derivata media dell'audio

                somma_MFCC = 0
                contatore_MFCC = 0
                for i in range(0, MFCC.shape[0]):
                    for j in range(0, MFCC.shape[1]):
                        somma_MFCC = somma_MFCC + MFCC[i][j]
                        contatore_MFCC = contatore_MFCC + 1

                media_MFCC5 = somma_MFCC / contatore_MFCC


                somma_ZCR = 0
                contatore_ZCR = 0
                for i in range(0, ZCR.shape[0]):
                    for j in range(0, ZCR.shape[1]):
                        somma_ZCR = somma_ZCR + ZCR[i][j]
                        contatore_ZCR = contatore_ZCR + 1

                media_ZCR5 = somma_ZCR / contatore_ZCR

                # Ora prendo la media della Spectral Rollof per l'audio
                somma_Rollof = 0
                contatore_Rollof = 0
                for i in range(0, ROLLOF.shape[0]):
                    for j in range(0, ROLLOF.shape[1]):
                        somma_Rollof = somma_Rollof + ROLLOF[i][j]
                        contatore_Rollof = contatore_Rollof + 1

                media_Rollof5 = somma_Rollof / contatore_Rollof




            if (filename == "disgusto.wav"):

                derivata6 = delta(MFCC, 2)
                derivata_media6 = derivata1.mean()  # qui memorizzo la derivata media dell'audio

                somma_MFCC = 0
                contatore_MFCC = 0
                for i in range(0, MFCC.shape[0]):
                    for j in range(0, MFCC.shape[1]):
                        somma_MFCC = somma_MFCC + MFCC[i][j]
                        contatore_MFCC = contatore_MFCC + 1

                media_MFCC6 = somma_MFCC / contatore_MFCC

                somma_ZCR = 0
                contatore_ZCR = 0
                for i in range(0, ZCR.shape[0]):
                    for j in range(0, ZCR.shape[1]):
                        somma_ZCR = somma_ZCR + ZCR[i][j]
                        contatore_ZCR = contatore_ZCR + 1

                media_ZCR6 = somma_ZCR / contatore_ZCR

                # Ora prendo la media della Spectral Rollof per l'audio
                somma_Rollof = 0
                contatore_Rollof = 0
                for i in range(0, ROLLOF.shape[0]):
                    for j in range(0, ROLLOF.shape[1]):
                        somma_Rollof = somma_Rollof + ROLLOF[i][j]
                        contatore_Rollof = contatore_Rollof + 1

                media_Rollof6 = somma_Rollof / contatore_Rollof




            if (filename == "gioia.wav"):

                derivata7 = delta(MFCC, 2)
                derivata_media7 = derivata1.mean()  # qui memorizzo la derivata media dell'audio

                somma_MFCC = 0
                contatore_MFCC = 0
                for i in range(0, MFCC.shape[0]):
                    for j in range(0, MFCC.shape[1]):
                        somma_MFCC = somma_MFCC + MFCC[i][j]
                        contatore_MFCC = contatore_MFCC + 1

                media_MFCC7 = somma_MFCC / contatore_MFCC

                somma_ZCR = 0
                contatore_ZCR = 0
                for i in range(0, ZCR.shape[0]):
                    for j in range(0, ZCR.shape[1]):
                        somma_ZCR = somma_ZCR + ZCR[i][j]
                        contatore_ZCR = contatore_ZCR + 1

                media_ZCR7 = somma_ZCR / contatore_ZCR

                # Ora prendo la media della Spectral Rollof per l'audio
                somma_Rollof = 0
                contatore_Rollof = 0
                for i in range(0, ROLLOF.shape[0]):
                    for j in range(0, ROLLOF.shape[1]):
                        somma_Rollof = somma_Rollof + ROLLOF[i][j]
                        contatore_Rollof = contatore_Rollof + 1

                media_Rollof7 = somma_Rollof / contatore_Rollof






for n_est in GDB_params['n_estimators']:
    for crit in GDB_params['criterion']:
        for max_d in GDB_params['max_depth']:
            for min_samples in GDB_params['min_samples_leaf']:
                for max_feat in GDB_params['max_features']:
                    clf = RandomForestClassifier(n_estimators=n_est,criterion=crit, max_depth=max_d, min_samples_leaf=min_samples, max_features=max_feat)
                    clf.fit(X_train, Y_train)
                    train_score = clf.score(X_train, Y_train)
                    test_score = clf.score(X_test, Y_test)


                    y_pred1 = clf.predict([[media_MFCC1, media_ZCR1, media_Rollof1, SPECTRAL_ENTROPY1, Energy1,
                                Pitch_max1, Pitch_mean1, Intensity_min1, Intensity_mean1, Intensity_max1,
                                Pitch_diff_max_mean1, derivata_media1]])


                    y_pred2 = clf.predict([[media_MFCC2, media_ZCR2, media_Rollof2, SPECTRAL_ENTROPY2, Energy2,
                            Pitch_max2, Pitch_mean2, Intensity_min2, Intensity_mean2, Intensity_max2,
                            Pitch_diff_max_mean2, derivata_media2]])


                    y_pred3 = clf.predict([[media_MFCC3, media_ZCR3, media_Rollof3, SPECTRAL_ENTROPY3, Energy3,
                            Pitch_max3, Pitch_mean3, Intensity_min3, Intensity_mean3, Intensity_max3,
                            Pitch_diff_max_mean3, derivata_media3]])


                    y_pred4 = clf.predict([[media_MFCC4, media_ZCR4, media_Rollof4, SPECTRAL_ENTROPY4, Energy4,
                                Pitch_max4, Pitch_mean4, Intensity_min4, Intensity_mean4, Intensity_max4,
                                Pitch_diff_max_mean4, derivata_media4]])


                    y_pred5 = clf.predict([[media_MFCC5, media_ZCR5, media_Rollof5, SPECTRAL_ENTROPY5, Energy5,
                            Pitch_max5, Pitch_mean5, Intensity_min5, Intensity_mean5, Intensity_max5,
                            Pitch_diff_max_mean5, derivata_media5]])


                    y_pred6 = clf.predict([[media_MFCC6, media_ZCR6, media_Rollof6, SPECTRAL_ENTROPY6, Energy6,
                            Pitch_max6, Pitch_mean6, Intensity_min6, Intensity_mean6, Intensity_max6,
                            Pitch_diff_max_mean6, derivata_media6]])


                    y_pred7 = clf.predict([[media_MFCC7, media_ZCR7, media_Rollof7, SPECTRAL_ENTROPY7, Energy7,
                            Pitch_max7, Pitch_mean7, Intensity_min7, Intensity_mean7, Intensity_max7,
                            Pitch_diff_max_mean7, derivata_media7]])


                    vettore_risultati = ["niente","niente","niente","niente","niente","niente","niente"]

                    if(y_pred1[0] == 5):
                        vettore_risultati[0] = "rabbia.wav"
                    else:
                        vettore_risultati[0] = "no"

                    if(y_pred2[0] == 6):
                        vettore_risultati[1] = "sorpresa.wav"
                    else:
                        vettore_risultati[1] = "no"

                    if(y_pred3[0] == 4):
                        vettore_risultati[2] = "paura.wav"
                    else:
                        vettore_risultati[2] = "no"

                    if(y_pred4[0] == 3):
                        vettore_risultati[3] = "mio_neutrale.wav"
                    else:
                        vettore_risultati[3] = "no"

                    if(y_pred5[0] == 7):
                        vettore_risultati[4] = "tristezza.wav"
                    else:
                        vettore_risultati[4] = "no"

                    if(y_pred6[0] == 1):
                        vettore_risultati[5] = "disgusto.wav"
                    else:
                        vettore_risultati[5] = "no"

                    if(y_pred7[0] == 2):
                        vettore_risultati[6] = "gioia.wav"
                    else:
                        vettore_risultati[6] = "no"



#                   print("For ({},{},{},{},{},{},{}) - train, test score: \t {:.5f} \t-\t {:.5f}".format(n_est,crit,max_d, min_samples,max_feat,train_score,test_score))
                    print("For ({},{},{},{},{}) - train, test score: \t {:.5f} \t-\t {:.5f}".format(n_est, crit, max_d,min_samples,max_feat,train_score,test_score))
                    print("Vettore risultati: ", vettore_risultati)
                    print("")





'''

'''
print("Miglioramenti random forest: ")
#Provo a migliorare il random forest e vedo i risultati


GDB_params = {
    'n_estimators': [10, 100, 500, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 100, 1000, None],
    'min_samples_leaf':[1, 2, 4],
    'max_features':["sqrt", "log2", None,"auto"]

}
'''

#df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_glass, y_col_glass, x_cols_glass, 0.7)


'''
for n_est in GDB_params['n_estimators']:
    for crit in GDB_params['criterion']:
        for max_d in GDB_params['max_depth']:
            for min_samples in GDB_params['min_samples_leaf']:
                for max_feat in GDB_params['max_features']:
                    clf = RandomForestClassifier(n_estimators=n_est,criterion=crit, max_depth=max_d, min_samples_leaf=min_samples, max_features=max_feat)
                    clf.fit(X_train, Y_train)
                    train_score = clf.score(X_train, Y_train)
                    test_score = clf.score(X_test, Y_test)
                    print("For ({},{},{},{},{}) - train, test score: \t {:.5f} \t-\t {:.5f}".format(n_est,crit,max_d, min_samples,max_feat,train_score,test_score))
'''

'''
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
'''


'''
print("classifier: ")
regressor2 = RandomForestClassifier(n_estimators=1000, random_state=0)
regressor2.fit(X_train, Y_train)

y_pred = regressor2.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
'''