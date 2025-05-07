import win32com.client
import wave, array
import time
import asyncio
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
import cv2
import numpy as np
import pyglet
import speech_recognition as sr
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import pyAudioAnalysis.audioFeatureExtraction as AuF
from pyAudioAnalysis import audioBasicIO
from Praat_feature.prosodic_feature import PraatFeatures
import os
import librosa
import eyed3
import numpy
import matplotlib.pyplot as plt
import scipy
import matplotlib
import pitch
import magic
from python_speech_features import delta
import py2exe



class DummyAgent(Agent):
    class MyBehav(CyclicBehaviour):
        #async def on_start(self):
            #print("Starting behaviour . . .")



        async def run(self):
            video("loading.mp4")
            video("start.mp4")
            while True:
                video("still.mp4")
                ##################
                microfono()
                ##################
                prediction = classifier()
                # print("ciao")
                # os.remove("audio_file.wav")
                print(prediction)
                # print(type(prediction))
                ###################

                if prediction[0] == 1:
                    video("disgusto.mp4")
                if prediction[0] == 2:
                    video("gioia.mp4")
                if prediction[0] == 3:
                    video("neutrale.mp4")
                if prediction[0] == 4:
                    video("paura.mp4")
                if prediction[0] == 5:
                    video("rabbia.mp4")
                if prediction[0] == 6:
                    video("sorpresa.mp4")
                if prediction[0] == 7:
                    video("tristezza.mp4")

             #self.kill(exit_code="finished")
                #return
            #await asyncio.sleep(1)

        #async def on_end(self):
            #print("Behaviour: {}.".format(self.exit_code))

    async def setup(self):
        print("Agent starting . . .")
        self.my_behav = self.MyBehav()
        self.add_behaviour(self.my_behav)

if __name__ == "__main__":
    dummy = DummyAgent("ourserver@404.city", "ourserver")
    dummy.start()

    dummy.stop()


def classifier():
    with open('modello3.pickle', 'rb') as mp:
        regressor = pickle.load(mp)

    audio=open("audio_file.wav", "rb")
    make_stereo("audio_file.wav", "output.wav")

    ########
    feature = PraatFeatures()
    for filename in os.listdir():
        if (filename == "output.wav"):
            #print(filename= "output.wav")
            y, sr = librosa.load(filename)
            ZCR = librosa.feature.zero_crossing_rate(y=y)
            MFCC = librosa.feature.mfcc(y, sr)
            ROLLOF = librosa.feature.spectral_rolloff(y, sr)
            SPECTRAL_ENTROPY = AuF.stSpectralEntropy(y)  # Qui prendo il valore di Spectral_entropy per l'audio
            data = feature.get_all_sound_feature(filename)
            #print("data: ", data)

            x = data.keys()
            #print(x)
            y = -1
            for i in x:
                y = y + 1

                if (y == 0):
                    c = data.get(i).keys()
                    #print("c:", c)
                    a = -1
                    for j in c:
                        a = a + 1
                        if (a == 1):
                            #print("data[0].[1]: ", data.get(i).get(j))
                            Energy = data.get(i).get(j)  # qui prendo l'energia dell'audio

                # Ora prendo i valori del pitch massimo e del pitch medio per ogni audio
                if (y == 1):
                    #print("")
                    c = data.get(i).keys()
                    #print("c:", c)
                    a = -1
                    for j in c:
                        a = a + 1
                        if (a == 0):
                            #print("data[1].[0]: ", data.get(i).get(j))
                            Pitch_max = data.get(i).get(j)  # qui memorizzo il pitch massimo dell'audio

                        if (a == 1):
                            #print("data[1][1]: ", data.get(i).get(j))
                            Pitch_mean = data.get(i).get(j)  # qui memorizzo il pitch medio dell'audio

                        if (a == 5):
                            #print("data[1][5]: ", data.get(i).get(j))
                            Pitch_diff_max_mean = data.get(i).get(j)  # qui memorizzo il Pitch_diff_max_mean dell'audio

                if (y == 2):
                    #print("")
                    c = data.get(i).keys()
                    #print("c:", c)

                    a = -1
                    for j in c:
                        a = a + 1
                        if (a == 0):
                            #print("data[2].[0]: ", data.get(i).get(j))
                            Intensity_min = data.get(i).get(j)  # qui memorizzo l'intensity min dell'audio

                        if (a == 1):
                            #print("data[2][1]: ", data.get(i).get(j))
                            Intensity_mean = data.get(i).get(j)  # qui memorizzo l'intensity mean dell'audio

                        if (a == 2):
                            #print("data[2][2]: ", data.get(i).get(j))
                            Intensity_max = data.get(i).get(j)  # qui memorizzo l'intensity max dell'audio

            derivata = delta(MFCC, 2)
            derivata_media = derivata.mean()  # qui memorizzo la derivata media dell'audio
            #print("")
            #print("Derivata media: ", derivata_media)

            # Qui prendo la media dell'MFCC dell'audio
            somma_MFCC = 0
            contatore_MFCC = 0
            for i in range(0, MFCC.shape[0]):
                for j in range(0, MFCC.shape[1]):
                    somma_MFCC = somma_MFCC + MFCC[i][j]
                    contatore_MFCC = contatore_MFCC + 1

            media_MFCC = somma_MFCC / contatore_MFCC
            #print("")
            #print("MFCC: ", MFCC)
            #print("media_MFCC: ", media_MFCC)

            # Ora prendo la media della ZCR dell'audio
            somma_ZCR = 0
            contatore_ZCR = 0
            for i in range(0, ZCR.shape[0]):
                for j in range(0, ZCR.shape[1]):
                    somma_ZCR = somma_ZCR + ZCR[i][j]
                    contatore_ZCR = contatore_ZCR + 1

            media_ZCR = somma_ZCR / contatore_ZCR
            #print("")
            #print("ZCR: ", ZCR)
            #print("media_ZCR: ", media_ZCR)

            # Ora prendo la media della Spectral Rollof per l'audio
            somma_Rollof = 0
            contatore_Rollof = 0
            for i in range(0, ROLLOF.shape[0]):
                for j in range(0, ROLLOF.shape[1]):
                    somma_Rollof = somma_Rollof + ROLLOF[i][j]
                    contatore_Rollof = contatore_Rollof + 1

            media_Rollof = somma_Rollof / contatore_Rollof
            #print("")
            #print("ROLLOF: ", ROLLOF)
            #print("media_Rollof: ", media_Rollof)

            #print("")
            #print("Valore di spectral_entropy: ", SPECTRAL_ENTROPY)

    y_pred = regressor.predict([[media_MFCC, media_ZCR, media_Rollof, SPECTRAL_ENTROPY, Energy, Pitch_max, Pitch_mean,
                                 Intensity_min, Intensity_mean, Intensity_max, Pitch_diff_max_mean, derivata_media]])
    #print("Predizione:", y_pred)

    ######


    return y_pred



def  video(video_name):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_name)
    # If the input is the camera, pass 0 instead of the video file name
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('ESpeechPY', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    # cv2.destroyAllWindows()









def microfono():

    r = sr.Recognizer()  # riconoscitore
    mic = sr.Microphone()  # classe dei microfoni
    lista_mic = mic.list_microphone_names()  # lista dei microfoni
    #print(type(lista_mic[0]))
    #for i in range(0, len(lista_mic)):
        #  if(lista_mic[i][0:len("Microfono")]=="Microfono" or lista_mic[i][0:len("Mic in at front ")]=="Mic in at front " ):
        #print(lista_mic[i], i)
        # 5 o 9      11
    #mic_scelto = sr.Microphone(device_index=6)  # il numero(6 per Claudio -  1 per Michele) è l'indice nella lista dei mic del device
    #print('microfono mic_scelto')
    # acquisire l'audio
    with mic as source:
        print('listening...')
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        print('stop listening...')
        print('processing...')

        # salvataggio audio
    with open("audio_file.wav", "wb") as file:
        file.write(audio.get_wav_data())
        file.close()

        # classifier(audio.get_wav_data())

def make_stereo(file1: object, output: object) -> object:
    ifile = wave.open(file1)
    # (1, 2, 44100, 2013900, 'NONE', 'not compressed')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
    assert comptype == 'NONE'  # Compressed not supported yet
    array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
    left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
    ifile.close()

    stereo = 2 * left_channel
    stereo[0::2] = stereo[1::2] = left_channel

    ofile = wave.open(output, 'w')
    ofile.setparams((2, sampwidth, 48000, nframes, comptype, compname))
    ofile.writeframes(stereo.tostring())
    ofile.close()