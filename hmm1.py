from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix
from matplotlib.mlab import psd
import itertools
import os
import soundfile


def convert_to_16bit(path):
    for file in os.listdir(path):
        data, samplerate = soundfile.read(path + file)
        soundfile.write(path + file, data, samplerate, subtype='PCM_16')


def load(path):
    files = os.listdir(path)
    # convert_to_16bit(path)
    X = []
    lengths = []
    for file in files:

        sampling_freq, audio = wavfile.read(path + file)
        mfcc_features = mfcc(audio, sampling_freq, highfreq=22000, nfft=2048)
        X.extend(get_emmissions(mfcc_features))
        lengths.append(len(mfcc_features))
    return X, lengths

def get_emmissions(mfcc_features):
  X = []
  for obs in mfcc_features:
    # x = [sum(obs[:3]), sum(obs[3:7]), sum(obs[7:])]
    x = [sum(obs[:2]), sum(obs[2:5]), sum(obs[5:7]), sum(obs[7:])]
    # x = [sum(obs[:2]), sum(obs[2:4]), sum(obs[4:6]), sum(obs[6:8]), sum(obs[8:])]
    greater = x[0]
    emission = 0
    for i, val in enumerate(x):
      if val > greater:
          greater = val
          emission = i
    X.append([emission])
  return X

if __name__ == '__main__':
    path = "audio/cough/"
    X, lengths = load(path)
    # detects audio with only cough
    # remodel = hmm.GaussianHMM(n_components=5, covariance_type="tied", n_iter=1000, init_params="mcs")
    # covers more sounds with cough in it
    # remodel = hmm.GaussianHMM(n_components=3, covariance_type="tied", n_iter=1000, init_params="mcs")
    # super sensible
    remodel = hmm.GaussianHMM(n_components=2, covariance_type="tied", n_iter=1000, init_params="mcs")
    # remodel.transmat_ = np.array([[0, 1, 0, 0, 0],
    #                              [0, 0, 1, 0, 0],
    #                              [0, 0, 0, 1, 0],
    #                              [.6, 0, 0, 0, .4],
    #                              [.1, 0, 0, 0, .9]])

    remodel.fit(X, lengths)
    dist = remodel.get_stationary_distribution()


    test_path = 'audio/maybe cough/'
    files = os.listdir(test_path)
    # convert_to_16bit(test_path)
    for file in files:
        sampling_freq, audio = wavfile.read(test_path + file)
        mfcc_features = mfcc(audio, sampling_freq, highfreq=22000, nfft=2048)
        score = remodel.score(get_emmissions(mfcc_features))
        print(file,score)
        print("Cough" if score>-1300 else "Not Cough")
