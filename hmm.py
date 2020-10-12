from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix
import itertools
import os
import soundfile

def convert_to_16bit(path):
  for file in os.listdir(path):
    data, samplerate = soundfile.read(path+file)
    soundfile.write(path+file, data, samplerate, subtype='PCM_16')

def load(path):
  files = os.listdir(path)
  # convert_to_16bit(path)
  windows = []
  for file in files:

    sampling_freq, audio = wavfile.read(path+file)
    mfcc_features = mfcc(audio, sampling_freq)
    filterbank_features = logfbank(audio, sampling_freq)
    windows=windows+(np.split(mfcc_features, range(0,mfcc_features.shape[0], 5))[:-1])
    print(len(windows))
  return windows


path = "audio/cough/"
load(path)

class HMMTrainer(object):
  def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
    self.model_name = model_name
    self.n_components = n_components
    self.cov_type = cov_type
    self.n_iter = n_iter
    self.models = []
    if self.model_name == 'GaussianHMM':
      self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter)
    else:
      raise TypeError('Invalid model type')

  def train(self, X):
    np.seterr(all='ignore')
    self.models.append(self.model.fit(X))
    # Run the model on input data
  def get_score(self, input_data):
    return self.model.score(input_data)




