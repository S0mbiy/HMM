import soundfile
from python_speech_features import mfcc
from scipy.io import wavfile
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix
import os

def convert_to_16bit(path):
    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
        filepath = os.path.join(subfolder, filename)
        data, samplerate = soundfile.read(filepath)
        soundfile.write(filepath, data, samplerate, subtype='PCM_16')

def evaluate(hmm_models, filepath):
    sampling_freq, audio = wavfile.read(filepath)
    mfcc_features = mfcc(audio, sampling_freq, nfft=2048)
    output_label = None
    max_score = -9999999999999999999
    for item in hmm_models:
       hmm_model, label = item
       score = hmm_model.get_score(mfcc_features)
       if score > max_score:
          max_score = score
          output_label = label
    return output_label

class HMMTrainer(object):
  def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
    self.model_name = model_name
    self.n_components = n_components
    self.cov_type = cov_type
    self.n_iter = n_iter
    self.models = []
    if self.model_name == 'GaussianHMM':
      self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type,n_iter=self.n_iter)
    else:
      raise TypeError('Invalid model type') 

  def train(self, X, lengths):
    np.seterr(all='ignore')
    self.models.append(self.model.fit(X, lengths))
    # Run the model on input data
  def get_score(self, input_data):
    return self.model.score(input_data)



hmm_models = []
input_folder = 'audio/'
# Parse the input directory
for dirname in os.listdir(input_folder):
    # Get the name of the subfolder
    subfolder = os.path.join(input_folder, dirname)
    if not os.path.isdir(subfolder):
        print("Subfolder doesn't exist: ", subfolder)
        continue
    convert_to_16bit(subfolder)
    # Extract the label
    label = subfolder[subfolder.rfind('/') + 1:]
    # Initialize variables
    X = np.array([])
    lengths = []
    # Iterate through the audio files (leaving 1 file for testing in each class)
    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
        # Read the input file
        filepath = os.path.join(subfolder, filename)
        sampling_freq, audio = wavfile.read(filepath)
        # Extract MFCC features
        mfcc_features = mfcc(audio, sampling_freq, nfft=2048)
        # Append to the variable X
        if len(X) == 0:
            X = mfcc_features
        else:
            X = np.append(X, mfcc_features, axis=0)
        lengths.append(len(mfcc_features))

    print('X.shape =', X.shape)
    # Train and save HMM model
    print('Beginning training')
    hmm_trainer = HMMTrainer(n_components=2)
    hmm_trainer.train(X, lengths)
    hmm_models.append((hmm_trainer, label))
    hmm_trainer = None
    print("Training finished")

input_folder = 'testing/'
real_labels = []
pred_labels = []
for dirname in os.listdir(input_folder):
  subfolder = os.path.join(input_folder, dirname)
  if not os.path.isdir(subfolder):
    print("Subfolder doesn't exist: ", subfolder)
    continue

  convert_to_16bit(subfolder)

  # Extract the label
  label_real = subfolder[subfolder.rfind('/') + 1:]

  for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
    real_labels.append(label_real)
    filepath = os.path.join(subfolder, filename)
    sampling_freq, audio = wavfile.read(filepath)
    mfcc_features = mfcc(audio, sampling_freq, nfft=2048)
    max_score = -9999999999999999999
    output_label = None

    for item in hmm_models:
       hmm_model, label = item
       score = hmm_model.get_score(mfcc_features)
       if score > max_score:
          max_score = score
          output_label = label
    pred_labels.append(output_label)

print("real ", real_labels)
print("pred ", pred_labels)

cm = confusion_matrix(real_labels, pred_labels)
print(cm)
print("Accuracy: ", (cm[0,0]+cm[1,1])/len(real_labels))

print(evaluate(hmm_models, "C:/Users/alvar/Downloads/audio.wav"))
