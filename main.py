from python_speech_features import mfcc
import scipy.io.wavfile as wav
import tensorflow as tf

import csv
import re
import numpy as np
import librosa
from string import ascii_lowercase

def str_to_int(target_text):
    # space_token = ' '
    # end_token = '>'
    # blank_token = '%'
    # alphabet = list("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя") + ["'", space_token, end_token, blank_token]
    # char_to_index = {}
    # for idx, char in enumerate(alphabet):
    #     char_to_index[char] = idx
    #
    # y = []
    # for char in target_text:
    #     y.append(char_to_index[char])
    # return y

    space_token = ' '
    end_token = '>'
    blank_token = '%'
    alphabet = list(ascii_lowercase) + ["'", space_token, end_token, blank_token]
    char_to_index = {}
    for idx, char in enumerate(alphabet):
        char_to_index[char] = idx

    y = []
    for char in target_text:
        y.append(char_to_index[char])
    return y


def create_spectrogram(signals):
    '''
    function to create spectrogram from signals loaded from an audio file
    :param signals:
    :return:
    '''
    stfts = tf.signal.stft(signals, frame_length=200, frame_step=80, fft_length=256)
    spectrograms = tf.math.pow(tf.abs(stfts), 0.5)
    return spectrograms



def generate_input_from_audio_file(path_to_audio_file, resample_to=8000):
    '''
    function to create input for our neural network from an audio file.
    The function loads the audio file using librosa, resamples it, and creates spectrogram form it
    :param path_to_audio_file: path to the audio file
    :param resample_to:
    :return: spectrogram corresponding to the input file
    '''
    # load the signals and resample them
    signal, sample_rate = librosa.core.load(path_to_audio_file)
    if signal.shape[0] == 2:
        signal = np.mean(signal, axis=0)
    signal_resampled = librosa.core.resample(signal, sample_rate, resample_to)

    # create spectrogram
    X = create_spectrogram(signal_resampled)

    # normalisation
    means = tf.math.reduce_mean(X, 1, keepdims=True)
    stddevs = tf.math.reduce_std(X, 1, keepdims=True)
    X = tf.divide(tf.subtract(X, means), stddevs)
    return X



files = []
temps = []
audio = []
path = "C:/Users/dik19/Downloads/common-voice/"
with open(path+'cv-valid-dev.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    step=0
    for row in reader:
        if step == 100:
            break
        decod = re.sub(r'[,:;.—!?\-…]', "", row[1].encode('cp1251').decode('utf-8')).lower()
        # decod = re.sub(r'i', "і", decod)
        # decod = re.sub(r'x', "х", decod)
        # decod = re.sub(r'a', "а", decod)
        # decod = re.sub(r"’", "'", decod)
        print(decod)

        temps.append(tf.expand_dims(str_to_int(decod), axis=0))
        step = step + 1

        x = generate_input_from_audio_file(path+"cv-valid-dev/"+row[0].split(".")[0]+".wav")
        x = tf.expand_dims(x, axis=0)

        # (rate,sig) = wav.read(path+"cv-valid-dev/"+row[0].split(".")[0]+".wav")
        # x = tf.expand_dims(mfcc(sig,rate, numcep=16), axis=0)
        audio.append(x)


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=200, kernel_size=11,
                      strides=2, padding="valid",
                      activation="relu"),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True, activation='tanh'), backward_layer = tf.keras.layers.LSTM(200, return_sequences=True, go_backwards=True, activation='tanh')),
  tf.keras.layers.Dense(30),
])

optimizer = tf.keras.optimizers.Adam()

for i in range(len(temps)):
    for step in range(1, 10):
        with tf.GradientTape() as tape:
            x = audio[i]
            y = temps[i]
            logits = model(x)
            labels = y
            logit_length = [logits.shape[1]]*logits.shape[0]
            label_length = [labels.shape[1]]*labels.shape[0]
            loss = tf.nn.ctc_loss(
                        labels=labels,
                        logits=logits,
                        label_length=label_length,
                        logit_length=logit_length,
                        logits_time_major=False,
                        unique=None,
                        blank_index=-1,
                        name=None
                        )
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print('Epoch {}, Loss: {}'.format(str(i) + " " + str(step), loss))

        ctc_output = model(x)
        ctc_output = tf.nn.log_softmax(ctc_output)

        # space_token = ' '
        # end_token = '>'
        # blank_token = '%'
        # alphabet = list("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя") + ["'", space_token, end_token, blank_token]

        space_token = ' '
        end_token = '>'
        blank_token = '%'
        alphabet = list(ascii_lowercase) + ["'", space_token, end_token, blank_token]


        output_text = ''
        for timestep in ctc_output[0]:
            output_text += alphabet[tf.math.argmax(timestep)]
        print(output_text)