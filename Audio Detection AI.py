#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[2]:


data = []
for root, directories, files in os.walk(os.path.join(os.getcwd(), 'cats_dogs')):
    for file in files[:100]:
        file_path = os.path.join(root, file)
              
        audio_binary = tf.io.read_file(file_path)
        waveform, _ = tf.audio.decode_wav(audio_binary)
        waveform = tf.reshape(waveform, [-1,])
        label = 0.0 if file.split('_')[0] == 'cat' else 1.0
        
        data.append([waveform, label])

data = np.array(data)
np.random.shuffle(data)


# In[3]:


rows = 3
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
for i, (waveform, label) in enumerate(data[:9]):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(waveform.numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    ax.set_title('cat' if not label else 'dog')

plt.show()


# In[4]:


spectrograms = []
max_len = max([waveform.shape[0] for waveform, label in data])

for waveform, label in data:
    zero_padding = tf.zeros(max_len - tf.shape(waveform), dtype=tf.float32)

    waveform = tf.cast(waveform, tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)
    spectrograms.append(spectrogram)


# In[5]:


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, max_len])

spectrogram_T = spectrogram.numpy().T
X = np.arange(max_len, step = max_len/spectrogram_T.shape[1])
Y = range(spectrogram_T.shape[0])

axes[1].set_title('Spectrogram')
axes[1].pcolormesh(X, Y, spectrogram_T)

plt.show()


# In[6]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2246, 129, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.summary()


# In[7]:


X = []
for spectrogram in spectrograms:
    spectrogram = tf.reshape(spectrogram, [spectrogram.shape[0], spectrogram.shape[1], 1])
    X.append(spectrogram.numpy())

X = np.array(X, dtype='float32')
y = data[:,1].astype('float32')


# In[8]:


history = model.fit(X, y, validation_split=0.2, epochs=10, callbacks=tf.keras.callbacks.EarlyStopping(patience=3))
