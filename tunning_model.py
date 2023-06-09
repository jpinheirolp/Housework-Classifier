import numpy as np

from time import time

import pandas as pd
from sync_lib import *
from functools import reduce

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from sklearn.neural_network import MLPClassifier


# Load data

train_input_ml_model_df = np.load('Generated Data/train_input_ml_model_df.npy',allow_pickle=True)
train_input_series = np.load('Generated Data/train_input_series.npy',allow_pickle=True)
test_input_ml_model_df = np.load('Generated Data/test_input_ml_model_df.npy',allow_pickle=True)
test_input_series = np.load('Generated Data/test_input_series.npy',allow_pickle=True)

np.random.seed(1)


X_train, X_test = train_input_ml_model_df, test_input_ml_model_df
y_train, y_test = train_input_series, test_input_series

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='adam', verbose=0, tol=1e-8, random_state=1,
                    learning_rate_init=.01)

N_TRAIN_SAMPLES = X_train.shape[0]
N_EPOCHS = 10
N_BATCH = 128
N_CLASSES = np.unique(y_train)

scores_train = []
scores_test = []

epoch = 0
while epoch < N_EPOCHS:
    random_perm = np.random.permutation(N_TRAIN_SAMPLES)
    mini_batch_index = 0
    while True:
        indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
        mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
        mini_batch_index += N_BATCH

        if mini_batch_index >= N_TRAIN_SAMPLES:
            break

    scores_train.append(mlp.score(X_train, y_train))
    scores_test.append(mlp.score(X_test, y_test))

    epoch += 1

fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(scores_train)
ax[0].set_title('Train')
ax[1].plot(scores_test)
ax[1].set_title('Test')
fig.suptitle("Accuracy over epochs", fontsize=14)
plt.show()
