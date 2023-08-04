import numpy as np

from time import time

import pandas as pd

from functools import reduce

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from sktime.classification.deep_learning.mlp import MLPClassifier


def main():

    # Load data

    train_input_ml_model_df = np.load('Generated Data/train_input_ml_model_df.npy',allow_pickle=True)
    train_input_series = np.load('Generated Data/train_input_series.npy',allow_pickle=True)
    test_input_ml_model_df = np.load('Generated Data/test_input_ml_model_df.npy',allow_pickle=True)
    test_input_series = np.load('Generated Data/test_input_series.npy',allow_pickle=True)

    np.random.seed(1)

    #DEBUGGING
    print("loaded",train_input_ml_model_df.shape,train_input_series.shape,test_input_ml_model_df.shape,test_input_series.shape)


    X_train, X_test = train_input_ml_model_df, test_input_ml_model_df
    y_train, y_test = train_input_series, test_input_series

    mlp = MLPClassifier(n_epochs=1, batch_size=16, activation='relu', optimizer='adam')

    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 15
    N_BATCH = 16
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []


    epoch = 0
    while epoch < N_EPOCHS:
        # random_perm = np.random.permutation(N_TRAIN_SAMPLES)
        # mini_batch_index = 0
        # while True:
        #     indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
        #     mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
        #     mini_batch_index += N_BATCH

        #     if mini_batch_index >= N_TRAIN_SAMPLES:
        #         break

        mlp = MLPClassifier(n_epochs=epoch + 14, batch_size=16, activation='relu', optimizer='adam')
        try:
            mlp.fit(X_train, y_train)
            scores_train.append(mlp.score(X_train, y_train))
            scores_test.append(mlp.score(X_test, y_test))
        except:
            break
        epoch += 1
        #DEBUGGING
        print("epoch",epoch)

    fig, ax = plt.subplots()
    ax.plot(scores_train, label='Train', color='blue')
    ax.plot(scores_test, label='Test', color='red')
    ax.set_title('Train and Test Accuracy')
    ax.legend()
    fig.suptitle("Accuracy over epochs", fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
