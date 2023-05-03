#from sync_lib import *
#import sklearn as skl
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from nltk.util import ngrams
import numpy as np
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
#from tqdm import tqdm
from time import time

# Load data
# mabe you made shit while shuffling the data

input_ml_model_df = np.load('Generated Data/input_ml_model_df.npy',allow_pickle=True)
input_series = np.load('Generated Data/input_series.npy',allow_pickle=True)
#input_series.tofile('Generated Data/input_series_readable.txt',sep=",")

shape_array = (29115, 4, 300)
n_samples = input_series.shape[0]

# Create the 3D array with random values
#input_ml_model_df = np.random.rand(*shape_array).astype(np.float32)

print("loaded",input_ml_model_df.shape,input_series.shape)


percentage_train = 1
n_samples_train = int((n_samples*percentage_train) / 100)



input_x_test = input_ml_model_df[29102:]#[n_samples_train:]

input_x_train = input_ml_model_df[:n_samples_train]
input_y_train = input_series[:n_samples_train]

knn_dtw_classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1 , distance="dtw",algorithm = "brute")

start_time = time()
knn_dtw_classifier.fit(input_x_train, input_y_train)
end_time = time()
print("time to fit "+ str(n_samples_train),end_time - start_time)


start_time = time()
predicted_acttivitys = knn_dtw_classifier.predict(input_x_test[-1:])
end_time = time()
print("time to predict "+ str(n_samples_train),end_time - start_time)

len_test = input_x_test.shape[0]

n_checkpoints = 10
previous_checkpoint = 0
checkpoint = 0
for i in range(1, n_checkpoints+1):
    checkpoint = int(len_test * ( i / n_checkpoints) )
    print("checkpoint",checkpoint,previous_checkpoint)
    predicted_acttivitys = knn_dtw_classifier.predict(input_x_test[previous_checkpoint:checkpoint])
    predicted_acttivitys.tofile(f'model_output/output{previous_checkpoint}_{checkpoint}.txt',sep=",")
    np.save(f'model_output/output{previous_checkpoint}_{checkpoint}.npy',predicted_acttivitys)
    previous_checkpoint = checkpoint
    #print(predicted_acttivitys)
    
print(input_series[29102:])
knn_dtw_classifier.reset()




# ''' 
# Actual Acuracy test 
# 
# shuffle the data by joining the two arrays shuffling and then splitting them again :D 
# split into 70% train and 30% test 
# then run the model by making it save every 1000 predictions in a different file 
# make graphs to compare the predictions to the actual results






