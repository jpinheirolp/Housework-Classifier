#from sync_lib import *
#import sklearn as skl
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from nltk.util import ngrams
import numpy as np
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
#from tqdm import tqdm
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score



# Load data

input_ml_model_df = np.load('Generated Data/input_ml_model_df.npy',allow_pickle=True)
input_series = np.load('Generated Data/input_series.npy',allow_pickle=True)
#input_series.tofile('Generated Data/input_series_readable.txt',sep=",")


n_samples = input_series.shape[0]



print("loaded",input_ml_model_df.shape,input_series.shape)


percentage_train = 70
n_samples_train = int((n_samples*percentage_train) / 100)



input_x_test = input_ml_model_df[n_samples_train:]
input_y_test = input_series[n_samples_train:]

percentage_train = 15
n_samples_train = int((n_samples*percentage_train) / 100)

input_x_train = input_ml_model_df[:n_samples_train]
input_y_train = input_series[:n_samples_train]

#'''
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

output_file_names = [ 'output0_954.npy', 'output954_1909.npy', 'output1909_2863.npy', 'output2863_3818.npy', 'output3818_4772.npy',
 'output4772_5727.npy', 'output5727_6681.npy', 'output6681_7636.npy', 'output7636_8590.npy', 'output8590_9545.npy']


output_labels = np.array([])
for file_name in output_file_names:
    output_labels = np.append(output_labels,np.load(f'model_output/{file_name}',allow_pickle=True))
print(output_labels.shape)
print(input_y_test.shape)

# Calculating the accuracy of classifier
print(f"Accuracy of the classifier is: {accuracy_score(input_y_test, output_labels)}")

print(confusion_matrix(input_y_test, output_labels))
    # Actual Acuracy test 
# 
# shuffle the data by joining the two arrays shuffling and then splitting them again :D 
# split into 70% train and 30% test 
# then run the model by making it save every 1000 predictions in a different file 
# make graphs to compare the predictions to the actual results






