#from sync_lib import *
#import sklearn as skl
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from nltk.util import ngrams
import numpy as np
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
#from tqdm import tqdm
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load data

train_input_ml_model_df = np.load('Generated Data/train_input_ml_model_df.npy',allow_pickle=True)
train_input_series = np.load('Generated Data/train_input_series.npy',allow_pickle=True)
test_input_ml_model_df = np.load('Generated Data/test_input_ml_model_df.npy',allow_pickle=True)
test_input_series = np.load('Generated Data/test_input_series.npy',allow_pickle=True)

print("loaded",train_input_ml_model_df.shape,train_input_series.shape,test_input_ml_model_df.shape,test_input_series.shape)


n_samples_train = train_input_series.shape[0]
n_samples_test = test_input_series.shape[0]

input_x_train = train_input_ml_model_df
input_y_train = train_input_series

input_x_test = test_input_ml_model_df
input_y_test = test_input_series


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
    #predicted_acttivitys.tofile(f'model_output/output{previous_checkpoint}_{checkpoint}.txt',sep=",")
    np.save(f'model_output/output{previous_checkpoint}_{checkpoint}.npy',predicted_acttivitys)
    previous_checkpoint = checkpoint
    #print(predicted_acttivitys)
    

knn_dtw_classifier.reset()

# '''

#'''

# output_file_names = ['output0_796.npy', 'output796_1592.npy', 'output1592_2388.npy', 'output2388_3184.npy', 'output3184_3980.npy', 'output3980_4776.npy', 'output4776_5572.npy', 'output5572_6368.npy', 'output6368_7164.npy', 'output7164_7960.npy']
output_file_names = ['output0_223.npy', 'output223_446.npy', 'output446_669.npy', 'output669_892.npy', 'output892_1115.npy', 'output1115_1338.npy', 'output1338_1561.npy', 'output1561_1784.npy', 'output1784_2007.npy', 'output2007_2230.npy']
print(output_file_names)

output_labels = np.array([])
for file_name in output_file_names:
    output_labels = np.append(output_labels,np.load(f'model_output/{file_name}',allow_pickle=True))
print(output_labels.shape)
print(input_y_test.shape)

# Calculating the accuracy of classifier
print(f"Accuracy of the classifier is: {accuracy_score(input_y_test, output_labels)}")

labels = ["Aera", "AS1", "Asp", "Bougie", "BricoC", "BricoP", "Nett", "Oeuf", "Saber", "SdB"]

cm = confusion_matrix(input_y_test, output_labels,labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot()
ax = plt.gca()
ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=90)
plt.savefig('confusion_matrix.png')
print(cm)


#'''

# next steps: compare one day for train another for test: being close days
# them make the same thing for separate days with a long distance in time
# after make a kind of cross validation with a sliding window through all the data
# where for example the train is 5 day and the test is the next 3 days

    # Actual Acuracy test 

# make graphs to compare the predictions to the actual results



