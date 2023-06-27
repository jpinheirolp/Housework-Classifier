#from sync_lib import *
#import sklearn as skl
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from nltk.util import ngrams

import numpy as np
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.deep_learning.mlp import MLPClassifier
#from tqdm import tqdm
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool, cv


# Load data

train_input_ml_model_df = np.load('Generated Data/train_input_ml_model_df.npy',allow_pickle=True)
train_input_series = np.load('Generated Data/train_input_series.npy',allow_pickle=True)
test_input_ml_model_df = np.load('Generated Data/test_input_ml_model_df.npy',allow_pickle=True)
test_input_series = np.load('Generated Data/test_input_series.npy',allow_pickle=True)

print("loaded",train_input_ml_model_df.shape,train_input_series.shape,test_input_ml_model_df.shape,test_input_series.shape)



n_samples_train = train_input_series.shape[0]
n_samples_test = test_input_series.shape[0]

input_x_train = train_input_ml_model_df.reshape((n_samples_train,train_input_ml_model_df.shape[2]))
input_y_train = train_input_series

input_x_test = test_input_ml_model_df.reshape((n_samples_test,test_input_ml_model_df.shape[2]))
input_y_test = test_input_series

print("reshaped",input_x_train.shape,input_y_train.shape,input_x_test.shape,input_y_test.shape)

class_weights_dict = {}
class_weights_list = []


class_elements, class_counts = np.unique(input_y_train, return_counts=True)
print("class_elements",class_elements)
print("class_counts",class_counts)
for i in range(len(class_counts)):
    class_weights_dict[class_elements[i]] = ( 1 - (class_counts[i] / n_samples_train) )**2
    class_weights_list.append(1 - (class_counts[i] / n_samples_train))

class_weights_list = []



# def binarize_samples(classifiers_class):
#     class_returned = classifiers_class
#     if classifiers_class != "AS1":
#         class_returned = "Not_AS1"
#     return class_returned

# vectorizer_binarizer_samples = np.vectorize(binarize_samples)
# input_y_train_binarized = vectorizer_binarizer_samples(input_y_train)
# input_y_test_binarized = vectorizer_binarizer_samples(input_y_test)

# '''
classifier = CatBoostClassifier(
                        loss_function='MultiClass',
                        iterations=300,
                        depth=10,
                        #    learning_rate=1,
                        verbose=True,
                        class_weights=class_weights_dict
                        )

# classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1 , distance="wddtw", algorithm = "brute")
# classifier = MLPClassifier(n_epochs=20, batch_size=4, activation='relu', optimizer='adam')

print("class_weight", len(class_weights_dict))

start_time = time()
classifier.fit(Pool(input_x_train, input_y_train))
end_time = time()
print("time to fit "+ str(n_samples_train),end_time - start_time)


# start_time = time()
# predicted_acttivitys = classifier.predict(input_x_test)#input_x_test[-1:])
# end_time = time()
print("time to predict "+ str(n_samples_train),end_time - start_time)

len_test = input_x_test.shape[0]


output_labels = classifier.predict(Pool(input_x_test))
classifier.save_model("./catboost_classifier", format='cbm')

# classifier.reset()


# enhanced_output_labels = output_labels[:]

# for i in range(enhanced_output_labels.shape[0]):
#     if output_labels[i] == "Not_AS1" and input_y_test[i] != "AS1":
#         enhanced_output_labels[i] = input_y_test[i]

# classifier_2 = KNeighborsTimeSeriesClassifier(n_neighbors=1 , distance="wddtw", algorithm = "brute")

# input_x_test_without_AS1 = input_x_test[:]

# indexes_to_delete = []
# deleted_items_input_y_test = []
# deleted_output_labels = []

# for i in range(output_labels.shape[0]):
#     if output_labels[i] == "AS1":
#         indexes_to_delete.append(i)

# deleted_items_input_y_test = np.array(input_y_test[indexes_to_delete])
# deleted_output_labels = np.array(["AS1"]*len(indexes_to_delete))

# input_x_test_without_AS1 = np.delete(input_x_test_without_AS1, indexes_to_delete, axis=0)
# input_y_test_without_AS1 = np.delete(input_y_test, indexes_to_delete, axis=0)

# indexes_to_delete = []
# for i in range(input_y_train.shape[0]):
#     if input_y_train[i] == "AS1":
#         indexes_to_delete.append(i)

# input_x_train = np.delete(input_x_train, indexes_to_delete, axis=0)
# input_y_train = np.delete(input_y_train, indexes_to_delete, axis=0)

# classifier_2.fit(input_x_train, input_y_train)


# output_labels = classifier_2.predict(input_x_test_without_AS1)

# output_labels = np.concatenate((output_labels,deleted_output_labels),axis=0)
# input_y_test_without_AS1 = np.concatenate((input_y_test_without_AS1,deleted_items_input_y_test),axis=0)


# classifier_2.reset()

#'''

#'''

# Calculating the accuracy of classifier
print(f"Accuracy of the classifier is: {accuracy_score(input_y_test, output_labels)}")


labels = ["Aera", "AS1", "Asp", "Bougie", "BricoC", "BricoP", "Nett", "Oeuf", "Saber", "SdB", "no_activity"]#, "Not_AS1"]

cm = confusion_matrix(input_y_test, output_labels,labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot()
ax = plt.gca()
ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=90)
cm_title = "acuracy: "+str(accuracy_score(input_y_test, output_labels)) + " recall: " + str(recall_score(input_y_test, output_labels,average='macro'))
disp.ax_.set_title(cm_title) #+ "roc_auc_score: " + str(roc_auc_score(input_y_test, output_labels,average='samples', multi_class='ovr'))
plt.savefig('confusion_matrix.png')
print(cm)


#''' 

# in catboost put different weights in each class prioratizing the least frenquent classes, accordingly to the medium article

# !!! try to penalize false negatives related to AS1 class: it looks like the data from the beggining of the activitys is always classified as AS1
# !!! make so that every class has about the same number of samples



