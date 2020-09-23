# Load in packages and dataset from CWRU
import numpy as np
import matplotlib.pyplot as plt
import cwru
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
data = cwru.CWRU("12FanEndFault", 2048, 0.8, 1, '1797','1750')

# Reshape dataset to whole 
all_data = np.vstack((data.X_train, data.X_test))
all_labels = np.hstack((data.y_train, data.y_test))

# Construct time domain features
mean = np.mean(all_data, axis = 1)
std = np.std(all_data, axis = 1)
var = np.var(all_data, axis = 1)
median = np.median(all_data, axis = 1)
maximum = np.max(all_data, axis = 1)
minimum = np.min(all_data, axis = 1)
range_distance = maximum - minimum

# Construct frequency domain

# with window of 2048
fft_data = np.fft.fft(all_data)
# Signal power
p = sp.sum(fft_data*fft_data, 1)/fft_data.size
# Signal energy
e = p*fft_data.size

# Create final dataset
final_data = np.concatenate([mean,std,var,median,maximum,minimum,range_distance,p,e], axis = 1)
all_labels.shape

# Make train/test split
X_train, X_test, y_train, y_test = train_test_split(final_data, all_labels, test_size=0.2035, shuffle=False)

# Initialize SVM on dataset with extracted features
parameters = {'kernel':('linear', 'rbf'), 'C':[1,5,10,15,20,25,30,40], 'gamma':[0.00001, 0.0001,0.001,0.01]}
svc = svm.SVC(gamma='scale')
clf = GridSearchCV(svc, parameters, cv=10, verbose = 100, n_jobs = 4)

# Fit the model using the .real part of the dataset (this is created by using the FFT algorithm) 
clf.fit(X_train.real, y_train)
sorted(clf.cv_results_.keys())
clf2 = clf.best_estimator_
predictions = clf2.predict(X_test.real)

# Print classification report and create confusion matrix
print(classification_report(y_test, predictions))
mat = confusion_matrix(y_test, predictions)
plt.figure(figsize=(12,12))
sns.heatmap(mat.T, square=True, annot=True, fmt='d')
plt.xlabel('true label')
plt.ylabel('predicted label');

# See which fitted SVM was the best estimator
clf.best_estimator_