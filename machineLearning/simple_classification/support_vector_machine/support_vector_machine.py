# SVM is a binary classifier
# find the best separating hyperplane and work with the intuition from there

#Using the same code we used for knn

import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd
import os

baseDir = os.path.dirname(os.path.abspath(__file__))

targetDir = 'breast+cancer+wisconsin+original'
file_name = 'breast-cancer-wisconsin.data'
finalDir = os.path.join(baseDir, targetDir)

df = pd.read_csv(os.path.join(finalDir, file_name))
df.replace('?', -99999, inplace=True)

df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(['class'],axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

classifier = svm.SVC(kernel='linear') #kernell set to linear since this dataset does not fare well when increasing model flexibility
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)


print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures.shape),-1)

prediction = classifier.predict(example_measures)

print(prediction)

