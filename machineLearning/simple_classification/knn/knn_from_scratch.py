#Euclidean Distance = sqrt( ( (y2-y1)**2) + ((x2-x1)**2) )
#knn can work on both linear and non-linear datasets



import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
from math import sqrt
import os
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use('fivethirtyeight')
from collections import Counter


dataset = {'k': [[1,2], [2,3], [3,1]], 'r': [[5,6],[7,7],[8,6]]}
new_features = [5,7]


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        print("K is set to a value less than total voting value")
        
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence  = Counter(votes).most_common(1)[0][1] / k



    return vote_result, confidence



'''
result = k_nearest_neighbors(data=dataset,predict=new_features, k=1)
print(result)

for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0],j[1], s=100, color = i)

plt.scatter(new_features[0], new_features[1], s=100, marker='v',color=result)

plt.show()
'''

baseDir = os.path.abspath(__file__)
baseDir = os.path.dirname(baseDir)
targetDir = 'breast+cancer+wisconsin+original'
file_name = 'breast-cancer-wisconsin.data'
finalDir = os.path.join(baseDir, targetDir)

accurracies = []

for i in range(25):
    df = pd.read_csv(os.path.join(finalDir, file_name))
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)



    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2: [], 4:[]}
    test_set = {2: [], 4:[]}

    train_data = full_data[: -int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]


    for i in train_data:
        train_set[i[-1]].append(i[:-1]) # append data to trainset selecting i[-1] as the index for dict as i[-1] is the last column value: malign/benign and data is append(everything) except the last columns cus that's the label

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct =0
    total = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5) #every data point checked against the trianing set and a vote casted
            if group == vote:
                correct += 1

            total += 1

    #print(f'Accuracy: {correct/total}')
    accurracies.append(correct/total)

print(sum(accurracies)/len(accurracies))

