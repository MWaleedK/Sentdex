#Euclidean Distance = sqrt( ( (y2-y1)**2) + ((x2-x1)**2) )



import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
from math import sqrt
import os
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
from collections import Counter




baseDir = os.getcwd()
targetDir = 'breast+cancer+wisconsin+original'
file_name = 'breast-cancer-wisconsin.data'
finalDir = os.path.join(baseDir, targetDir)

df = pd.read_csv(os.path.join(finalDir, file_name))


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



    return vote_result

result = k_nearest_neighbors(data=dataset,predict=new_features, k=1)
print(result)

for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0],j[1], s=100, color = i)

plt.scatter(new_features[0], new_features[1], s=100, marker='v',color=result)

plt.show()
