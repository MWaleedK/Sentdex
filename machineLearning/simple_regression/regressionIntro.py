import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn import model_selection #model_selction used instead of cross_validation same thing version changes
import matplotlib.pyplot as plt
import datetime
from matplotlib import style
import pickle
import os


style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


df = df[['Adj. Close','HL_PCT', 'PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df))) #predicting using 10 percent of the dataframe rows 0.1 is 10%

print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)


X =  np.array(df.drop(['label', 'Adj. Close'], axis=1))

X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[: -forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

classifier = LinearRegression(n_jobs=-1)
classifier = svm.SVR(kernel = 'poly')

classifier.fit(X_train, y_train)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'linearregression.pickle'), 'wb') as f:
    pickle.dump(classifier, f)

pickle_in = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'linearregression.pickle'), 'rb')
classifier = pickle.load(pickle_in)

accuracy = classifier.score(X_test, y_test)

#print(accuracy)

#To predict stuff based on the X data

forecast_set = classifier.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date =  datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()