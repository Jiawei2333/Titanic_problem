import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df1 = pd.read_csv(os.path.join(os.getcwd(), 'Titanic/train.csv'))
df_train = df[['Survived', 'Age']].dropna()

df2 = pd.read_csv(os.path.join(os.getcwd(), 'Titanic/test2.csv'))
df_test = df2[['PassengerId', 'Age']].dropna()

X_train = np.array(df_train['Age'])
X_train = X_train.reshape(X_train.shape[0],1)
y_train = np.array(df_train['Survived']).astype(np.int)
# y_train = y_train.reshape(y_train.shape[0],1)

model = LogisticRegression()
model.fit(X_train, y_train)

X_test = np.array(df_test['Age'])
X_test = X_test.reshape(X_test.shape[0],1)

predictions = model.predict(X_test)

plt.plot(X_train, y_train, 'ro')
predictions