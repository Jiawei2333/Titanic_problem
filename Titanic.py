import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

df1 = pd.read_csv(os.path.join(os.getcwd(), 'Titanic/train.csv'))
df2 = pd.read_csv(os.path.join(os.getcwd(), 'Titanic/test2.csv'))

columns_to_drop = ['Name', 'Ticket', 'Cabin']
columns_to_keep = [c for c in df2.columns if c not in columns_to_drop]


df_train = df1[columns_to_keep + ['Survived']].dropna()
df_test = df2[columns_to_keep].dropna()

## one-hot-encoding of `categorical` columns
columns_to_encode = ['Sex', 'Embarked'] # Change ['Gender', 'Cabin'] to ['Sex', 'Embarked']. Cabin is dropped before
df_train = pd.get_dummies(df_train, columns_to_encode)
df_test = pd.get_dummies(df_test, columns_to_encode)

y_train = df_train.pop('Survived') # pop removes the column and returns it
X_train = df_train.values
X_test = df_test.values

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# compute the probabilities (predict_proba)
# predict_proba gives two columns - probability of 0 and probability of 1
# to get probability of 1, we take second column  `[:,1]`
predictions_train = model.predict_proba(X_train)[:,1]
predictions_test = model.predict_proba(X_test)[:,1]

## Let's see what is the `goodness` of these predictions on train data
# look at roc_auc_score imported above
score = roc_auc_score(y_train, predictions_train)
print("AUC is {0}".format(score))

# But we should not test model accuracy on the same data that we trained on
# we should use cross_validation instead. The following will do a 5-fold
# cross validation - seem cross_val_score imported above

scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5)
mean, std = scores.mean(), scores.std()
print("AUC by cross validation is {0:.3f} +- {1:.3f}".format(mean, std))


# Decition Tree
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier().fit(X_train, y_train)
print("Acurracy is {0}".format(dt_clf.score(X_train, y_train)))
predictions_train_dr = dt_clf.predict_proba(X_train)[:,1]
score_dt = roc_auc_score(y_train, predictions_train_dr)
print("AUC of Decision Tree Classifier is {0}".format(score_dt))

scores_dt = cross_val_score(dt_clf, X_train, y_train, scoring='roc_auc', cv=5)
mean_dt, std_dt = scores_dt.mean(), scores_dt.std()
print("AUC of Decision Tree Classifier by cross validation is {0:.3f} +- {1:.3f}".format(mean, std))

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state = 0)
rf_clf.fit(X_train, y_train)

predictions_train_rf = rf_clf.predict_proba(X_train)[:,1]
score_rf = roc_auc_score(y_train, predictions_train_rf)
print("AUC of Random Forest Classifier is {0}".format(score_rf))

scores_rf = cross_val_score(rf_clf, X_train, y_train, scoring='roc_auc', cv=5)
mean_rf, std_rf = scores_rf.mean(), scores_rf.std()
print("AUC of Random Forest Classifier by cross validation is {0:.3f} +- {1:.3f}".format(mean_rf, std_rf))

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(max_depth = 2, learning_rate = 0.1, random_state = 0)
gb_clf.fit(X_train, y_train)

predictions_train_gb = gb_clf.predict_proba(X_train)[:,1]
score_gb = roc_auc_score(y_train, predictions_train_gb)
print("AUC of Gradient Boosting Classifier is {0}".format(score_gb))

scores_gb = cross_val_score(gb_clf, X_train, y_train, scoring='roc_auc', cv=5)
mean_gb, std_gb = scores_gb.mean(), scores_gb.std()
print("AUC of Random Forest Classifier by cross validation is {0:.3f} +- {1:.3f}".format(mean_gb, std_gb))
