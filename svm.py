import matplotlib.pyplot as plt
import sklearn
import numpy as np
import statistics as st
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,learning_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,f1_score
import pickle

data = pd.read_csv("data/four.csv")

w = len(data.columns.values.tolist())-2
for i in range(2,w+1):
    mean = st.mean(data.iloc[:, i].values.tolist())
    std = st.stdev(data.iloc[:,i].values.tolist())
    data.iloc[:,i] = (data.iloc[:,i] - mean) / std

X = data.loc[:,'Gender':'Interviews']
y = data.loc[:,'Company Placed']

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
# svclassifier=SVC(kernel='rbf',gamma=0.001,C=100)
# svclassifier.fit(X_train,y_train)
#
#
# y_pred=svclassifier.predict(X_test)
# print(accuracy_score(y_test,y_pred))

#pickle.dump(svclassifier,open('prediction-svm.sav','wb'))

def plot_learning_curve(estimator, title, X, y, cv,
                        n_jobs, train_sizes):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

title="Learning Curve(SVM)"
#title="Learning Curve(Logistic Regression)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator=SVC(kernel='rbf',gamma=0.001,C=100)
#estimator=LogisticRegression()
train_size=np.linspace(.1, 1.0, 5)
plot_learning_curve(estimator, title, X, y, cv, 4, train_size)

plt.show()