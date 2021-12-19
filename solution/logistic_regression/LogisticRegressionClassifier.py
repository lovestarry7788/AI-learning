import numpy as np
import pandas as pd
from math import exp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_iris

class LogisticRegressionClassifier:
    def __init__(self, max_iter=2000, learning_rate=0.01):
        self.max_iter = max_iter
        self.lr = learning_rate
    
    def sigmoid(self, x):
        return 1. / (1. + exp(-x))
    
    def score(self, X_test, y_test):
        

def create_logistic_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    X, y = data[:,:2], data[:,-1]
    return train_test_split(X, y, test_size=0.3)

if __name__ == "__main__":
    X_train , X_test , y_train , y_test = create_logistic_data()
    
    my_lr = LogisticRegressionClassifier()
    my_lr.fit(X_train, y_train)
    print("my LogisticRegression score", my_lr.score(X_test,y_test))

    sklearn_lr = LogisticRegression(max_iter = 2000)
    sklearn_lr.fit(X_train, y_train)
    print("sklearn LogisticRegression score", sklearn_lr.score(X_test, y_test))