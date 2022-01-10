import pdb
import numpy as np
import matplotlib.pyplot as plt
# from CARTRegression import CARTRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from CART import CART

class GBDT():

    def __init__(self, max_tree_num=3):
        self.tree_list = []
        self.max_tree_num = 3
    
    def fit(self, x, y):
        residual = y
        for i in range(self.max_tree_num):
            model = CART()
            model.fit(x, residual)
            self.tree_list.append(model)
            prediction = model.predict(x)
            residual = residual - prediction

    def predict(self, x):
        y = np.zeros(x.shape[0])
        for model in self.tree_list:
            new_pred = np.array(model.predict(x))
            y += new_pred
            print('new_pred', new_pred)
        return y

def Dataprocessing():
    data = load_iris()
    x = data['data']
    y = data['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    """
    print("train:")
    print(x_train)
    print(y_train)
    print("test:")
    print(x_test)
    print(y_test)
    """
    return x_train, x_test, y_train, y_test

def MSE(x, y):
    r = 0.
    for i in range(len(x)):
        r += (x[i] - y[i]) ** 2
    return r

if __name__ == "__main__":

    x_train, x_test, y_train, y_test = Dataprocessing()

    model = GBDT(max_tree_num = 3)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    res = MSE(y_pred, y_test)

    print("残差平方和: %f"%res)

    plt.plot(y_test)
    plt.plot(y_pred)
    plt.show()

