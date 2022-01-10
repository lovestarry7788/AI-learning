import pdb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Node:

    def __init__(self, best_feature_id, best_split_value):
        self.feature_id = best_feature_id
        self.split_value = best_split_value
        self.left_child = None
        self.right_child = None
        self.ymean = None
    

class CART:

    def __init__(self, max_depth = 10):
        self.root = Node(-1, -1)
        self.tree_size = 0
        self.feature_num = 0
        self.max_depth = max_depth
    
    def get_mse(self, x, y, id_list,id, feature_id):
        best_split_value = x[id][feature_id]
        left_id_list,right_id_list = [], []
        left_y_sum, right_y_sum = 0, 0
        for id in id_list:
            if x[id][feature_id] <= best_split_value:
                left_id_list.append(id)
                left_y_sum += y[id]
            else:
                right_id_list.append(id)
                right_y_sum += y[id]
        
        left_y_mean = left_y_sum / len(left_id_list) if len(left_id_list) > 0 else 0
        right_y_mean = right_y_sum / len(right_id_list) if len(right_id_list) > 0 else 0
        left_mse , right_mse = 0, 0
        for id in left_id_list:
            left_mse += (left_y_mean - y[id]) ** 2
        for id in right_id_list:
            right_mse += (right_y_mean - y[id]) ** 2
        return left_mse + right_mse
            

    def build_tree(self, x, y, id_list, parent_node, dir, depth):
        # pdb.set_trace()
        if len(id_list) == 0:
            return
        elif len(id_list) == 1 or depth > self.max_depth:
            ymean = 0.0
            for id in id_list:
                ymean = ymean + y[id]
            ymean = ymean / len(id_list)
            parent_node.ymean = ymean
            return
        else:
            best_feature_id, best_split_value = 0, 0
            best_mse = -1
            for id in id_list:
                for feature_id in range(self.feature_num):
                    this_mse = self.get_mse(x, y, id_list, id, feature_id)
                    if best_mse == -1 or this_mse < best_mse:
                        best_mse = this_mse
                        best_feature_id = feature_id
                        best_split_value = x[id][feature_id]
            self.tree_size += 1

            left_id_list , right_id_list = [] , []
            for id in id_list:
                if x[id, best_feature_id] <= best_split_value:
                    left_id_list.append(id)
                else:
                    right_id_list.append(id)
            this_node = Node(best_feature_id, best_split_value)
            if dir==0 : parent_node.left_child = this_node
            else: parent_node.right_child = this_node

            print(best_feature_id)
            print(best_split_value)
            print(left_id_list)
            print(right_id_list)
            

            self.build_tree(x, y, left_id_list, this_node, 0, depth + 1)
            self.build_tree(x, y, right_id_list, this_node, 1, depth + 1)

    def search_in_tree(self, feature, node):
        while node.left_child or node.right_child:
            if feature[node.feature_id] <= node.split_value :
                node = node.left_child
            else:
                node = node.right_child
            if node == None:
                return -1
        return node.ymean

    def predict(self, x):
        pred_list = []
        for id in range(x.shape[0]):
            pred = self.search_in_tree(x[id], self.root.right_child)
            pred_list.append(pred)
        return pred_list

    def fit(self, x, y):
        id_list = [i for i in range(len(x))]
        self.root = Node(-1, -1)
        self.feature_num = len(x[0])
        self.build_tree(x, y, id_list, self.root, 1, 0)


def Dataprocessing():
    data = load_iris()
    x = data['data']
    y = data['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    return x_train, x_test, y_train, y_test

def MSE(x, y):
    r = 0.
    for i in range(len(x)):
        r += (x[i] - y[i]) ** 2
    return r

if __name__ == "__main__":

    x_train, x_test, y_train, y_test = Dataprocessing()

    model = CART()
    
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    print("y_pred",y_pred)
    print("y_test",y_test)
    # res = MSE(y_pred, y_test)

    # print("残差平方和: %f"%res)

    # plt.plot(y_test)
    # plt.plot(y_pred)
    # plt.show()