from sklearn.neighbors import KNeighborsClassifier

from model import Model


class KNNModel(Model):

    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict):
        # 取出标签为1所属列
        pos_index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x_predict)[:, pos_index]

if __name__ == '__main__':
    X = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    y = [0, 0, 0, 1, 1, 1, 1 , 1 , 1]
    neigh = KNNModel()
    neigh.train(X , y)
    x_test = [[0.5] , [1.5] , [2.5] , [3.5] , [4.5] , [5.5]]
    print(neigh.predict(x_test))
