from sklearn.neural_network import MLPRegressor

from model import Model


class MLPModel(Model):

    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=130,learning_rate="adaptive")

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict):
        # pos_index = list(self.model.classes_).index(1)
        print(self.model.predict_proba(x_predict))
        return self.model.predict_proba(x_predict)

if __name__ == '__main__':
    X = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    y = [0, 0, 0, 1, 1, 1, 1 , 1 , 1]
    neigh = MLPModel()
    neigh.train(X , y)
    x_test = [[0.5] , [1.5] , [2.5] , [3.5] , [4.5] , [5.5]]
    print(neigh.predict(x_test))