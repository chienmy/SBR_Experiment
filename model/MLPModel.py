from sklearn.neural_network import MLPClassifier

from model import Model


class MLPModel(Model):

    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=130, learning_rate="adaptive")

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict):
        pos_index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x_predict)[:, pos_index]
