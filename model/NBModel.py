from sklearn.naive_bayes import GaussianNB

from model import Model


class NBModel(Model):

    def __init__(self):
        self.model = GaussianNB()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict):
        pos_index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x_predict)[:, pos_index]
