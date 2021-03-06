from sklearn.neighbors import KNeighborsClassifier

from model import Model


class KNNModel(Model):

    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict):
        # 取出标签为1所属列
        pos_index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x_predict)[:, pos_index]
