from sklearn.ensemble import RandomForestClassifier

from model import Model


class RFModel(Model):

    def __init__(self):
        self.model = RandomForestClassifier(oob_score=True, n_estimators=30)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_predict):
        pos_index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x_predict)[:, pos_index]
