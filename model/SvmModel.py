from sklearn.decomposition import PCA
from sklearn.svm import SVC

from model import Model


class SvmModel(Model):

    def __init__(self):
        self.model = SVC(kernel='linear', probability=True)
        self.pca = PCA(n_components=200, svd_solver="auto")

    def train(self, x_train, y_train):
        if min(len(x_train[0]), len(x_train)) > self.pca.components_:
            x_train = self.pca.fit_transform(x_train, y_train)
        self.model.fit(x_train, y_train)

    def predict(self, x_predict):
        if len(x_predict[0]) != self.model.shape_fit_[1]:
            x_predict = self.pca.transform(x_predict)
        pos_index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x_predict)[:, pos_index]
