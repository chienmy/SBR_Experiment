import pandas as pd
import math
import warnings

from sklearn.feature_extraction.text import CountVectorizer
from model import RFModel, SvmModel, MLPModel
from .dimension_reduce import selectFromLinearSVC2
from .Sample import Sample
from .utilities import *

warnings.filterwarnings('ignore')


class SupervisedLearner:

    model_dict = {
        "svm": SvmModel(),
        "rf": RFModel(),
        "mlp": MLPModel()
    }

    def __init__(self, train_data_path: str, test_data_path: str, model: str):
        self.sample_list = []
        self.model = SupervisedLearner.model_dict[model]

        if train_data_path == test_data_path:
            df = pd.read_csv(train_data_path)
            df = df.sample(frac=1)
            num_n = len(df)
            self.train_dataset = df[:int(0.5 * num_n)]
            self.test_dataset = df[int(0.5 * num_n):]
        else:
            self.train_dataset = pd.read_csv(train_data_path)
            self.test_dataset = pd.read_csv(test_data_path)
        self.positive_num = 0
        for line in self.test_dataset.itertuples():
            if line.security == 1:
                self.positive_num += 1

    def clear(self):
        self.sample_list = []

    def clear_all(self):
        self.sample_list = []
        self.positive_num = 0
        self.model = RFModel()
        self.train_dataset = []
        self.test_dataset = []

    def run(self):

        global train_x
        global test_x
        global train_y
        global test_y

        train_content = self.train_dataset.description
        train_label = self.train_dataset.security.tolist()

        test_content = self.test_dataset.description
        test_label = self.test_dataset.security.tolist()

        vectorizer = CountVectorizer(stop_words='english')
        train_content_matrix = vectorizer.fit_transform(train_content)
        test_content_matrix = vectorizer.transform(test_content)
        train_content_matrix_dr, test_content_matrix_dr = selectFromLinearSVC2(train_content_matrix, train_label,
                                                                                  test_content_matrix)

        train_x = train_content_matrix_dr.toarray()
        train_y = train_label
        test_x = test_content_matrix_dr.toarray()
        test_y = test_label
        data_transfer(train_x, train_y, test_x, test_y)

        self.model.train(train_x, train_y)
        predicted = self.model.predict(test_x)
        predicted = list(predicted)
        index = 0
        for line in self.test_dataset.itertuples():
            sample = Sample(line.id, line.security, predicted[index])
            self.sample_list.append(sample)
            index += 1
        self.sample_list.sort(key=lambda x: x.predict, reverse=True)

    def getRecall(self, cost):
        LR = 0
        size = int(math.ceil(cost * len(self.test_dataset.iloc[:, 0])))
        for i in range(size):
            if self.sample_list[i].label == 1:
                LR += 1
        return 1.0 * LR / self.positive_num

    def getCost(self, recall):
        L = 0
        size = int(math.ceil(recall * self.positive_num))
        for sample in self.sample_list:
            if size == 0:
                break
            if sample.label == 1:
                size -= 1
            L += 1
        if size != 0:
            L = -1
        return L
