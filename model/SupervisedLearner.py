import pandas as pd
import math
import warnings



from model import utilities
from sklearn.feature_extraction.text import CountVectorizer
from model import RFModel
from model import SvmModel
from model import dimension_reduce as dr

from model import Sample

warnings.filterwarnings('ignore')


class SupervisedLearner:
    def __init__(self):
        self.sample_list = []
        self.postive_num = 0
        self.model = RFModel()
        self.train_dataset = []
        self.test_dataset = []
    def clear(self):
        self.sample_list = []

    def clear_all(self):
        self.sample_list = []
        self.postive_num = 0
        self.model = RFModel()
        self.train_dataset = []
        self.test_dataset = []

    def read_data(self ,path):
        data = pd.read_csv(path)
        # data = data.drop(['id'], axis=1)
        # data = data.sample(frac=1)
        return data

    def init(self , train_data_path , test_data_path , model):
        if train_data_path == test_data_path:
            df = self.read_data(train_data_path)
            # 取后一半
            num_n = len(df)
            self.train_dataset = df[:int(0.5 * num_n)]
            self.test_dataset = df[int(0.5 * num_n):]
        else:
            self.train_dataset = self.read_data(train_data_path)
            self.test_dataset = self.read_data(test_data_path)
        for line in self.test_dataset.itertuples():
            if line.security == 1:
                self.postive_num += 1
        self.model = model

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
        train_content_matrix_dr, test_content_matrix_dr  = dr.selectFromLinearSVC2(train_content_matrix, train_label, test_content_matrix)

    
        train_x = train_content_matrix_dr.toarray()
        train_y = train_label
        test_x = test_content_matrix_dr.toarray()
        test_y = test_label
        utilities.data_transfer(train_x, train_y, test_x, test_y)

        self.model.train(train_x , train_y)
        predicted = self.model.predict(test_x)
        predicted = list(predicted)
        index = 0
        for line in self.test_dataset.itertuples():
            sample = Sample(line.id ,line.security , predicted[index])
            self.sample_list.append(sample)
            index += 1
        self.sample_list.sort(key=lambda x: x.predict , reverse=True)

    def getRecall(self ,cost):
        LR = 0
        size = int(math.ceil(cost * len(self.test_dataset.iloc[:,0])))
        for i in range(size):
            if self.sample_list[i].label == 1:
                LR += 1
        return LR

    def getCost(self ,recall):
        L = 0
        size = int(math.ceil(recall * self.postive_num))
        for sample in self.sample_list:
            if size == 0:
                break
            if sample.label == 1:
                size -= 1
            L += 1
        if size != 0:
            L = -1
        return L

if __name__ == "__main__":
    e = SupervisedLearner()
    e.init("..\datasets\\report\\Ambari.csv" , "..\datasets\\report\\Wicket.csv" , SvmModel())
    e.run()
     # 1. 16 2. 46 3. 332
    target = [0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0]
    print(e.postive_num)
    for num in target:
        print(e.getCost(num))
