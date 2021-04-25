from RQutils import save_excel
from supervisedlearn import SupervisedLearner


class RQ2:

    def __init__(self):
        # 需要计算cost的recall列表
        self.recall_list = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        # 每个实验的重复次数
        self.repeat_num = 10
        # 结果
        self.result_dict = {}

    def run(self, train_file: str, test_file: str, model_name: str):
        result = []
        for i in range(self.repeat_num):
            e = SupervisedLearner(
                "./datasets/report/%s.csv" % (train_file, ),
                "./datasets/report/%s.csv" % (test_file, ),
                "svm"
            )
            e.run()
            result.append([e.getCost(recall) for recall in self.recall_list])
        self.result_dict["|".join([train_file, test_file, model_name])] = result


if __name__ == "__main__":
    r = RQ2()
    for file_name in ["Ambari", "Camel", "Derby", "Wicket"]:
        for model_name in ["svm", "rf", "nb", "knn", "mlp"]:
            r.run(file_name, file_name, model_name)
    save_excel("output-RQ2.xlsx", r.result_dict, ["train", "test", "model"] + r.recall_list)
