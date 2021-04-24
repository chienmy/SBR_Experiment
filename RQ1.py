import traceback

import numpy as np
import pandas as pd
import logging
from alive_progress import alive_bar
from experiment import ExperimentFactory
from RQutils import save_excel


class RQ1:

    def __init__(self):
        # 需要计算cost的recall列表
        self.recall_list = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        # 每个实验的重复次数
        self.repeat_num = 10
        # 生成实验实例的配置列表
        self.config_list = []
        # 结果
        self.result_dict = {}

    def build_config(self, experiment_type: str, model_type: str, name_prefix: str) -> None:
        """
        生成新的实验配置

        :param experiment_type: 实验类型
        :param model_type: 模型类型
        :param name_prefix: 实验名称前缀，用于区分
        """
        self.config_list = [{
            "name": name_prefix + "|One-hot|随机|" + model_type,
            "type": experiment_type,
            "model": model_type,
            "encoder": "onehot"
        }, {
            "name": name_prefix + "|One-hot|风险策略|" + model_type,
            "type": experiment_type,
            "model": model_type,
            "init_sample_method": "risk",
            "encoder": "onehot"
        }, {
            "name": name_prefix + "|Word sequence|随机|" + model_type,
            "type": experiment_type,
            "model": model_type,
            "encoder": "word"
        }, {
            "name": name_prefix + "|Word sequence|风险策略|" + model_type,
            "type": experiment_type,
            "model": model_type,
            "init_sample_method": "risk",
            "encoder": "word"
        }, {
            "name": name_prefix + "|Tf-idf|随机|" + model_type,
            "type": experiment_type,
            "model": model_type,
            "encoder": "tf"
        }, {
            "name": name_prefix + "|Tf-idf|风险策略|" + model_type,
            "type": experiment_type,
            "model": model_type,
            "init_sample_method": "risk",
            "encoder": "tf"
        }, {
            "name": name_prefix + "|完全随机||",
            "type": "extract_random"
        }]

    def run(self, report_csv: str) -> None:
        """
        构建实验并重复运行，获取原始的实验结果

        :param report_csv: 目录下作为数据的csv文件名
        """
        new_result_dict = {}
        for config in self.config_list:
            e = ExperimentFactory.build(config)
            e.log_output = False
            e.recall_threshold = 1.0
            e.init_data_dict("./datasets", report_csv)
            result = []
            with alive_bar(e.get_recall_target(1.0) * self.repeat_num, title=config["name"]) as bar:
                for i in range(self.repeat_num):
                    try:
                        e.run(bar=bar)
                        result.append([e.get_cost(recall) for recall in self.recall_list])
                        e.clear()
                    except Exception as e:
                        traceback.print_exc()
                        logging.error(config["name"] + " run error!")
                        result.append([-1] * len(self.recall_list))
                new_result_dict[config["name"]] = result
        self.result_dict.update(new_result_dict)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    r = RQ1()
    r.repeat_num = 1
    for file_name in ["Ambari", "Camel", "Derby", "Wicket"]:
        # "svm", "rf", "nb", "knn", "mlp"
        for model_name in ["svm"]:
            r.build_config("ml", model_name, file_name)
            r.run(file_name + ".csv")
    save_excel("output.xlsx", r.result_dict, ["project", "feature", "init sample", "model"] + r.recall_list)
