import numpy as np
import pandas as pd
import logging
from alive_progress import alive_bar
from experiment import ExperimentFactory


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

    def build_config(self, experiment_type: str, name_prefix: str) -> None:
        """
        生成新的实验配置

        :param experiment_type: 实验类型
        :param name_prefix: 实验名称前缀，用于区分
        """
        if name_prefix is None or name_prefix.isspace():
            name_prefix = ""
        else:
            name_prefix = name_prefix + " "
        self.config_list = [{
            "name": name_prefix + "One-hot+随机",
            "type": experiment_type,
            "encoder": "onehot"
        }, {
            "name": name_prefix + "One-hot+风险策略",
            "type": experiment_type,
            "init_sample_method": "risk",
            "encoder": "onehot"
        }, {
            "name": name_prefix + "Word sequence+随机",
            "type": experiment_type,
            "encoder": "word"
        }, {
            "name": name_prefix + "Word sequence+风险策略",
            "type": experiment_type,
            "init_sample_method": "risk",
            "encoder": "word"
        }, {
            "name": name_prefix + "Tf-idf+随机",
            "type": experiment_type,
            "encoder": "tf"
        }, {
            "name": name_prefix + "Tf-idf+风险策略",
            "type": experiment_type,
            "init_sample_method": "risk",
            "encoder": "tf"
        }, {
            "name": name_prefix + "完全随机",
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
                    except Exception:
                        logging.error(config["name"] + " run error!")
                        result.append([-1] * len(self.recall_list))
                new_result_dict[config["name"]] = result
        self.result_dict.update(new_result_dict)

    def save_excel(self, file_path: str, save_original=True) -> None:
        """
        计算实验结果的平均值并保存到指定文件

        :param file_path: 文件存储位置
        :param save_original: 是否保存原始数据
        """
        average_dict = {}
        # 计算均值
        for name, data in self.result_dict.items():
            average_dict[name] = np.mean(np.array(data), axis=0).tolist()
        # 保存结果至xlsx
        with pd.ExcelWriter(file_path) as writer:
            df = pd.DataFrame(average_dict, index=self.recall_list).transpose()
            df.to_excel(writer, sheet_name="平均结果")
            if not save_original:
                return
            for name, data in self.result_dict.items():
                df = pd.DataFrame(data, columns=self.recall_list)
                df.to_excel(writer, sheet_name=name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    r = RQ1()
    r.repeat_num = 10
    for file_name in ["Ambari", "Camel", "Derby", "Chromium", "Wicket"]:
        r.build_config("extract_first", file_name + " 先提取")
        r.run(file_name + ".csv")
        r.build_config("extract_process", file_name + " 边提取")
        r.run(file_name + ".csv")
    r.save_excel('output.xlsx')
