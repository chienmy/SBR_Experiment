from supervisedlearn import SupervisedLearner
from model import SvmModel
import numpy as np
import pandas as pd
recall_list = [0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0]
config_list = [
    {
        "name": "Amb_Cam",
        "train_data_path": "./datasets/report/Ambari.csv",
        "test_data_path": "./datasets/report/Camel.csv",
        "model": SvmModel()
    } ,
    {
        "name": "Amb_Chr",
        "train_data_path": "./datasets/report/Ambari.csv",
        "test_data_path": "./datasets/report/Chromium.csv",
        "model": SvmModel()
    },
    {
        "name": "Amb_Der",
        "train_data_path": "./datasets/report/Ambari.csv",
        "test_data_path": "./datasets/report/Derby.csv",
        "model": SvmModel()
    },
    {
        "name": "Amb_Wic",
        "train_data_path": "./datasets/report/Ambari.csv",
        "test_data_path": "./datasets/report/Wicket.csv",
        "model": SvmModel()
    },
    {
        "name": "Cam_Amb",
        "train_data_path": "./datasets/report/Camel.csv",
        "test_data_path": "./datasets/report/Ambari.csv",
        "model": SvmModel()
    } ,
    {
        "name": "Cam_Chr",
        "train_data_path": "./datasets/report/Camel.csv",
        "test_data_path": "./datasets/report/Chromium.csv",
        "model": SvmModel()
    },
    {
        "name": "Cam_Der",
        "train_data_path": "./datasets/report/Camel.csv",
        "test_data_path": "./datasets/report/Derby.csv",
        "model": SvmModel()
    },
    {
        "name": "Cam_Wic",
        "train_data_path": "./datasets/report/Camel.csv",
        "test_data_path": "./datasets/report/Wicket.csv",
        "model": SvmModel()
    },
    {
        "name": "Chr_Amb",
        "train_data_path": "./datasets/report/Chromium.csv",
        "test_data_path": "./datasets/report/Ambari.csv",
        "model": SvmModel()
    } ,
    {
        "name": "Chr_Cam",
        "train_data_path": "./datasets/report/Chromium.csv",
        "test_data_path": "./datasets/report/Camel.csv",
        "model": SvmModel()
    },
    {
        "name":"Chr_Der",
        "train_data_path": "./datasets/report/Chromium.csv",
        "test_data_path": "./datasets/report/Derby.csv",
        "model": SvmModel()
    },
    {
        "name": "Chr_Wic",
        "train_data_path": "./datasets/report/Chromium.csv",
        "test_data_path": "./datasets/report/Wicket.csv",
        "model": SvmModel()
    },
    {
        "name": "Der_Amb",
        "train_data_path": "./datasets/report/Derby.csv",
        "test_data_path": "./datasets/report/Ambari.csv",
        "model": SvmModel()
    } ,
    {
        "name": "Der_Cam",
        "train_data_path": "./datasets/report/Derby.csv",
        "test_data_path": "./datasets/report/Camel.csv",
        "model": SvmModel()
    },
    {
        "name": "Der_Chr",
        "train_data_path": "./datasets/report/Derby.csv",
        "test_data_path": "./datasets/report/Chromium.csv",
        "model": SvmModel()
    },
    {
        "name": "Der_Wic",
        "train_data_path": "./datasets/report/Derby.csv",
        "test_data_path": "./datasets/report/Wicket.csv",
        "model": SvmModel()
    },
    {
        "name": "Wic_Amb",
        "train_data_path": "./datasets/report/Wicket.csv",
        "test_data_path": "./datasets/report/Ambari.csv",
        "model": SvmModel()
    } ,
    {
        "name": "Wic_Cam",
        "train_data_path": "./datasets/report/Wicket.csv",
        "test_data_path": "./datasets/report/Camel.csv",
        "model": SvmModel()
    },
    {
        "name": "Wic_Chr",
        "train_data_path": "./datasets/report/Wicket.csv",
        "test_data_path": "./datasets/report/Chromium.csv",
        "model": SvmModel()
    },
    {
        "name": "Wic_Der",
        "train_data_path": "./datasets/report/Wicket.csv",
        "test_data_path": "./datasets/report/Derby.csv",
        "model": SvmModel()
    },
]

def supervised_recall():
    repeat_num = 10
    result_dict = {}
    for config in config_list:
        result = []
        for i in range(repeat_num):
            e = SupervisedLearner()
            e.init(config["train_data_path"], config["test_data_path"], config["model"])
            e.run()
            result.append([e.getRecall(recall) for recall in recall_list])
        result_dict[config["name"]] = result
    average_data = []
    average_index = []
    for name, data in result_dict.items():
        average_data.append(np.mean(np.array(data), axis=0))
        average_index.append(name)
    # 保存结果至xlsx
    with pd.ExcelWriter('output.xlsx') as writer:
        df = pd.DataFrame(average_data, index=average_index, columns=recall_list)
        df.to_excel(writer, sheet_name="平均结果")
        for name, data in result_dict.items():
            df = pd.DataFrame(data, columns=recall_list)
            df.to_excel(writer, sheet_name=name)

def supervised_cost():
    repeat_num = 10
    result_dict = {}
    for config in config_list:
        result = []
        for i in range(repeat_num):
            e = SupervisedLearner()
            e.init(config["train_data_path"], config["test_data_path"], config["model"])
            e.run()
            result.append([e.getCost(recall) for recall in recall_list])
        result_dict[config["name"]] = result
    average_data = []
    average_index = []
    for name, data in result_dict.items():
        average_data.append(np.mean(np.array(data), axis=0))
        average_index.append(name)
    # 保存结果至xlsx
    with pd.ExcelWriter('output.xlsx') as writer:
        df = pd.DataFrame(average_data, index=average_index, columns=recall_list)
        df.to_excel(writer, sheet_name="平均结果")
        for name, data in result_dict.items():
            df = pd.DataFrame(data, columns=recall_list)
            df.to_excel(writer, sheet_name=name)
if __name__ == "__main__":
    supervised_recall()